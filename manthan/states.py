"""Stored top-layer states in 8 bits, and the one-pass warm-start rebuild of an evicted KV cache.

The FBT feeds each token's top-layer state into the input of the next token, so the KV cache of a context depends
on these states. A cache rebuilt from the tokens alone must first recover them: Jacobi passes from no feedback (a
cold start; pass 1 is a plain prefill and each later pass makes the states more exact; the FBT authors use three).
Prefill and decoding compute the states anyway. We keep them in 8 bits per dimension (1,280 bytes per token for the
d20 models, 1.25% of the KV cache) and rebuild the cache with one pass fed these stored states (a warm start).

Quantiser: each dimension is standardised with a mean and a standard deviation fitted once per model (`fit_stats`),
clipped at +-6 standard deviations, and mapped to 2^bits uniform bins. One scale per vector fails: one dimension
holds most of the energy of these states (83.6% for the base FBT, 68.0% for the fine-tuned one), so a scale set by
the largest entry leaves too few levels for the other dimensions (state error after one pass 0.73% instead of 0.071%,
fine-tuned FBT, fp32).

Store convention: slot t of a row holds the state fed to position t+1 when the cache was built. For decoding this is
the state the model computed at position t; for a prefill of several passes, the state of the pass before the last.
A warm rebuild of L positions reads slots 0..L-2 and writes slot L-1, so it reproduces the cache that decoding used,
up to the 8-bit rounding.
"""
import math
import os

import torch

from manthan.runtime import DT

__all__ = ["Quantizer", "quantize", "dequantize", "feedback_mask", "exact_states", "fit_stats", "save_stats",
           "load_stats", "FeedbackCache", "prefill_store", "capture_rebuild", "rebuild"]

STATS_FILE = "feedback_stats.pt"        # the quantiser statistics in a checkpoint folder


# ----------------------------------------------------------------------------- quantiser
class Quantizer:
    """Uniform quantiser per dimension: z = (h - mu) / sd, clipped to [-clip, clip], is coded as the index of its
    bin among 2^bits uniform bins on [-clip, clip], and decoded as the bin's centre. mu, sd: (D,) statistics of the
    model's states (fit_stats). Both maps are one multiply-add per entry (code = h * a + b, truncated and clamped to
    [0, 2^bits - 1]; state = code * s + t), so few kernels run around a rebuild or inside a decoding graph.
    The round trip is within clip * sd / 2^bits of h wherever |z| <= clip."""

    def __init__(self, mu, sd, bits=8, clip=6.0):
        assert 1 <= bits <= 8, "codes are stored as uint8"
        self.mu, self.sd, self.bits, self.clip = mu.float(), sd.float(), bits, clip
        self.code_dim, self.code_dtype = mu.numel(), torch.uint8    # one byte per dimension (the codec interface)
        self.n = n = 2 ** bits
        self.a = n / (2 * clip * self.sd)
        self.b = n / 2 - self.mu * self.a
        self.s = 2 * clip * self.sd / n
        self.t = self.mu + (1 / n - 1) * clip * self.sd

    def encode(self, h):
        """uint8 codes of states h (..., D)."""
        return torch.addcmul(self.b, h.float(), self.a).clamp_(0, self.n - 1).to(torch.uint8)

    def decode(self, codes, out=None):
        """float32 states of codes (..., D), or written into `out` (any float dtype, e.g. a graph's static feedback
        buffer) and returned."""
        h = torch.addcmul(self.t, codes.float(), self.s)
        if out is None:
            return h
        out.copy_(h)
        return out


def quantize(h, mu, sd, bits=8, clip=6.0):
    """uint8 codes of states h (..., D) with the per-dimension quantiser of statistics mu, sd (see Quantizer)."""
    return Quantizer(mu, sd, bits, clip).encode(h)


def dequantize(codes, mu, sd, bits=8, clip=6.0, out=None):
    """States of codes (..., D): float32, or written into `out` and returned (see Quantizer)."""
    return Quantizer(mu, sd, bits, clip).decode(codes, out)


# ----------------------------------------------------------------------------- per-model statistics
def feedback_mask(ids, bos=None, out=None):
    """(B, T) bool mask of the positions that take a fed-back state: every position except the first of each row
    and, when `bos` is given, BOS tokens (a document start has no state before it). Written into `out` if given."""
    if out is None:
        out = torch.empty_like(ids, dtype=torch.bool)
    if bos is None:
        out.fill_(True)
    else:
        torch.ne(ids, bos, out=out)
    out[:, 0] = False
    return out


@torch.no_grad()
def exact_states(m, ids, bos, tol=None, max_passes=None):
    """Top-layer states of the exact token-by-token recurrence on tokens ids (B, T), found as the fixed point of the
    parallel map: pass 1 without feedback, then passes fed the states of the pass before, shifted by one position
    (no feedback at position 0 or at BOS), until the largest relative change of a state falls below `tol`. By
    causality position j is exact after j passes, so the fixed point is the sequential answer.

    Uses the repo's eager forward (GPT.forward without a cache), so it runs in any compute dtype. Defaults follow the
    post's runs: tol 2e-6 and at most 64 passes in fp32 (NANOCHAT_DTYPE=float32), else tol 2e-3 and at most 24
    passes (bf16 cannot get closer). In fp32 the largest change settles near 6e-6 from rounding (16 texts of 512
    tokens), so the cap ends the loop. Returns (states (B, T, D), passes run, last relative change).
    """
    fp32 = DT == torch.float32
    tol = tol if tol is not None else (2e-6 if fp32 else 2e-3)
    cap = max_passes if max_passes is not None else (64 if fp32 else 24)
    fm = feedback_mask(ids, bos)
    _, H = m.forward(ids, return_hidden=True)
    n, rel = 1, math.inf
    while n < cap and rel > tol:
        _, Hn = m.forward(ids, feedback_hidden=torch.roll(H, 1, dims=1), feedback_mask=fm, return_hidden=True)
        rel = ((Hn - H).float().norm(dim=-1) / Hn.float().norm(dim=-1)).max().item()
        H, n = Hn, n + 1
    return H, n, rel


@torch.no_grad()
def fit_stats(m, token_batches, bos, tol=None, max_passes=None):
    """Per-dimension mean and standard deviation of the exact states (`exact_states`) of model m, over every
    position but the first of each row, for an iterable of (B, T) token-id batches on m's device.

    The post's statistics (`feedback_stats.pt` in the FBT checkpoint) come from 64 OpenMathInstruct-2 chat texts of
    512 tokens in fp32. Returns (mu, sd), float32 (D,) on m's device; sd is clamped at 1e-8."""
    s = ss = None
    n = 0
    for ids in token_batches:
        H, _, _ = exact_states(m, ids, bos, tol, max_passes)
        X = H[:, 1:].double().reshape(-1, H.shape[-1])
        s = X.sum(0) if s is None else s + X.sum(0)
        ss = (X * X).sum(0) if ss is None else ss + (X * X).sum(0)
        n += X.shape[0]
    mu = s / n
    var = (ss - n * mu * mu) / (n - 1)
    return mu.float(), var.clamp_min(0).sqrt().float().clamp_min(1e-8)


def save_stats(path, mu, sd, **meta):
    """Save the quantiser statistics (mu, sd) and any metadata (e.g. checkpoint, step, texts) with torch.save."""
    torch.save({"mu": mu.detach().float().cpu(), "sd": sd.detach().float().cpu(), **meta}, path)


def load_stats(path=None, device="cpu"):
    """(mu, sd) float32 (D,) on `device`. `path` is a file of `save_stats`; a checkpoint folder or a Hugging Face
    repo id that holds one as `feedback_stats.pt` (the FBT checkpoint does); or a states file of the post's runs
    ({"H": (N, T, D) exact states}), whose statistics are taken over positions 1.. of every row, as the post's
    quantiser did. path defaults to the environment variable FBT_STATS (or the layout under FBT_ROOT, see
    manthan.data.path)."""
    if path is None:
        from manthan.data import path as env_path
        path = env_path("FBT_STATS")
    path = os.fspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, STATS_FILE)
    elif not os.path.exists(path) and not path.endswith(".pt"):    # a Hugging Face repo id
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=path, filename=STATS_FILE)
    blob = torch.load(path, map_location="cpu")
    if "mu" in blob:
        mu, sd = blob["mu"].float(), blob["sd"].float()
    else:
        X = blob["H"][:, 1:].float().reshape(-1, blob["H"].shape[-1])
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-8)
    return mu.to(device), sd.to(device)


# ----------------------------------------------------------------------------- the store
class FeedbackCache:
    """The feedback cache: the fed-back states of B rows x S positions, stored as codes on the GPU next to the KV
    cache (on `device`, else on mu's device when it is a GPU, else on the current CUDA device).

    Slot t of a row holds the state fed to position t+1 (module docstring). `write` is graph-safe when given the
    positions as a GPU tensor, so decoding can store each new state inside its graph:
        cache = FeedbackCache(B, S, mu, sd)
        runner.capture(1, extra=cache.hook(runner))   # every replay also stores the new states at runner.pos

    codec: how states become codes. The default is the 8-bit per-dimension Quantizer(mu, sd, bits, clip). Any object
    with encode(h (..., D)) -> codes (..., code_dim), decode(codes, out=None) -> states (written into `out` when
    given), and attributes code_dim and code_dtype can take its place (then mu and sd are not needed).
    """

    def __init__(self, B, S, mu=None, sd=None, bits=8, clip=6.0, device=None, codec=None):
        if codec is None and (mu is None or sd is None):
            raise ValueError("give the state statistics (mu, sd) for the default 8-bit codec, or a codec")
        if device is not None:
            dev = torch.device(device)
        elif mu is not None and mu.is_cuda:
            dev = mu.device
        else:
            dev = torch.device("cuda")
        self.q = codec if codec is not None else Quantizer(mu.to(dev), sd.to(dev), bits, clip)
        self.B, self.S, self.C = B, S, self.q.code_dim
        self.codes = torch.zeros(B, S, self.C, dtype=self.q.code_dtype, device=dev)
        self._flat = self.codes.view(B * S, self.C)
        self._base = torch.arange(B, device=dev) * S
        self._ar = torch.arange(S, device=dev)

    @torch.inference_mode()
    def write(self, h, pos, row=None):
        """Store states h (B, T, D) in slots pos..pos+T-1 of each row. pos: an int, or a (B,) int tensor of
        per-row start slots read on the GPU (no host sync: usable inside a CUDA graph). With `row` (an int), h is
        (1, T, D) and only that row is written, at the int slot pos."""
        T = h.shape[1]
        if isinstance(pos, int):
            self.codes[slice(None) if row is None else slice(row, row + 1), pos:pos + T] = self.q.encode(h)
            return
        idx = (self._base + pos.long())[:, None] + self._ar[:T]
        self._flat.index_copy_(0, idx.reshape(-1), self.q.encode(h).reshape(-1, self.C))

    @torch.inference_mode()
    def read(self, start, n, out=None):
        """Decoded states of slots start..start+n-1 of every row, (B, n, D); written into `out` when given."""
        return self.q.decode(self.codes[:, start:start + n], out)

    def hook(self, runner):
        """`extra` for runner.capture(T, extra=...): after the call, store its T new states at the runner's
        positions (runner.pos is the call's first position: calls do not advance it). The capture's warm-up writes
        junk to the slots at the capture-time positions, so capture before storing anything there."""
        def extra(o, bufs):
            self.write(o[1], runner.pos)
        return extra

    @property
    def bytes_per_token(self):
        """Bytes stored per position (code_dim codes; the 8-bit codec's codes of fewer bits still take a byte)."""
        return self.C * self.codes.element_size()


# ----------------------------------------------------------------------------- filling and rebuilding the cache
@torch.inference_mode()
def prefill_store(r, ids, bos, store, row=0, passes=2):
    """Prompt prefill of ids (1, P) into FBT runner r (batch 1), as manthan.runtime.prefill, that also writes the
    states the cache was built from into `row` of `store`: slots 0..P-2 hold the states fed in the last pass (the
    pass before it), slot P-1 the last pass's state at P-1 (fed to the first generated token).
    Leaves pos = P and prev = the last prompt token's embedding.
    Returns (first generated token, state at the last prompt position (1, 1, D), its smear input (1, 1, D))."""
    P = ids.shape[1]
    r.prev.zero_()
    fm = feedback_mask(ids, bos)
    fb, mask = torch.zeros(1, P, r.D, dtype=DT, device=r.dev), torch.zeros_like(ids, dtype=torch.bool)
    fed = None
    for p in range(passes):
        r.pos.zero_()
        tk, h, xn = r.fwd(ids, fb, mask)
        if p < passes - 1:
            fed = h
            fb, mask = torch.roll(h, 1, dims=1), fm
    if fed is None:         # a single pass is fed nothing: keep its own states
        fed = h
    store.write(fed[:, :P - 1], 0, row=row)
    store.write(h[:, -1:], P - 1, row=row)
    r.pos.fill_(P)
    r.prev.copy_(xn[:, -1:])
    return int(tk[0, -1]), h[:, -1:].clone(), xn[:, -1:].clone()


@torch.no_grad()
def capture_rebuild(r, L):
    """Capture the graph of one rebuild call of L positions per row (logits of the last position only) for runner
    r, recorded on a view of the first L cache positions. With the whole cache of S > L positions, FlashAttention's
    append kernel sizes its KV splits by S, and each rebuild call at L 512, B 1 took 0.5 ms longer
    in our runs. The capture's warm-up overwrites cache positions 0..L-1: capture before serving.
    Returns the graph's static input buffers."""
    kc, vc, p0 = r.kc, r.vc, r.pos.clone()
    r.kc, r.vc = kc[:, :, :L], vc[:, :, :L]
    r.pos.zero_()
    try:
        b, _ = r.capture(L, last_only=True)
    finally:
        r.kc, r.vc = kc, vc
        r.pos.copy_(p0)
    return b


@torch.inference_mode()
def rebuild(r, ids, store=None, passes=3, bos=None):
    """Rebuild runner r's KV cache at positions 0..L-1 of every row from context tokens ids (B, L).

    Warm start (FBT, `store` given): one pass fed the stored states of slots 0..L-2 (dequantised on the GPU into
    the call's feedback buffer); the new state of position L-1 is then stored in slot L-1.
    Cold start (FBT, no store): `passes` passes from no feedback; pass 1 is plain and each later pass is fed the
    states of the pass before, shifted by one position.
    Twin (r.fbt False): one plain pass (a prefill).
    No feedback at position 0, nor at BOS tokens when `bos` is given. Uses the graph of capture_rebuild(r, L) when
    it exists, else eager calls. Leaves r.pos = L and r.prev = the last token's embedding, as decoding expects.

    Returns (next tokens (B,), states at position L-1 (B, 1, D), states of the last pass (B, L, D)): the token to
    decode next and the state fed with it. The full states are the graph's static output when a graph was used:
    valid until the runner's next replay.
    """
    B, L = ids.shape
    if store is not None and not r.fbt:
        raise ValueError("stored states are for the FBT; the twin rebuilds with one plain pass")
    if (L, True) in r.g:
        b = r.bufs(L, True)
        b["idx"].copy_(ids)

        def call():
            return r.replay(L, True)
    else:
        b = {"idx": ids}
        if r.fbt:
            b["fb"] = torch.zeros(B, L, r.D, dtype=DT, device=r.dev)
            b["fm"] = torch.zeros(B, L, dtype=torch.bool, device=r.dev)

        def call():
            return r.fwd(b["idx"], b.get("fb"), b.get("fm"), last_only=True)
    r.prev.zero_()
    r.pos.zero_()
    if not r.fbt:
        o = call()
    elif store is not None:
        store.read(0, L - 1, out=b["fb"][:, 1:])
        feedback_mask(ids, bos, out=b["fm"])
        o = call()
        store.write(o[1][:, -1:], L - 1)
    else:
        b["fm"].zero_()
        o = call()
        for p in range(passes - 1):
            b["fb"][:, 1:].copy_(o[1][:, :-1])
            if p == 0:
                feedback_mask(ids, bos, out=b["fm"])
            r.pos.zero_()
            o = call()
    r.pos.fill_(L)
    r.prev.copy_(o[2][:, -1:])
    return o[0][:, -1].clone(), o[1][:, -1:].clone(), o[1]
