"""Graph-safe forward of the FBT and its twin, and the Runner that replays it as CUDA graphs.

The FBT repo's forward (nanochat.gpt.GPT.forward) is not graph-safe: it reads the cache position with .item()
(rotary offset, SDPA fallback), slices by it, and keeps the smear embedding as a Python attribute. The functions
below compute the same function with the cache position held on the GPU, FlashAttention-2's kvcache kernel (in-place
append at `cache_seqlens`, the same window convention as the repo's fallback), and the smear input in a static
buffer. The smear is also applied to the first token of a chunk (the repo skips it there).

One call of a model is one graph replay: `Runner.capture(T)` records a call of T positions per row, and
`Runner.replay(T)` runs it on whatever the static buffers hold.
"""
import os
import subprocess
import sys

import torch

try:
    from flash_attn import flash_attn_with_kvcache as fa_kv
except ImportError as e:
    raise ImportError("manthan needs FlashAttention-2 and an NVIDIA GPU: run `uv sync` in the repository "
                      "(README.md, 'Install')") from e
from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.common import COMPUTE_DTYPE as DT
from nanochat.gpt import apply_rotary_emb, norm

__all__ = ["DT", "attn", "trunk", "inputs", "head", "Runner", "load", "prefill", "Snap", "emb_of", "plain_decode",
           "kick"]


# ----------------------------------------------------------------------------- graph-safe forward
def attn(a, x, ve, cos, sin, ws, kc, vc, pos):
    """One attention block. Appends this call's keys and values to the layer cache (kc, vc) at `pos` and attends
    causally over the cache (window `ws`). x: (B, T, D) normalised inputs; ve: value embeddings or None; cos, sin:
    rotary tables at the call's positions; pos: (B,) int32 cache lengths. Returns the block output (B, T, D)."""
    B, T, _ = x.shape
    q = a.c_q(x).view(B, T, a.n_head, a.head_dim)
    k = a.c_k(x).view(B, T, a.n_kv_head, a.head_dim)
    v = a.c_v(x).view(B, T, a.n_kv_head, a.head_dim)
    if ve is not None:
        ve = ve.view(B, T, a.n_kv_head, a.head_dim)
        gate = 3 * torch.sigmoid(a.ve_gate(x[..., :a.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * ve
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)
    q = q * 1.2
    k = k * 1.2
    y = fa_kv(q, kc, vc, k=k, v=v, cache_seqlens=pos, causal=True, window_size=ws)
    return a.c_proj(y.contiguous().view(B, T, -1))


def trunk(m, idx, xin, cos, sin, kc, vc, pos):
    """The transformer layers of model m on input states xin (B, T, D) for tokens idx (B, T), with the backout
    subtraction at the middle layer. kc, vc: (n_layer, B, S, n_kv_head, head_dim) caches, written at `pos`.
    Returns the normalised top-layer states (B, T, D): the states the FBT feeds back."""
    x0 = x = xin
    bo = m.config.n_layer // 2
    xb = None
    for i, blk in enumerate(m.transformer.h):
        x = m.resid_lambdas[i] * x + m.x0_lambdas[i] * x0
        ve = m.value_embeds[str(i)](idx).to(x.dtype) if str(i) in m.value_embeds else None
        x = x + attn(blk.attn, norm(x), ve, cos, sin, m.window_sizes[i], kc[i], vc[i], pos)
        x = x + blk.mlp(norm(x))
        if i == bo:
            xb = x
    if xb is not None:
        x = x - m.backout_lambda.to(x.dtype) * xb
    return norm(x)


def inputs(m, idx, prev, fb, fm):
    """Input states of a call. The smear mixes each token's normalised embedding with the one before it (`prev`
    for the first position). Where the feedback mask fm (B, T) is set, the FBT's latent feedback of the fed-back
    states fb (B, T, D) replaces the input. Returns (input states, normalised token embeddings)."""
    emb = m._embed_tokens(idx)
    x = norm(emb)
    pv = torch.cat((prev, x[:, :-1]), dim=1)
    gate = m.smear_lambda.to(x.dtype) * torch.sigmoid(m.smear_gate(x[:, :, :24]))
    xin = x + gate * pv
    if fb is not None:
        xin = torch.where(fm.unsqueeze(-1), m.latent_feedback(fb, emb), xin)
    return xin, x


def head(m, hid):
    """Soft-capped logits (fp32) of states hid (..., D)."""
    lg = m.lm_head(hid)[..., :m.config.vocab_size].float()
    return 15 * torch.tanh(lg / 15)


class Runner:
    """One model at batch B with a static KV cache of S positions and one CUDA graph per call shape.

    State: `kc`, `vc` the cache; `pos` (B,) int32 cache lengths on the GPU (a call writes at pos..pos+T-1 and does
    not advance pos: the caller does); `prev` (B, 1, D) the smear input, i.e. the normalised embedding of the token
    before the call's first token. `fbt` selects whether calls take fed-back states (static buffers "fb", "fm").

    A call returns (argmax next tokens (B, T), top-layer states (B, T, D), normalised token embeddings (B, T, D));
    the last embedding is the smear input for the token after the call. All graphs of a runner share one memory
    pool, so read (or clone) a replay's outputs before replaying another graph of the same runner.
    """

    def __init__(self, m, B, S, dev, fbt):
        c = m.config
        self.m, self.B, self.S, self.fbt, self.dev, self.D = m, B, S, fbt, dev, c.n_embd
        hd = c.n_embd // c.n_head
        self.kc = torch.zeros(c.n_layer, B, S, c.n_kv_head, hd, dtype=DT, device=dev)
        self.vc = torch.zeros_like(self.kc)
        self.pos = torch.zeros(B, dtype=torch.int32, device=dev)
        self.prev = torch.zeros(B, 1, c.n_embd, dtype=DT, device=dev)
        self.cos, self.sin = m.cos[0, :, 0], m.sin[0, :, 0]
        self.ar = torch.arange(S, device=dev)
        self.g, self.pool = {}, None

    def fwd(self, idx, fb=None, fm=None, last_only=False):
        """Eager call on tokens idx (B, T) at positions pos..pos+T-1 (fb, fm: fed-back states and their mask, FBT
        only; ignored for the twin). last_only: logits of the last position only. Returns (tokens, states, emb)."""
        T = idx.shape[1]
        ix = self.pos.long()[:, None] + self.ar[:T][None, :]
        cos, sin = self.cos[ix].unsqueeze(2), self.sin[ix].unsqueeze(2)
        xin, xn = inputs(self.m, idx, self.prev, fb if self.fbt else None, fm)
        hid = trunk(self.m, idx, xin, cos, sin, self.kc, self.vc, self.pos)
        lg = head(self.m, hid[:, -1:] if last_only else hid)
        return lg.argmax(-1), hid, xn

    @torch.no_grad()
    def capture(self, T, last_only=False, extra=None):
        """Record the graph for calls of T positions per row. extra(out, bufs) may append ops inside the graph.
        Warm-up writes junk to the cache at pos..pos+T-1 (never read before it is overwritten); pos is restored.
        Returns (static input buffers {"idx", "fb", "fm"}, static outputs)."""
        B, D = self.B, self.D
        bufs = {"idx": torch.zeros(B, T, dtype=torch.long, device=self.dev)}
        if self.fbt:
            bufs["fb"] = torch.zeros(B, T, D, dtype=DT, device=self.dev)
            bufs["fm"] = torch.zeros(B, T, dtype=torch.bool, device=self.dev)
        p0 = self.pos.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                o = self.fwd(bufs["idx"], bufs.get("fb"), bufs.get("fm"), last_only)
                if extra:
                    extra(o, bufs)
        torch.cuda.current_stream().wait_stream(s)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            o = self.fwd(bufs["idx"], bufs.get("fb"), bufs.get("fm"), last_only)
            if extra:
                extra(o, bufs)
        self.pos.copy_(p0)
        self.g[(T, last_only)] = (g, bufs, o)
        return bufs, o

    def bufs(self, T, last_only=False):
        """Static input buffers of the graph for T positions per row."""
        return self._graph(T, last_only)[1]

    def replay(self, T, last_only=False):
        """Replay the graph for T positions per row; returns its static outputs (tokens, states, emb)."""
        g, _, o = self._graph(T, last_only)
        g.replay()
        return o

    def _graph(self, T, last_only):
        try:
            return self.g[(T, last_only)]
        except KeyError:
            raise KeyError(f"no graph for T={T}, last_only={last_only}: call capture({T}"
                           f"{', last_only=True' if last_only else ''}) first") from None


# ----------------------------------------------------------------------------- helpers
def load(ckpt, dev, step=None, tokenizer_dir=None):
    """Load a nanochat checkpoint folder (the last step unless `step` is given) for inference.
    The tokenizer comes from `tokenizer_dir` (a folder named `tokenizer`), else from the folder's own `tokenizer/`
    (as in our released checkpoints), else from $NANOCHAT_BASE_DIR/tokenizer. Returns (model, tokenizer)."""
    step = step if step is not None else find_last_step(ckpt)
    if tokenizer_dir is None and os.path.isfile(os.path.join(ckpt, "tokenizer", "tokenizer.pkl")):
        tokenizer_dir = os.path.join(ckpt, "tokenizer")
    if tokenizer_dir is None:
        return build_model(ckpt, step, dev, "eval")[:2]
    tokenizer_dir = os.path.abspath(tokenizer_dir)
    if os.path.basename(tokenizer_dir) != "tokenizer":
        raise ValueError(f"tokenizer_dir must be a folder named 'tokenizer', not {tokenizer_dir}")
    # nanochat reads its tokenizer from $NANOCHAT_BASE_DIR/tokenizer: point the variable at this one for the call
    old = os.environ.get("NANOCHAT_BASE_DIR")
    os.environ["NANOCHAT_BASE_DIR"] = os.path.dirname(tokenizer_dir)
    try:
        return build_model(ckpt, step, dev, "eval")[:2]
    finally:
        if old is None:
            del os.environ["NANOCHAT_BASE_DIR"]
        else:
            os.environ["NANOCHAT_BASE_DIR"] = old


def emb_of(m, t, dev):
    """Normalised embedding (1, 1, D) of token id t: the smear input for the token after t."""
    return norm(m._embed_tokens(torch.tensor([[t]], device=dev)))


@torch.inference_mode()
def prefill(r, ids, bos, passes=2):
    """Prompt prefill of ids (1, P) into runner r's cache (eager calls, batch 1).

    FBT: `passes` passes over the prompt; pass 1 runs without feedback, and each later pass is fed the states of
    the pass before, shifted by one position (no feedback at position 0, nor at BOS tokens when `bos` is given). The
    cache keeps the last pass. Twin: one pass (`passes` is ignored). Leaves pos = P and prev = the last prompt
    token's embedding.
    Returns (first generated token, state at the last prompt position (1, 1, D), its smear input (1, 1, D))."""
    P = ids.shape[1]
    r.prev.zero_()
    if r.fbt:
        fm = torch.ones_like(ids, dtype=torch.bool) if bos is None else ids.ne(bos)
        fm[:, 0] = False
        fb, mask = torch.zeros(1, P, r.D, dtype=DT, device=r.dev), torch.zeros_like(ids, dtype=torch.bool)
        for _ in range(passes):
            r.pos.zero_()
            tk, h, xn = r.fwd(ids, fb, mask)
            fb, mask = torch.roll(h, 1, dims=1), fm
    else:
        r.pos.zero_()
        tk, h, xn = r.fwd(ids)
    r.pos.fill_(P)
    r.prev.copy_(xn[:, -1:])
    return int(tk[0, -1]), h[:, -1:].clone(), xn[:, -1:].clone()


class Snap:
    """Copy of runner r's cache below position P and its smear input, taken after a prefill; restore() puts the
    runner back in that state (pos = P), so several decoding runs can start from one prefill."""

    @torch.no_grad()
    def __init__(self, r, P):
        self.r, self.P = r, P
        self.k, self.v = r.kc[:, :, :P].clone(), r.vc[:, :, :P].clone()
        self.prev = r.prev.clone()

    @torch.inference_mode()
    def restore(self):
        self.r.kc[:, :, :self.P].copy_(self.k)
        self.r.vc[:, :, :self.P].copy_(self.v)
        self.r.pos.fill_(self.P)
        self.r.prev.copy_(self.prev)


@torch.inference_mode()
def plain_decode(t, tok0, h0, max_new, terminal):
    """Greedy plain decoding at batch 1 after prefill: one replay of the T=1 graph per token. FBT: each token is
    fed the state of the token before it (h0 for the first). Stops at a terminal token (not emitted) or after
    max_new tokens. Returns (tokens, number of target calls)."""
    out, tk = [], tok0
    b = t.bufs(1)
    if t.fbt:
        b["fm"].fill_(True)
        b["fb"].copy_(h0)
    nxt = torch.tensor([[tok0]], device=t.dev)
    calls = 0
    while True:
        if tk in terminal:
            break
        out.append(tk)
        if len(out) >= max_new:
            break
        b["idx"].copy_(nxt)
        o = t.replay(1)
        calls += 1
        t.pos.add_(1)
        t.prev.copy_(o[2])
        if t.fbt:
            b["fb"].copy_(o[1])
        nxt = o[0]
        tk = int(nxt)
    return out, calls


def kick():
    """Start and end one short CUDA process. On some machines (an H200 with driver 565.57.01 in our runs), a process
    that has created its CUDA context runs every kernel of a graph replay about 0.5 us slower (a decoding step of the
    863M models: 4.20 instead of 3.70 ms) until some other process initialises CUDA. Each graph capture puts all
    graphs of the process back in the slow regime, so call this after the last capture and before timing. The child
    process imports torch, which takes a few seconds; `LLM` calls this unless the environment sets MANTHAN_KICK=0."""
    subprocess.run([sys.executable, "-c", "import torch; torch.zeros(1, device='cuda'); torch.cuda.synchronize()"],
                   check=False, capture_output=True)
