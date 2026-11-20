"""Pass diagnostics: how fast Jacobi passes solve an FBT's parallel forward, against the exact recurrence.

An FBT feeds the top-layer state of position t-1 back into the input of position t. The decode loop computes these
states one token at a time: the exact recurrence. A parallel forward computes every position at once from guessed
fed-back states, and Jacobi passes repeat it with the states of the previous pass as the guesses. `diagnose` runs the
passes on fixed text from no feedback (pass 1 is an ordinary forward) and measures, for passes p = 1 .. max_passes:

  error            the state error after pass p: the relative RMS distance to the exact states over positions
                   1 .. T-1 of a text, averaged over texts (the "state error" of the post)
  differ           the fraction of positions whose most likely next token differs from that of the exact states
  reduction        error[p-1] / error[p], the factor by which pass p divides the error (p >= 2)
  gain             error_by_position averaged over positions t >= t0, after pass p over after pass p-1 (p >= 2);
                   `contraction_rate` is the gain of pass 4, one pass applied to the errors after pass 3: the
                   contraction rate along the errors
  floor            the rounding floor: the relative RMS distance between the states of the token-by-token decode
                   loop (the repo's KV-cache forward) and the exact parallel states over the same tokens. It is set by
                   the dtype (about 1e-6 in fp32, 0.6% in bf16 for the d20 FBTs)
  passes_to_floor  the first pass whose error is within floor_margin (10%) of the floor, None if none is

The exact states are the fixed point of the parallel map, solved by passes until the largest relative change of any
state is at most tol, or `cap` passes. By causality pass p makes position p-1 exact, so the fixed point is the exact
recurrence (it matches a T-pass solve and the decode loop to fp32 precision). In bf16
the passes stall at the rounding floor, so the solve runs `cap` passes; the reference is then within the floor of the
exact recurrence, and so is every later pass.

Arithmetic is the loaded model's: nanochat fixes COMPUTE_DTYPE at import from NANOCHAT_DTYPE. TF32 is switched off
while `diagnose` runs, so fp32 matmuls stay fp32.

CLI:  python -m manthan.diagnostics --ckpt <dir> --n-texts 64 --seq-len 512 --dtype fp32|bf16 --out x.json
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import torch
from nanochat.checkpoint_manager import find_last_step
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache

from manthan.data import path
from manthan.runtime import load

__all__ = ["ParallelMap", "climbmix_rows", "decode_states", "diagnose", "format_report", "rel_rms", "solve_settings",
           "top_tokens"]

DTYPES = {"fp32": "float32", "bf16": "bfloat16"}


def solve_settings(dtype):
    """Default (tol, cap) of the reference solve for a compute dtype: fp32 reaches a largest relative change of 2e-6
    within 64 passes; lower precision stalls at its rounding floor, so the solve stops after 24 passes."""
    return (2e-6, 64) if dtype == torch.float32 else (2e-3, 24)


@contextlib.contextmanager
def _no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


class ParallelMap:
    """The parallel forward of one batch of texts, as a map from fed-back states to new states.

    tokens: (B, T) token ids on the model's device. Position t is fed the state of position t-1, except at t = 0 and
    at BOS tokens (document boundaries), as in the repo's decode loop."""

    def __init__(self, model, tokens, bos):
        self.model, self.x = model, tokens
        self.te = model._embed_tokens(tokens)
        self.ti = model._prepare_token_inputs(self.te, None)
        self.fm = tokens.ne(bos)
        self.fm[:, 0] = False

    def plain(self):
        """Pass 1: an ordinary forward without feedback. Returns the top-layer states (B, T, D)."""
        return self.model._run_trunk(self.x, self.ti, None)

    def step(self, H):
        """One Jacobi pass: every position t is fed H[:, t-1]. Returns the new top-layer states (B, T, D)."""
        fused = self.model.latent_feedback(H[:, :-1].to(self.te.dtype), self.te[:, 1:])
        xin = torch.where(self.fm.unsqueeze(-1), torch.cat((self.ti[:, :1], fused), dim=1), self.ti)
        return self.model._run_trunk(self.x, xin, None)

    def solve(self, H, tol, cap):
        """Passes from the states H until the largest relative change of any state is at most tol, or cap passes.
        Returns (states, passes run, last largest relative change)."""
        n, rel = 0, float("inf")
        while n < cap and rel > tol:
            Hn = self.step(H)
            rel = ((Hn - H).float().norm(dim=-1) / Hn.float().norm(dim=-1)).max().item()
            H, n = Hn, n + 1
        return H, n, rel


def decode_states(model, tokens, bos):
    """Top-layer states of the token-by-token decode loop: the repo's KV-cache forward, one token per call, each token
    fed the state of the token before it (none at t = 0 and at BOS). tokens: (1, T). Returns (1, T, D)."""
    cfg, T = model.config, tokens.shape[1]
    kv = KVCache(batch_size=1, num_heads=cfg.n_kv_head, seq_len=T + 8, head_dim=cfg.n_embd // cfg.n_head,
                 num_layers=cfg.n_layer, device=tokens.device, dtype=COMPUTE_DTYPE)
    fm = tokens.ne(bos)
    fm[:, 0] = False
    _, h = model.forward(tokens[:, :1], kv_cache=kv, return_hidden=True)
    hs = [h]
    for t in range(1, T):
        _, h = model.forward(tokens[:, t:t + 1], kv_cache=kv, feedback_hidden=h, feedback_mask=fm[:, t:t + 1],
                             return_hidden=True)
        hs.append(h)
    return torch.cat(hs, dim=1)


def rel_rms(H, ref):
    """Relative RMS distance of states H to ref, per text, over positions 1 .. T-1 (both (B, T, D)). Returns (B,)."""
    a, b = H[:, 1:].float(), ref[:, 1:].float()
    return ((a - b).pow(2).sum((1, 2)) / b.pow(2).sum((1, 2)).clamp_min(1e-30)).sqrt()


def top_tokens(model, H, chunk=4):
    """Most likely next token at every position, read out from states H (B, T, D). Returns (B, T) token ids."""
    return torch.cat([model._project_and_loss(H[i:i + chunk], None, None).argmax(-1)
                      for i in range(0, H.shape[0], chunk)])


def climbmix_rows(tok, n_texts, seq_len, parquet, min_tokens=2049):
    """The first n_texts documents of a ClimbMix parquet shard that have at least min_tokens tokens (BOS included),
    each cut to its first seq_len tokens. Returns (n_texts, seq_len) int64 token ids, every row starting with BOS.

    With min_tokens = 2049 and the prefilter of more than 3 characters per required token, this selects the same
    documents as the post's runs (their cached rows), for any seq_len up to 2049."""
    import pyarrow.parquet as pq
    assert seq_len <= min_tokens, "seq_len must not exceed min_tokens"
    bos, rows, pf = tok.get_bos_token_id(), [], pq.ParquetFile(parquet)
    for rg in range(pf.num_row_groups):
        texts = [t for t in pf.read_row_group(rg).column("text").to_pylist() if len(t) > 3 * min_tokens]
        rows += [ids[:seq_len] for ids in tok.encode(texts, prepend=bos) if len(ids) >= min_tokens]
        if len(rows) >= n_texts:
            break
    if len(rows) < n_texts:
        raise ValueError(f"{parquet} has only {len(rows)} documents of >= {min_tokens} tokens")
    return torch.tensor(rows[:n_texts], dtype=torch.long)


def diagnose(model, tokens, max_passes=6, batch_size=16, floor_texts=4, bos=None, tol=None, cap=None, t0=64,
             floor_margin=0.1):
    """Error of Jacobi passes against the exact recurrence on fixed text (see the module docstring).

    model: a nanochat FBT (eval mode, on its device). tokens: (N, T) token ids, one text per row, starting with BOS
    (e.g. `climbmix_rows`). bos: the BOS id (default: tokens[0, 0]). Texts run batch_size at a time; the floor uses
    the decode loop on the first floor_texts texts (0 skips it). tol, cap: the reference solve (default
    `solve_settings`). t0: first position of the gain average. floor_margin: passes_to_floor counts a pass whose
    error is at most (1 + floor_margin) x floor.

    Returns a JSON-ready dict: settings; reference (passes and last largest relative change of the solve, per batch);
    error, error_se (standard error over texts), differ (lists over passes 1 .. max_passes); reduction and gain
    (passes 2 .. max_passes); contraction_rate (gain of pass 4, None if max_passes < 4); floor, floor_differ (None
    if floor_texts = 0); passes_to_floor; error_by_position (max_passes x T, mean over texts of the relative error
    at each position); time_s.
    """
    dev = next(model.parameters()).device
    bos = int(tokens[0, 0]) if bos is None else bos
    dtol, dcap = solve_settings(COMPUTE_DTYPE)
    tol, cap = (dtol if tol is None else tol), (dcap if cap is None else cap)
    N, T = tokens.shape
    P = max_passes
    err = torch.zeros(P, N, dtype=torch.float64)
    differ = torch.zeros(P, N, dtype=torch.float64)
    field = torch.zeros(P, T, dtype=torch.float64)
    solves, floor_d, floor_differ = [], [], []
    t_start = time.time()
    with torch.no_grad(), _no_tf32():
        for i in range(0, N, batch_size):
            x = tokens[i:i + batch_size].to(dev)
            B = x.shape[0]
            S = ParallelMap(model, x, bos)
            H = S.plain()
            ref, n, rel = S.solve(H, tol, cap)
            solves.append({"passes": n, "max_rel_change": rel})
            ref_top = top_tokens(model, ref)
            ref_norm = ref.float().norm(dim=-1).clamp_min(1e-30)
            for p in range(P):
                if p:
                    H = S.step(H)
                err[p, i:i + B] = rel_rms(H, ref).double().cpu()
                differ[p, i:i + B] = (top_tokens(model, H) != ref_top)[:, 1:].double().mean(1).cpu()
                field[p] += ((H.float() - ref.float()).norm(dim=-1) / ref_norm).sum(0).double().cpu()
            if i == 0:
                for d in range(min(floor_texts, B)):
                    Hd = decode_states(model, x[d:d + 1], bos)
                    floor_d.append(rel_rms(Hd, ref[d:d + 1]).item())
                    floor_differ.append((top_tokens(model, Hd) != ref_top[d:d + 1])[:, 1:].double().mean().item())
            del ref, H

    error = err.mean(1).tolist()
    error_se = (err.std(1) / math.sqrt(N)).tolist() if N > 1 else [0.0] * P
    field = field / N
    M = field[:, t0:].mean(1).tolist()
    ratio = lambda a, b: a / b if b > 0 else float("inf")
    gain = [ratio(M[p], M[p - 1]) for p in range(1, P)]
    floor = sum(floor_d) / len(floor_d) if floor_d else None
    to_floor = None
    if floor is not None:
        to_floor = next((p + 1 for p in range(P) if error[p] <= (1 + floor_margin) * floor), None)
    return {
        "settings": {"dtype": str(COMPUTE_DTYPE), "n_texts": N, "seq_len": T, "max_passes": P, "batch_size": batch_size,
                     "floor_texts": len(floor_d), "tol": tol, "cap": cap, "t0": t0, "floor_margin": floor_margin,
                     "bos": bos},
        "reference": solves,
        "error": error,
        "error_se": error_se,
        "differ": differ.mean(1).tolist(),
        "reduction": [ratio(error[p - 1], error[p]) for p in range(1, P)],
        "gain": gain,
        "contraction_rate": gain[2] if P >= 4 else None,
        "floor": floor,
        "floor_differ": sum(floor_differ) / len(floor_differ) if floor_differ else None,
        "passes_to_floor": to_floor,
        "error_by_position": field.tolist(),
        "time_s": round(time.time() - t_start, 1),
    }


def format_report(res):
    """A short text table of a `diagnose` result."""
    lines = [f"{'pass':>4} | {'state error':>11} | {'tokens differ':>13} | {'reduction':>9} | {'gain':>6}"]
    for p, (e, d) in enumerate(zip(res["error"], res["differ"]), start=1):
        red = f"{res['reduction'][p - 2]:9.2f}" if p > 1 else " " * 9
        g = f"{res['gain'][p - 2]:6.3f}" if p > 1 else " " * 6
        lines.append(f"{p:4d} | {100 * e:10.4g}% | {100 * d:12.4g}% | {red} | {g}")
    if res["floor"] is not None:
        lines.append(f"floor (decode loop vs parallel, {res['settings']['floor_texts']} texts): {100 * res['floor']:.4g}%"
                     f"  tokens differ {100 * res['floor_differ']:.4g}%")
        lines.append(f"passes to the floor (within {100 * res['settings']['floor_margin']:.0f}%): {res['passes_to_floor']}")
    if res["contraction_rate"] is not None:
        lines.append(f"contraction rate along the errors (gain of pass 4): {res['contraction_rate']:.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(prog="python -m manthan.diagnostics",
                                 description="Error of Jacobi passes against the exact recurrence on ClimbMix text.")
    ap.add_argument("--ckpt", default=None, help="checkpoint folder (default: $FBT_CKPT)")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: the last)")
    ap.add_argument("--data", default=None, help="ClimbMix validation parquet (default: $CLIMBMIX_VAL)")
    ap.add_argument("--n-texts", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    ap.add_argument("--max-passes", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--floor-texts", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.environ.get("NANOCHAT_DTYPE") != DTYPES[args.dtype]:
        # nanochat fixes its compute dtype at import: restart with it set
        env = {**os.environ, "NANOCHAT_DTYPE": DTYPES[args.dtype]}
        os.execve(sys.executable, [sys.executable, "-m", "manthan.diagnostics", *sys.argv[1:]], env)
    ckpt, data = args.ckpt or path("FBT_CKPT"), args.data or path("CLIMBMIX_VAL")
    step = args.step if args.step is not None else find_last_step(ckpt)
    model, tok = load(ckpt, torch.device("cuda"), step)
    rows = climbmix_rows(tok, args.n_texts, args.seq_len, data)
    res = diagnose(model, rows, max_passes=args.max_passes, batch_size=args.batch_size, floor_texts=args.floor_texts,
                   bos=tok.get_bos_token_id())
    res["settings"].update(ckpt=ckpt, step=step, data=data)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(format_report(res))
    print(f"{res['time_s']}s -> {args.out}")


if __name__ == "__main__":
    main()
