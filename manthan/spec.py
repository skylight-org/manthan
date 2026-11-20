"""Greedy speculative decoding at batch 1 with every model call one CUDA-graph replay.

A small drafter proposes a block of K tokens [pending, d1 .. d_{K-1}]; the target verifies the block and accepts
drafts up to the first one that differs from its own argmax, then takes its own token there. The FBT's state at
each position depends on the state before it, so one parallel call over the block is one Jacobi pass of that
recurrence. Verification comes in four kinds:

  plain       no drafter: one target call per token (the reference).
  token       FBT, exact: one call of one position per drafted token, each fed the exact state before it, up to the
              first mismatch. Outputs are identical to plain decoding.
  rounds      `passes` parallel calls over the block per round (pass 1 without feedback inside the block, each later
              pass fed the states of the pass before). The twin's one pass is exact up to rounding.
  pipelined   FBT: one call per round; the further passes of the accepted tokens run inside the next round's call,
              ahead of the new block, until each position has had `passes` passes (or its exact state).

The graph sizes each kind needs are given by `graph_sizes`; `generate` captures them, prefills both models and decodes.
"""
import time

import torch

from .runtime import emb_of, plain_decode, prefill

__all__ = ["Drafter", "verify_tok", "spec_rounds", "spec_pipe", "finish", "graph_sizes", "ensure_graphs", "decode",
           "generate", "MODES"]

MODES = ("plain", "token", "rounds", "pipelined")


class Drafter:
    """A drafter model's Runner (twin-style, T=1 graph) that drafts blocks greedily, one replay per token."""

    def __init__(self, r):
        self.r = r
        self.b = r.bufs(1)
        self.calls = 0

    def draft(self, catch, head_tok, K):
        """Feed the catch-up tokens `catch` (tokens the drafter has not seen yet), then draft K-1 tokens after
        head_tok. Returns the block [head_tok, d1 .. d_{K-1}] as a (1, K) tensor on the GPU."""
        r, b = self.r, self.b
        for t in catch:
            b["idx"].fill_(t)
            o = r.replay(1)
            self.calls += 1
            r.pos.add_(1)
            r.prev.copy_(o[2])
        blk = torch.empty(1, K, dtype=torch.long, device=r.dev)
        blk[0, 0] = head_tok
        b["idx"].fill_(head_tok)
        for j in range(K - 1):
            o = r.replay(1)
            self.calls += 1
            r.pos.add_(1)
            r.prev.copy_(o[2])
            blk[:, j + 1:j + 2].copy_(o[0])
            b["idx"].copy_(o[0])
        return blk

    def set(self, L, prev_tok):
        """Rewind the drafter to cache length L, where prev_tok is the token at position L-1."""
        self.r.pos.fill_(L)
        self.r.prev.copy_(emb_of(self.r.m, prev_tok, self.r.dev))


def verify_tok(t, blk, h, L, K):
    """Exact verification of a block (FBT target t): one call of one position per token, each fed the exact state
    of the token before it (h for the first, at cache length L), up to the first drafted token that differs from
    the target's. Returns (accepted drafts a, the target's token after them, the state there, target calls)."""
    b1 = t.bufs(1)
    b1["fm"].fill_(True)
    d = blk[0].tolist()
    for j in range(K):
        b1["idx"].copy_(blk[:, j:j + 1])
        b1["fb"].copy_(h)
        t.pos.fill_(L + j)
        o = t.replay(1)
        p = int(o[0])
        h = o[1].clone()
        if j == K - 1 or p != d[j + 1]:
            return j, p, h, j + 1


def _event(ev):
    if ev is None:
        return None
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def spec_rounds(t, dr, K, P, tok0, h0, max_new, terminal, passes=1, exact=False, ev=None):
    """Speculative decoding in rounds after prefill (cache length P, first token tok0, FBT state h0 at P-1).
    Each round drafts a block of K and verifies it with `passes` calls over the block (exact=True: verify_tok).
    Stops at a terminal token or after max_new tokens. ev: optional list that receives one (start, drafted,
    verified) CUDA-event triple per round. Returns (tokens, rounds, accepted drafts, target calls)."""
    assert t.fbt or (passes == 1 and not exact), "the twin verifies with one pass"
    L, pending, h = P, tok0, h0
    out, catch = [], []
    blocks = acc = calls = 0
    if pending in terminal:
        return out, 0, 0, 0
    b = None if exact else t.bufs(K)
    while True:
        e0 = _event(ev)
        blk = dr.draft(catch, pending, K)
        e1 = _event(ev)
        if exact:
            a, new_pending, h, c = verify_tok(t, blk, h, L, K)
            calls += c
            d = blk[0].tolist()
        else:
            b["idx"].copy_(blk)
            if t.fbt:
                b["fb"].zero_()
                b["fb"][:, :1].copy_(h)
                b["fm"].zero_()
                b["fm"][:, 0] = True
            t.pos.fill_(L)
            o = t.replay(K)
            calls += 1
            for _ in range(passes - 1):
                b["fb"][:, 1:].copy_(o[1][:, :-1])
                b["fm"].fill_(True)
                t.pos.fill_(L)
                o = t.replay(K)
                calls += 1
            both = torch.cat((o[0], blk), dim=0).tolist()
            pred, d = both[0], both[1]
            a = next((j for j in range(K - 1) if pred[j] != d[j + 1]), K - 1)
            new_pending = pred[a]
            if t.fbt:
                h = o[1][:, a:a + 1].clone()
            else:
                t.prev.copy_(o[2][:, a:a + 1])
        if ev is not None:
            ev.append((e0, e1, _event(ev)))
        blocks += 1
        toks = d[:a + 1]
        cut = next((j for j in range(1, a + 1) if toks[j] in terminal), None)
        if cut is not None:
            acc += cut
            out.extend(toks[:cut])
            done = True
        else:
            acc += a
            out.extend(toks)
            done = new_pending in terminal
        if len(out) >= max_new:
            out, done = out[:max_new], True
        if done:
            return out, blocks, acc, calls
        L += a + 1
        pending = new_pending
        if a <= K - 2:
            dr.set(L, d[a])
            catch = []
        else:
            dr.set(L - 1, d[K - 2])
            catch = [d[K - 1]]


def _finish(fin, terminal, max_new):
    """Committed tokens fin cut at the first terminal (after position 0) or at max_new; (tokens, done)."""
    cut = next((j for j in range(1, len(fin)) if fin[j] in terminal), None)
    if cut is not None:
        return fin[:min(cut, max_new)], True
    if len(fin) - 1 >= max_new:
        return fin[:max_new], True
    return fin, False


finish = _finish


def spec_pipe(t, dr, lag, K, P, tok0, h0, max_new, terminal, ev=None):
    """Pipelined passes (FBT target t). Each round is one call of T = pending + K positions: the accepted but not
    yet converged positions (each fed the states of its previous pass) followed by the new drafted block. A position
    is committed to the cache frontier once it has had `lag` passes or was computed from exact states; a mismatch
    inside the pending part rolls back the tokens after it. Returns (tokens, drafted rounds, accepted drafts,
    target calls, rollbacks)."""
    assert t.fbt, "pipelined passes need the FBT"
    dev = t.dev
    seq = [tok0]
    Lc, hc = P, h0
    st, npass, exact = None, [], []
    catch, have = [], P
    blocks = acc = calls = rollbacks = 0
    if seq[0] in terminal:
        return [], 0, 0, 0, 0
    while True:
        n_p = len(npass)
        off = Lc - P
        hd = seq[-1]
        draft_ok = not any(x in terminal for x in seq[1:]) and len(seq) <= max_new
        e0 = _event(ev)
        if draft_ok:
            blk = dr.draft(catch, hd, K)
            catch, have = [], Lc + n_p + K - 1
            blocks += 1
        elif n_p == 0:
            return _finish(seq, terminal, max_new)[0], blocks, acc, calls, rollbacks
        else:
            blk = None
        e1 = _event(ev)
        N = n_p + (K if blk is not None else 0)
        nf = n_p + (1 if blk is not None else 0)
        b = t.bufs(N)
        if n_p:
            b["idx"][:, :n_p].copy_(torch.tensor([seq[off:off + n_p]], device=dev), non_blocking=True)
        if blk is not None:
            b["idx"][:, n_p:].copy_(blk)
        b["fb"].zero_()
        b["fb"][:, :1].copy_(hc)
        if nf > 1:
            b["fb"][:, 1:nf].copy_(st[:, :nf - 1])
        b["fm"].zero_()
        b["fm"][:, :nf] = True
        t.pos.fill_(Lc)
        o = t.replay(N)
        calls += 1
        if ev is not None:
            ev.append((e0, e1, _event(ev)))
        both = torch.cat((o[0], b["idx"]), dim=0).tolist()
        pred, toks = both[0], both[1]
        nxt = toks[1:] + ([] if blk is not None else [hd])
        mis = next((j for j in range(len(nxt)) if pred[j] != nxt[j]), None)
        v = N - 1 if mis is None else mis
        if mis is not None and mis < n_p:
            rollbacks += 1
        if blk is not None:
            acc += max(0, v - n_p)
        ex = [True] + [(exact[j - 1] if j <= n_p else False) for j in range(1, v + 1)]
        npv = [(npass[j] + 1 if j < n_p else 1) for j in range(v + 1)]
        c = 0
        while c <= v and (npv[c] >= lag or ex[c]):
            c += 1
        seq = seq[:off] + toks[:v + 1] + [pred[v]]
        hh = o[1]
        if c > 0:
            Lc += c
            hc = hh[:, c - 1:c].clone()
        st, npass, exact = hh[:, c:v + 1].clone(), npv[c:], ex[c:]
        out, done = _finish(seq[:Lc - P + 1], terminal, max_new)
        if done:
            return out, blocks, acc, calls, rollbacks
        need = Lc + len(npass)
        if need <= have:
            dr.set(need, seq[need - 1 - P])
            catch, have = [], need
        else:
            dr.set(have, seq[have - 1 - P])
            catch = seq[have - P:need - P]


def graph_sizes(mode, K, passes):
    """Call shapes T (positions per row) that a decoding mode replays on the target."""
    if mode in ("plain", "token"):
        return [1]
    if mode == "rounds":
        return [K]
    if mode == "pipelined":
        return list(range(1, passes * K + 1))
    raise ValueError(f"mode must be one of {MODES}")


def ensure_graphs(r, sizes):
    """Capture the graphs of runner r for the call shapes in `sizes` that it does not have yet. A capture slows
    later graph replays on some drivers until another process starts CUDA (runtime.kick)."""
    for T in sizes:
        if (T, False) not in r.g:
            r.capture(T)


@torch.inference_mode()
def decode(target, dr, mode, K, P, tok0, h0, max_new, terminal, passes=2):
    """Decode after both models were prefilled to cache length P (the target's first token tok0 and state h0).
    dr: a Drafter (unused for "plain"). Returns a dict: tokens, target_calls, drafter_calls, rounds (drafted
    blocks), accepted (accepted drafts), rollbacks (pipelined), ms (decode wall time), draft_ms, verify_ms (GPU
    time drafting and verifying, summed over rounds)."""
    if mode != "plain" and not isinstance(dr, Drafter):
        raise TypeError("dr must be a manthan.spec.Drafter (Drafter(drafter_runner)), not " + type(dr).__name__)
    ev = []
    d0 = dr.calls if dr is not None else 0
    rounds = acc = rollbacks = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if mode == "plain":
        out, calls = plain_decode(target, tok0, h0, max_new, terminal)
    elif mode in ("token", "rounds"):
        out, rounds, acc, calls = spec_rounds(target, dr, K, P, tok0, h0, max_new, terminal,
                                              passes=1 if mode == "token" or not target.fbt else passes,
                                              exact=mode == "token", ev=ev)
    elif mode == "pipelined":
        out, rounds, acc, calls, rollbacks = spec_pipe(target, dr, passes, K, P, tok0, h0, max_new, terminal, ev=ev)
    else:
        raise ValueError(f"mode must be one of {MODES}")
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0)
    return {"tokens": out, "target_calls": calls, "drafter_calls": (dr.calls - d0) if dr is not None else 0,
            "rounds": rounds, "accepted": acc, "rollbacks": rollbacks, "ms": ms,
            "draft_ms": sum(a.elapsed_time(b) for a, b, _ in ev), "verify_ms": sum(b.elapsed_time(c) for _, b, c in ev)}


@torch.inference_mode()
def generate(target, drafter, prompt_ids, mode, K, max_new, terminal, bos, passes=2, prefill_passes=2):
    """Greedy generation at batch 1 for one prompt.

    target: Runner of the FBT or the twin (B=1); drafter: Runner of the drafter model (twin-style), or None for
    mode "plain"; prompt_ids: list of token ids; mode: "plain", "token", "rounds" (`passes` per round; the twin
    always uses one) or "pipelined" (`passes` passes per position); K: block size; terminal: ids that end the
    generation; bos: BOS id (no feedback at BOS positions in the FBT prefill); prefill_passes: FBT prompt passes.
    Captures the needed graphs, prefills both models (eager), then decodes. Returns decode()'s dict."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if mode != "plain" and drafter is None:
        raise ValueError(f"mode {mode!r} needs a drafter Runner")
    if mode in ("token", "pipelined") and not target.fbt:
        raise ValueError(f"mode {mode!r} is for the FBT; the twin verifies with mode 'rounds' (one pass)")
    if target.B != 1 or (drafter is not None and drafter.B != 1):
        raise ValueError("generate runs at batch 1; use manthan.batching.BatchEngine for batches")
    P = len(prompt_ids)
    S = target.S if mode == "plain" else min(target.S, drafter.S)
    if P + max_new + passes * K + 8 > S:
        raise ValueError(f"prompt ({P}) + max_new ({max_new}) + passes * K ({passes * K}) + 8 exceeds the runners' "
                         f"caches ({S} positions)")
    ensure_graphs(target, graph_sizes(mode, K, passes))
    ids = torch.tensor([prompt_ids], device=target.dev)
    tok0, h0, _ = prefill(target, ids, bos, prefill_passes)
    dr = None
    if mode != "plain":
        ensure_graphs(drafter, [1])
        prefill(drafter, ids, bos)
        dr = Drafter(drafter)
    return decode(target, dr, mode, K, P, tok0, h0, max_new, terminal, passes)
