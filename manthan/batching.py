"""Continuous batching for the FBT and its twin, with every model call one CUDA-graph replay.

`BatchEngine` decodes a queue of requests in B rows. When a row finishes, it takes the next request: the prompt is
prefilled eagerly at batch 1 (`runtime.prefill`, `prefill_passes` for the FBT) and copied into the row's slice of the
batch-B caches. During decoding every call of a model is one graph replay at batch B. Rows with no request left (the
batch drains at the end) run along, and their results are discarded. Three modes:

  plain               one call of one position per step (FBT: each token is fed the state of the token before it)
  rounds(npass)       draft K-1 tokens, then verify the block of K positions with npass calls over it
                      (the twin: npass = 1; the FBT: npass >= 2, each pass feeds the previous pass's states)
  pipelined(lag)      FBT only: a token's later passes run inside the next round's call, next to the new block;
                      a token is committed after `lag` passes (or when its input state is already exact)

Rows of one call can need different numbers of positions (pending tokens in pipelined mode, catch-up tokens of the
drafter). Such a call runs at the largest count and shorter rows are padded at the end: a padded position writes a
cache entry past the row's length, which no valid position reads and a later call overwrites. The drafter's catch-up
tokens and the block's first token go into one call, so at batch 1 this engine is not call-for-call the same as
`spec.spec_rounds` / `spec.spec_pipe`, which feed catch-up tokens one call each.

Timing: decoding time excludes refills (prefill and row copy). Throughput = generated tokens / decoding time.
Drafter and target GPU times come from CUDA events around the drafting and verifying calls of each round; the rest
of a round's wall time is host work (bookkeeping, the device-to-host copy of the round's tokens).

Exactness (what is and is not guaranteed). At a fixed batch size B, a request's output does not depend on its row,
its batch-mates or when it was taken: each row's arithmetic in a graph is independent of the other rows' contents.
Against batch-1 decoding, the computed function is the same greedy recurrence, but the batched kernels round
differently, so outputs can part at near-ties of the top two logits (the post's runs: the same arm at batch 8 and 32
gives equal outputs for 966 to 1,000 of 1,319 GSM8K problems).
"""
import time
from dataclasses import dataclass, field

import torch
from nanochat.gpt import norm

from .runtime import DT, Runner, prefill
from .spec import _finish as finish

MODES = ("plain", "rounds", "pipelined")


@dataclass
class BatchResult:
    """Outputs and counters of one `BatchEngine.run`.

    outputs: one token list per request, in request order (None for a request skipped because prompt plus
             2 * max_new + 4K positions do not fit the cache). Outputs exclude the terminal token.
    stats:   counters of the run (see `BatchEngine.run`).
    """
    outputs: list
    stats: dict = field(default_factory=dict)


def _copy_row(dst, src, b, P):
    """Copy the first P cache positions of the batch-1 runner `src` into row b of the batch-B runner `dst`."""
    dst.kc[:, b, :P].copy_(src.kc[:, 0, :P])
    dst.vc[:, b, :P].copy_(src.vc[:, 0, :P])


class BatchEngine:
    """One target model (FBT or twin) and an optional drafter, served at batch B with continuous batching.

    model:     the target (a nanochat GPT; `fbt` says whether it takes latent feedback)
    drafter:   the draft model (needed by modes rounds and pipelined), or None for plain decoding only
    batch:     rows B decoded together
    k:         block size K of speculative decoding (the head token plus K-1 drafts)
    max_len:   cache positions per row
    bos:       the BOS token id (marks positions without feedback in the FBT prefill)
    terminal:  token ids that end a request
    prefill_passes: passes over each prompt (the FBT; the twin and the drafter prefill in one)
    Call `capture()` once before `run()`: on some machines a graph capture puts the process's graphs in a slow timing
    regime (manthan.runtime.kick), so capture everything first, then time.
    """

    def __init__(self, model, fbt, batch, drafter=None, k=8, max_len=2048, bos=None, terminal=(), dev=None,
                 prefill_passes=2):
        dev = dev if dev is not None else torch.device("cuda")
        self.m, self.fbt, self.B, self.K, self.S, self.dev = model, fbt, batch, k, max_len, dev
        self.bos, self.terminal, self.prefill_passes = bos, set(terminal), prefill_passes
        self.t = Runner(model, batch, max_len, dev, fbt)
        self.t1 = Runner(model, 1, max_len, dev, fbt)
        self.dm = drafter
        if drafter is not None:
            self.dr = Runner(drafter, batch, max_len, dev, False)
            self.dr1 = Runner(drafter, 1, max_len, dev, False)
        self.ar = torch.arange(batch, device=dev)
        self.max_lag = None         # set by capture()

    @torch.inference_mode()
    def capture(self, max_lag=2):
        """Capture the graphs the modes use; shapes already captured are skipped.

        max_lag = 0: plain decoding only (target T = 1). max_lag >= 1 adds rounds: target T = K, drafter
        T = 1..max_lag*K+1 (catch-up tokens plus the head token). max_lag >= 2 (FBT) adds pipelined calls with
        lag <= max_lag: target T = 1..max_lag*K (pending tokens plus a block)."""
        K = self.K
        if max_lag >= 1 and self.dm is None:
            raise ValueError("speculative modes need a drafter; capture(max_lag=0) for plain decoding")
        ts = {1}
        if max_lag >= 1:
            ts.add(K)
        if max_lag >= 2 and self.fbt:
            ts.update(range(1, max_lag * K + 1))
        for T in sorted(ts):
            if (T, False) not in self.t.g:
                self.t.capture(T)
        if max_lag >= 1:
            for T in range(1, max_lag * K + 2):
                if (T, False) not in self.dr.g:
                    self.dr.capture(T)
        self.max_lag = max(self.max_lag if self.max_lag is not None else 0, max_lag if self.fbt else min(max_lag, 1))

    @torch.inference_mode()
    def run(self, requests, mode="plain", npass=1, lag=2, max_new=192):
        """Decode every request (a list of prompt token ids) greedily and return a `BatchResult`.

        mode:    "plain", "rounds" (npass target calls per round) or "pipelined" (FBT; commit after `lag` passes)
        max_new: most generated tokens per request
        stats:   mode, batch, requests, skipped, tokens, dec_ms, tok_s, refill_ms, calls (target calls);
                 plain adds steps and ms_per_step; rounds/pipelined add rounds, blocks, acc (accepted drafts),
                 acceptance (acc / (blocks * (K-1))), draft_ms and verify_ms (GPU time of drafting and target calls),
                 draft_share, draft_ms_per_round, target_ms_per_call, host_ms_per_round (round wall time minus GPU
                 time of drafting and target calls), tokens_per_row_round; pipelined adds rollbacks.
        """
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if mode != "plain" and self.dm is None:
            raise ValueError(f"mode {mode!r} needs a drafter")
        if mode == "rounds" and npass > 1 and not self.fbt:
            raise ValueError("the twin verifies in one pass (npass = 1); more passes need the FBT")
        if mode == "pipelined" and not self.fbt:
            raise ValueError("pipelined passes need the FBT")
        need = {"plain": 0, "rounds": 1, "pipelined": lag}[mode]
        if self.max_lag is None or self.max_lag < need:
            raise ValueError(f"call capture(max_lag={need}) (or more) before run(mode={mode!r})")
        q = _Queue(self, [list(r) for r in requests], max_new, use_dr=mode != "plain")
        if mode == "plain":
            info = _run_plain(q)
        elif mode == "rounds":
            info = _run_rounds(q, npass)
        else:
            info = _run_pipe(q, lag)
        outputs = [q.res.get(i) for i in range(len(requests))]
        ntok = sum(len(o) for o in outputs if o is not None)
        st = {"mode": mode, "batch": self.B, "K": self.K, "max_new": max_new, "requests": len(requests),
              "skipped": [i for i, o in enumerate(outputs) if o is None], "tokens": ntok,
              "tok_s": 1000 * ntok / max(info["dec_ms"], 1e-9), "refill_ms": q.refill_ms, **info}
        if mode == "rounds":
            st["npass"] = npass
        if mode == "pipelined":
            st["lag"] = lag
        if mode == "plain":
            st["ms_per_step"] = info["dec_ms"] / max(info["steps"], 1)
        else:
            R = max(info["rounds"], 1)
            st["acceptance"] = info["acc"] / max(info["blocks"] * (self.K - 1), 1)
            st["draft_share"] = info["draft_ms"] / max(info["draft_ms"] + info["verify_ms"], 1e-9)
            st["draft_ms_per_round"] = info["draft_ms"] / R
            st["target_ms_per_call"] = info["verify_ms"] / max(info["calls"], 1)
            st["host_ms_per_round"] = (info["dec_ms"] - info["draft_ms"] - info["verify_ms"]) / R
            st["tokens_per_row_round"] = ntok / (R * self.B)
        return BatchResult(outputs, st)


class _Queue:
    """Request queue and row bookkeeping of one run: refills (eager batch-1 prefill copied into a row) and results."""

    def __init__(self, E, rows, max_new, use_dr):
        self.E, self.rows, self.max_new, self.use_dr = E, rows, max_new, use_dr
        self.t, self.t1 = E.t, E.t1
        if use_dr:
            self.dr, self.dr1 = E.dr, E.dr1
        self.fbt, self.B, self.K, self.dev, self.ar = E.fbt, E.B, E.K, E.dev, E.ar
        self.terminal = E.terminal
        self.q = 0
        self.prob = [-1] * self.B
        self.P = [0] * self.B
        self.res = {}
        self.refill_ms = 0.0
        self.act = torch.zeros(self.B, dtype=torch.int32, device=self.dev)

    def take(self, b):
        """Prefill the next request into row b. Returns (tok0, h0 [D], xn0 [D]) or None when the queue is empty.
        Requests whose prompt plus 2 * max_new + 4K positions exceed the cache are skipped."""
        S = self.t.S
        while self.q < len(self.rows):
            i = self.q
            self.q += 1
            ids_l = self.rows[i]
            P = len(ids_l)
            if P + 2 * self.max_new + 4 * self.K > S:
                continue
            ta = time.perf_counter()
            ids = torch.tensor([ids_l], device=self.dev)
            tok0, h0, xn0 = prefill(self.t1, ids, self.E.bos, passes=self.E.prefill_passes)
            _copy_row(self.t, self.t1, b, P)
            self.t.prev[b].copy_(xn0[0])
            if self.use_dr:
                prefill(self.dr1, ids, self.E.bos)
                _copy_row(self.dr, self.dr1, b, P)
                self.dr.prev[b].copy_(self.dr1.prev[0])
                self.dr.pos[b] = P
            self.act[b] = 1
            torch.cuda.synchronize()
            self.refill_ms += 1000 * (time.perf_counter() - ta)
            self.prob[b], self.P[b] = i, P
            return tok0, h0[0, 0], xn0[0, 0]
        self.prob[b] = -1
        self.act[b] = 0
        return None

    def record(self, b, out):
        self.res[self.prob[b]] = out

    def emb(self, toks):
        """Normalised drafter embeddings of tokens [B] -> [B, 1, D] (the drafter's smear input)."""
        return norm(self.E.dm._embed_tokens(toks[:, None]))


# ----------------------------------------------------------------------------- plain decoding
def _run_plain(A):
    t, B, term = A.t, A.B, A.terminal
    b1 = t.bufs(1)
    if A.fbt:
        b1["fm"].fill_(True)
    tk = [None] * B
    out = [[] for _ in range(B)]

    def start(b):
        while True:
            r = A.take(b)
            if r is None:
                return
            tok0, h0, _ = r
            t.pos[b] = A.P[b]
            if A.fbt:
                b1["fb"][b, 0].copy_(h0)
            out[b], tk[b] = [], tok0
            if tok0 not in term:
                return
            A.record(b, [])
    for b in range(B):
        start(b)
    steps, dec_ms = 0, 0.0
    nxt = torch.zeros(B, 1, dtype=torch.long, device=A.dev)
    while True:
        ta = time.perf_counter()
        # rows whose pending token ends them, or that are full, finish before the call
        for b in range(B):
            while A.prob[b] >= 0:
                if tk[b] in term:
                    A.record(b, out[b])
                elif len(out[b]) + 1 >= A.max_new:
                    out[b].append(tk[b])
                    A.record(b, out[b])
                else:
                    out[b].append(tk[b])
                    break
                tr = A.refill_ms
                start(b)
                ta += (A.refill_ms - tr) / 1000
        if not any(p >= 0 for p in A.prob):
            dec_ms += 1000 * (time.perf_counter() - ta)
            break
        nxt[:, 0] = torch.tensor([x if x is not None else 0 for x in tk], device=A.dev)
        b1["idx"].copy_(nxt)
        o = t.replay(1)
        steps += 1
        t.pos.add_(A.act)
        t.prev.copy_(o[2])
        if A.fbt:
            b1["fb"].copy_(o[1])
        tk = o[0][:, 0].tolist()
        dec_ms += 1000 * (time.perf_counter() - ta)
    return {"steps": steps, "dec_ms": dec_ms, "calls": steps}


# ----------------------------------------------------------------------------- drafting at batch B
def _draft(A, catch, heads, mask):
    """Rows with mask[b]: feed catch[b] + [heads[b]] in one call (padded to the longest), then K-2 single-token steps.
    Returns blk [B, K] (heads, then K-1 drafts) on the GPU. Rows without mask run along; their position stays."""
    dr, B, K, dev = A.dr, A.B, A.K, A.dev
    c = [len(catch[b]) if mask[b] else 0 for b in range(B)]
    T0 = max(c) + 1
    rows = []
    for b in range(B):
        h = heads[b] if mask[b] else 0
        r = (list(catch[b]) if mask[b] else []) + [h]
        rows.append(r + [h] * (T0 - len(r)))
    D0 = dr.bufs(T0)
    D0["idx"].copy_(torch.tensor(rows, device=dev))
    m = torch.tensor(mask, dtype=torch.int32, device=dev)
    cg = torch.tensor(c, device=dev)
    o = dr.replay(T0)
    blk = torch.empty(B, K, dtype=torch.long, device=dev)
    blk[:, 0] = D0["idx"][A.ar, cg]
    blk[:, 1] = o[0][A.ar, cg]
    dr.prev.copy_(o[2][A.ar, cg].unsqueeze(1))
    dr.pos.add_((cg.to(torch.int32) + 1) * m)
    D1 = dr.bufs(1)
    for j in range(2, K):
        D1["idx"][:, 0].copy_(blk[:, j - 1])
        o = dr.replay(1)
        dr.pos.add_(m)
        dr.prev.copy_(o[2])
        blk[:, j] = o[0][:, 0]
    return blk


# ----------------------------------------------------------------------------- passes per round
def _run_rounds(A, npass):
    t, dr, B, K, term, dev, ar = A.t, A.dr, A.B, A.K, A.terminal, A.dev, A.ar
    TB = t.bufs(K)
    Lt = torch.zeros(B, dtype=torch.int32, device=dev)
    h = torch.zeros(B, t.D, dtype=DT, device=dev)
    pend, catch, out = [None] * B, [[] for _ in range(B)], [[] for _ in range(B)]
    st = {"blocks": 0, "acc": 0}

    def start(b):
        while True:
            r = A.take(b)
            if r is None:
                return
            tok0, h0, _ = r
            Lt[b] = A.P[b]
            h[b].copy_(h0)
            pend[b], catch[b], out[b] = tok0, [], []
            if tok0 not in term:
                return
            A.record(b, [])
    for b in range(B):
        start(b)
    rounds, dec_ms, draft_ms, verify_ms = 0, 0.0, 0.0, 0.0
    while any(p >= 0 for p in A.prob):
        ta = time.perf_counter()
        mask = [p >= 0 for p in A.prob]
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        blk = _draft(A, catch, pend, mask)
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        TB["idx"].copy_(blk)
        if A.fbt:
            TB["fb"].zero_()
            TB["fb"][:, 0].copy_(h)
            TB["fm"].zero_()
            TB["fm"][:, 0] = True
        t.pos.copy_(Lt)
        o = t.replay(K)
        for _ in range(npass - 1):
            TB["fb"][:, 1:].copy_(o[1][:, :-1])
            TB["fm"].fill_(True)
            t.pos.copy_(Lt)
            o = t.replay(K)
        e2 = torch.cuda.Event(enable_timing=True)
        e2.record()
        pred = o[0]
        mm = pred[:, :-1] != blk[:, 1:]
        acc_i = torch.where(mm.any(1), mm.int().argmax(1), torch.full_like(mm[:, 0], K - 1, dtype=torch.long))
        newp = pred[ar, acc_i]
        if A.fbt:
            h.copy_(o[1][ar, acc_i])
        else:
            t.prev.copy_(o[2][ar, acc_i].unsqueeze(1))
        Lt.add_((acc_i.to(torch.int32) + 1) * A.act)
        full = acc_i == K - 1
        dr.pos.copy_(torch.where(full, Lt - 1, Lt) * A.act)
        dr.prev.copy_(A.emb(torch.where(full, blk[:, K - 2], blk[ar, acc_i])))
        host = torch.cat((acc_i[:, None], newp[:, None], blk), 1).tolist()
        rounds += 1
        torch.cuda.synchronize()
        draft_ms += e0.elapsed_time(e1)
        verify_ms += e1.elapsed_time(e2)
        for b in range(B):
            if not mask[b]:
                continue
            ab, nb, d = host[b][0], host[b][1], host[b][2:]
            st["blocks"] += 1
            toks = d[:ab + 1]
            cut = next((j for j in range(1, ab + 1) if toks[j] in term), None)
            if cut is not None:
                st["acc"] += cut
                out[b].extend(toks[:cut])
                done = True
            else:
                st["acc"] += ab
                out[b].extend(toks)
                done = nb in term
            if len(out[b]) >= A.max_new:
                out[b], done = out[b][:A.max_new], True
            if done:
                A.record(b, out[b])
                tr = A.refill_ms
                start(b)
                ta += (A.refill_ms - tr) / 1000
            else:
                pend[b] = nb
                catch[b] = [d[K - 1]] if ab == K - 1 else []
        dec_ms += 1000 * (time.perf_counter() - ta)
    return {"rounds": rounds, "dec_ms": dec_ms, "calls": rounds * npass, "draft_ms": draft_ms, "verify_ms": verify_ms,
            **st}


# ----------------------------------------------------------------------------- pipelined passes
def _run_pipe(A, lag):
    t, dr, B, K, term, dev, ar = A.t, A.dr, A.B, A.K, A.terminal, A.dev, A.ar
    D = t.D
    hc = torch.zeros(B, D, dtype=DT, device=dev)
    stt = torch.zeros(B, lag * K, D, dtype=DT, device=dev)
    arS = torch.arange(lag * K, device=dev)
    S = [None] * B          # per-row host state
    st = {"blocks": 0, "acc": 0, "rollbacks": 0}

    def start(b):
        while True:
            r = A.take(b)
            if r is None:
                S[b] = None
                return
            tok0, h0, _ = r
            hc[b].copy_(h0)
            P = A.P[b]
            S[b] = {"seq": [tok0], "Lc": P, "npass": [], "exact": [], "catch": [], "have": P}
            if tok0 not in term:
                return
            A.record(b, [])
    for b in range(B):
        start(b)
    rounds, dec_ms, draft_ms, verify_ms = 0, 0.0, 0.0, 0.0
    while any(p >= 0 for p in A.prob):
        ta = time.perf_counter()
        # rows that have nothing left to verify finish before the round
        for b in range(B):
            while A.prob[b] >= 0:
                s = S[b]
                dok = not any(x in term for x in s["seq"][1:]) and len(s["seq"]) <= A.max_new
                if dok or s["npass"]:
                    s["dok"] = dok
                    break
                A.record(b, finish(s["seq"], term, A.max_new)[0])
                tr = A.refill_ms
                start(b)
                ta += (A.refill_ms - tr) / 1000
        live = [p >= 0 for p in A.prob]
        if not any(live):
            dec_ms += 1000 * (time.perf_counter() - ta)
            break
        dok = [live[b] and S[b]["dok"] for b in range(B)]
        n_p = [len(S[b]["npass"]) if live[b] else 0 for b in range(B)]
        Nb = [n_p[b] + (K if dok[b] else 0) for b in range(B)]
        nf = [n_p[b] + (1 if dok[b] else 0) for b in range(B)]
        N = max(Nb)
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        blk = _draft(A, [S[b]["catch"] if dok[b] else [] for b in range(B)],
                     [S[b]["seq"][-1] if dok[b] else 0 for b in range(B)], dok)
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        for b in range(B):
            if dok[b]:
                S[b]["catch"], S[b]["have"] = [], S[b]["Lc"] + n_p[b] + K - 1
                st["blocks"] += 1
        TBf = t.bufs(N)
        pend = [[0] * N for _ in range(B)]
        for b in range(B):
            if live[b] and n_p[b]:
                off = S[b]["Lc"] - A.P[b]
                pend[b][:n_p[b]] = S[b]["seq"][off:off + n_p[b]]
        idx = torch.tensor(pend, device=dev)
        npg = torch.tensor(n_p, device=dev)
        dk = torch.tensor(dok, device=dev)
        if N >= K:
            cols = (npg[:, None] + torch.arange(K, device=dev)[None, :]).clamp(max=N - 1)
            idx = torch.where(dk[:, None], idx.scatter(1, cols, blk), idx)
        TBf["idx"].copy_(idx)
        fm = torch.arange(N, device=dev)[None, :] < torch.tensor(nf, device=dev)[:, None]
        fbv = torch.cat((hc[:, None], stt[:, :N - 1]), 1)
        TBf["fb"].copy_(fbv * fm[..., None].to(DT))
        TBf["fm"].copy_(fm)
        t.pos.copy_(torch.tensor([S[b]["Lc"] if live[b] else 0 for b in range(B)], dtype=torch.int32, device=dev))
        o = t.replay(N)
        e2 = torch.cuda.Event(enable_timing=True)
        e2.record()
        both = torch.cat((o[0], TBf["idx"]), 1).tolist()
        rounds += 1
        cs, dpos, dtok = [0] * B, [0] * B, [0] * B
        for b in range(B):
            if not live[b]:
                continue
            s, nb = S[b], Nb[b]
            pred, toks = both[b][:nb], both[b][N:N + nb]
            hd = s["seq"][-1]
            off = s["Lc"] - A.P[b]
            nxt = toks[1:] + ([] if dok[b] else [hd])
            mis = next((j for j in range(len(nxt)) if pred[j] != nxt[j]), None)
            v = nb - 1 if mis is None else mis
            if mis is not None and mis < n_p[b]:
                st["rollbacks"] += 1
            if dok[b]:
                st["acc"] += max(0, v - n_p[b])
            ex = [True] + [(s["exact"][j - 1] if j <= n_p[b] else False) for j in range(1, v + 1)]
            npv = [(s["npass"][j] + 1 if j < n_p[b] else 1) for j in range(v + 1)]
            c = 0
            while c <= v and (npv[c] >= lag or ex[c]):
                c += 1
            s["seq"] = s["seq"][:off] + toks[:v + 1] + [pred[v]]
            s["Lc"] += c
            s["npass"], s["exact"] = npv[c:], ex[c:]
            cs[b] = c
            P = A.P[b]
            need = s["Lc"] + len(s["npass"])
            if need <= s["have"]:
                dpos[b], dtok[b] = need, s["seq"][need - 1 - P]
                s["catch"], s["have"] = [], need
            else:
                dpos[b], dtok[b] = s["have"], s["seq"][s["have"] - 1 - P]
                s["catch"] = s["seq"][s["have"] - P:need - P]
        cg = torch.tensor(cs, device=dev)
        hh = o[1]
        hc.copy_(torch.where((cg > 0)[:, None], hh[ar, (cg - 1).clamp(min=0)], hc))
        gi = (cg[:, None] + arS[None, :]).clamp(max=N - 1)
        stt.copy_(hh.gather(1, gi[..., None].expand(-1, -1, D)))
        dr.pos.copy_(torch.tensor(dpos, dtype=torch.int32, device=dev))
        dr.prev.copy_(A.emb(torch.tensor(dtok, device=dev)))
        torch.cuda.synchronize()
        draft_ms += e0.elapsed_time(e1)
        verify_ms += e1.elapsed_time(e2)
        for b in range(B):
            if not live[b]:
                continue
            s = S[b]
            fin, done = finish(s["seq"][:s["Lc"] - A.P[b] + 1], term, A.max_new)
            if done:
                A.record(b, fin)
                tr = A.refill_ms
                start(b)
                ta += (A.refill_ms - tr) / 1000
        dec_ms += 1000 * (time.perf_counter() - ta)
    return {"rounds": rounds, "dec_ms": dec_ms, "calls": rounds, "draft_ms": draft_ms, "verify_ms": verify_ms, **st}
