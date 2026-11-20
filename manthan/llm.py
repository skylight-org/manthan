"""The high-level entry point, in the shape of vLLM's offline API: `LLM`, `SamplingParams`, `RequestOutput`.

    from manthan import LLM, SamplingParams

    llm = LLM("kumarkrishna/d20-fbt-gate-openmath-k3",
              speculative_config={"model": "kumarkrishna/drafter-d6-openmath", "num_speculative_tokens": 7})
    out = llm.chat([{"role": "user", "content": "What is 17 * 23?"}], SamplingParams(max_tokens=256))
    print(out[0].outputs[0].text)

A model is a nanochat checkpoint folder (`model_<step>.pt`, `meta_<step>.json`, and `tokenizer/`), given as a local
path or a Hugging Face repo id. Whether it is a feedback model comes from its metadata (trained with more than one
pass). Decoding is greedy. All CUDA graphs are captured when the LLM is built.

Feedback models decode with feedback caching (`FeedbackConfig`): each token's fed-back state is cached with the
number of passes it has had, and the passes a drafted token still needs run inside the next verification call
(`schedule="pipelined"`). `schedule="rounds"` runs them as separate calls, and `"exact"` verifies token by token.
"""
import glob
import json
import os
import re
from dataclasses import dataclass, field

import torch

from .batching import BatchEngine
from .data import terminal_ids
from .runtime import Runner, kick, load
from .spec import ensure_graphs, generate as _generate, graph_sizes

__all__ = ["LLM", "SamplingParams", "SpeculativeConfig", "FeedbackConfig", "RequestOutput", "CompletionOutput"]

_SCHEDULES = {"pipelined": "pipelined", "rounds": "rounds", "exact": "token"}


@dataclass
class SamplingParams:
    """max_tokens: most generated tokens, not counting the stop token. temperature: 0 only (v0.1 decodes greedily).
    stop_token_ids: extra token ids that end a generation (BOS and <|assistant_end|> always do); the stop token is not
    returned."""
    max_tokens: int = 256
    temperature: float = 0.0
    stop_token_ids: list = field(default_factory=list)

    def __post_init__(self):
        if self.temperature != 0.0:
            raise NotImplementedError("Manthan v0.1 decodes greedily: use temperature=0.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


@dataclass
class SpeculativeConfig:
    """model: the drafter (a local checkpoint folder or a Hugging Face repo id), with the target's tokenizer.
    num_speculative_tokens: tokens drafted per round; the target checks them with the round's own token, a block of
    num_speculative_tokens + 1 positions."""
    model: str
    num_speculative_tokens: int = 7
    revision: str = None


@dataclass
class FeedbackConfig:
    """How a feedback model runs its passes (ignored for standard models).

    passes:         Jacobi passes each drafted token gets before it is final. Our FBT fine-tune is near its fixed
                    point after 2.
    prefill_passes: passes over the prompt.
    schedule:       "pipelined" (feedback caching: a token's later passes run inside the next verification call),
                    "rounds" (each pass is its own call over the block) or "exact" (one call per drafted token, fed
                    the exact state: the same tokens as plain decoding, but slow).
    """
    passes: int = 2
    prefill_passes: int = 2
    schedule: str = "pipelined"

    def __post_init__(self):
        if self.schedule not in _SCHEDULES:
            raise ValueError(f"schedule must be one of {tuple(_SCHEDULES)}, not {self.schedule!r}")
        if self.passes < 1 or self.prefill_passes < 1:
            raise ValueError("passes and prefill_passes must be at least 1")
        if self.schedule == "pipelined" and self.passes < 2:
            raise ValueError("schedule 'pipelined' needs passes >= 2; use 'rounds' for one pass")


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list
    finish_reason: str          # "stop" (a stop token within max_tokens) or "length" (max_tokens tokens)


@dataclass
class RequestOutput:
    request_id: str
    prompt: str                 # None for token-id prompts and chats
    prompt_token_ids: list
    outputs: list               # one CompletionOutput
    metrics: dict = None        # batch 1: target_calls, drafter_calls, rounds, accepted, decode_ms


def _as(cls, x):
    if x is None or isinstance(x, cls):
        return x
    if isinstance(x, dict):
        return cls(**x)
    raise TypeError(f"expected a {cls.__name__} or a dict, not {type(x).__name__}")


def resolve(model, revision=None):
    """A local checkpoint folder for `model`: the path itself, or a download of the Hugging Face repo `model`."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model, revision=revision)


def checkpoint_meta(folder, step=None):
    """(step, metadata dict) of a nanochat checkpoint folder: the given step, else the last one."""
    if step is None:
        steps = [int(m.group(1)) for f in glob.glob(os.path.join(folder, "model_*.pt"))
                 if (m := re.search(r"model_(\d+)\.pt$", f))]
        if not steps:
            raise FileNotFoundError(f"no model_<step>.pt in {folder}")
        step = max(steps)
    with open(os.path.join(folder, f"meta_{step:06d}.json")) as f:
        return step, json.load(f)


def is_feedback_model(meta):
    """True for a model that feeds its states back (latent feedback, trained with more than one pass)."""
    return bool(meta["model_config"].get("latent_feedback")) and int(meta.get("num_forward_passes", 1)) > 1


class LLM:
    """A model, an optional drafter, and their CUDA graphs, ready to generate.

    model:              a nanochat checkpoint folder or a Hugging Face repo id (`revision`, `step` pick a version)
    speculative_config: a SpeculativeConfig or dict ({"model": ..., "num_speculative_tokens": 7}); None decodes
                        one token per call
    feedback_config:    a FeedbackConfig or dict, for feedback models
    max_model_len:      cache positions per sequence (prompt, generated tokens and one block of drafts)
    max_num_seqs:       sequences decoded together; above 1, requests are served with continuous batching
    """

    def __init__(self, model, *, speculative_config=None, feedback_config=None, max_model_len=2048, max_num_seqs=1,
                 revision=None, step=None, device="cuda"):
        self.dev = torch.device(device)
        self.spec = _as(SpeculativeConfig, speculative_config)
        self.fbc = _as(FeedbackConfig, feedback_config) or FeedbackConfig()
        self.max_model_len, self.max_num_seqs = max_model_len, max_num_seqs
        folder = resolve(model, revision)
        self.step, self.meta = checkpoint_meta(folder, step)
        self.feedback = is_feedback_model(self.meta)
        self.model, self.tokenizer = load(folder, self.dev, self.step)
        self.drafter = None
        if self.spec is not None:
            dfolder = resolve(self.spec.model, self.spec.revision)
            self.drafter, dtok = load(dfolder, self.dev, tokenizer_dir=self._tokenizer_dir(folder, dfolder))
            if dtok.get_vocab_size() != self.tokenizer.get_vocab_size():
                raise ValueError("the drafter's vocabulary differs from the target's")
        self.K = self.spec.num_speculative_tokens + 1 if self.spec is not None else 1
        self.mode = self._mode()
        self.bos = self.tokenizer.get_bos_token_id()
        self.terminal = terminal_ids(self.tokenizer)
        self.stats = None
        self._build()

    @staticmethod
    def _tokenizer_dir(folder, dfolder):
        """The drafter uses its own tokenizer/ if it has one, else the target's."""
        if os.path.isfile(os.path.join(dfolder, "tokenizer", "tokenizer.pkl")):
            return None
        t = os.path.join(folder, "tokenizer")
        return t if os.path.isfile(os.path.join(t, "tokenizer.pkl")) else None

    def _mode(self):
        if self.spec is None:
            return "plain"
        if not self.feedback:
            return "rounds"         # a standard model verifies a block in one exact pass
        return _SCHEDULES[self.fbc.schedule]

    def _passes(self):
        return self.fbc.passes if self.feedback else 1

    def _prefill_passes(self):
        return self.fbc.prefill_passes if self.feedback else 1

    def _build(self):
        """Allocate the caches and capture every graph the mode needs, then leave the slow timing regime that graph
        captures put the process in on some machines (manthan.runtime.kick; MANTHAN_KICK=0 skips it)."""
        S = self.max_model_len
        if self.max_num_seqs == 1:
            self.target = Runner(self.model, 1, S, self.dev, self.feedback)
            ensure_graphs(self.target, graph_sizes(self.mode, self.K, self._passes()))
            self.draft = None
            if self.drafter is not None:
                self.draft = Runner(self.drafter, 1, S, self.dev, False)
                ensure_graphs(self.draft, [1])
        else:
            if self.mode == "token":
                raise ValueError("schedule 'exact' runs at batch 1 (max_num_seqs=1)")
            self.engine = BatchEngine(self.model, self.feedback, self.max_num_seqs, drafter=self.drafter, k=self.K,
                                      max_len=S, bos=self.bos, terminal=self.terminal, dev=self.dev,
                                      prefill_passes=self._prefill_passes())
            self.engine.capture(max_lag={"plain": 0, "rounds": 1, "pipelined": self._passes()}[self.mode])
        torch.cuda.synchronize()
        if os.environ.get("MANTHAN_KICK", "1") != "0":
            kick()

    # ------------------------------------------------------------------------------------------------ requests
    def get_tokenizer(self):
        return self.tokenizer

    def _prompt_ids(self, p):
        if isinstance(p, str):
            return p, self.tokenizer.encode(p, prepend=self.bos)
        if isinstance(p, dict) and "prompt_token_ids" in p:
            return None, list(p["prompt_token_ids"])
        if isinstance(p, dict) and "prompt" in p:
            return p["prompt"], self.tokenizer.encode(p["prompt"], prepend=self.bos)
        if isinstance(p, (list, tuple)) and all(isinstance(t, int) for t in p):
            return None, list(p)
        raise TypeError("a prompt is a string, a list of token ids, or a dict with 'prompt' or 'prompt_token_ids'")

    def generate(self, prompts, sampling_params=None):
        """Complete each prompt (a string, a list of token ids, or a dict with "prompt" or "prompt_token_ids").
        A string is tokenized after a BOS token. Returns one RequestOutput per prompt, in order."""
        if isinstance(prompts, (str, dict)) or (isinstance(prompts, (list, tuple)) and prompts
                                                 and all(isinstance(t, int) for t in prompts)):
            prompts = [prompts]
        return self._run([self._prompt_ids(p) for p in prompts], sampling_params or SamplingParams())

    def chat(self, messages, sampling_params=None):
        """Answer a conversation (a list of {"role", "content"} messages, roles "system", "user" and "assistant",
        ending with a user message), or a list of conversations. Returns one RequestOutput per conversation."""
        convs = [messages] if messages and isinstance(messages[0], dict) else list(messages)
        reqs = []
        for conv in convs:
            ids = self.tokenizer.render_for_completion(
                {"messages": [dict(m) for m in conv] + [{"role": "assistant", "content": ""}]})
            reqs.append((None, ids))
        return self._run(reqs, sampling_params or SamplingParams())

    def _run(self, reqs, sp):
        if not reqs:
            return []
        terminal = self.terminal | set(sp.stop_token_ids)
        if self.max_num_seqs == 1:
            outs = []
            for i, (text, ids) in enumerate(reqs):
                o = _generate(self.target, self.draft, ids, self.mode, self.K, sp.max_tokens, terminal, self.bos,
                              passes=self._passes(), prefill_passes=self._prefill_passes())
                m = {k: o[k] for k in ("target_calls", "drafter_calls", "rounds", "accepted")}
                m["decode_ms"] = o["ms"]
                outs.append(self._output(i, text, ids, o["tokens"], sp, m))
            return outs
        need = [len(ids) + 2 * sp.max_tokens + 4 * self.K for _, ids in reqs]
        if max(need) > self.max_model_len:
            raise ValueError(f"with continuous batching a request needs prompt + 2 * max_tokens + 4 * block positions "
                             f"({max(need)}); raise max_model_len ({self.max_model_len}) or lower max_tokens")
        self.engine.terminal = terminal
        res = self.engine.run([ids for _, ids in reqs], mode=self.mode, npass=self._passes(), lag=self._passes(),
                              max_new=sp.max_tokens)
        self.stats = res.stats
        return [self._output(i, text, ids, toks, sp, None) for i, ((text, ids), toks) in enumerate(zip(reqs, res.outputs))]

    def _output(self, i, text, ids, toks, sp, metrics):
        reason = "length" if len(toks) >= sp.max_tokens else "stop"
        comp = CompletionOutput(0, self.tokenizer.decode(toks), list(toks), reason)
        return RequestOutput(str(i), text, list(ids), [comp], metrics)

    def __repr__(self):
        kind = "feedback" if self.feedback else "standard"
        spec = f", drafter K={self.K}" if self.spec is not None else ""
        return f"LLM({kind} model, step {self.step}, mode {self.mode!r}{spec}, max_num_seqs={self.max_num_seqs})"

