"""Manthan: fast inference for looped architectures. v0.1 serves FBT-style feedback transformers.

High-level API (`manthan.llm`), in the shape of vLLM's offline API:

    from manthan import LLM, SamplingParams
    llm = LLM("kumarkrishna/d20-fbt-gate-openmath-k3",
              speculative_config={"model": "kumarkrishna/drafter-d6-openmath", "num_speculative_tokens": 7})
    print(llm.chat([{"role": "user", "content": "What is 17 * 23?"}])[0].outputs[0].text)
"""
__version__ = "0.1.0"

from .runtime import DT, Runner, Snap, emb_of, kick, load, plain_decode, prefill
from .spec import (MODES, Drafter, decode, ensure_graphs, finish, generate, graph_sizes, spec_pipe, spec_rounds,
                   verify_tok)
from .data import gsm_rows, path, prompt_ids, terminal_ids
from .states import (FeedbackCache, Quantizer, capture_rebuild, dequantize, exact_states, feedback_mask, fit_stats,
                     load_stats, prefill_store, quantize, rebuild, save_stats)
from .batching import BatchEngine, BatchResult
from .diagnostics import ParallelMap, climbmix_rows, decode_states, diagnose, format_report
from .llm import LLM, CompletionOutput, FeedbackConfig, RequestOutput, SamplingParams, SpeculativeConfig

__all__ = [
    "__version__",
    # high-level API
    "LLM", "SamplingParams", "SpeculativeConfig", "FeedbackConfig", "RequestOutput", "CompletionOutput",
    # runtime
    "DT", "Runner", "Snap", "emb_of", "kick", "load", "plain_decode", "prefill",
    # speculative decoding
    "MODES", "Drafter", "decode", "ensure_graphs", "finish", "generate", "graph_sizes", "spec_pipe", "spec_rounds",
    "verify_tok",
    # data (GSM8K prompts, release layout)
    "gsm_rows", "path", "prompt_ids", "terminal_ids",
    # feedback cache and rebuild
    "FeedbackCache", "Quantizer", "capture_rebuild", "dequantize", "exact_states", "feedback_mask", "fit_stats",
    "load_stats", "prefill_store", "quantize", "rebuild", "save_stats",
    # continuous batching
    "BatchEngine", "BatchResult",
    # diagnostics
    "ParallelMap", "climbmix_rows", "decode_states", "diagnose", "format_report",
]
