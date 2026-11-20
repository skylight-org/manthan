"""GSM8K test problems as zero-shot chat prompts, and checkpoint/data paths from the environment.

Paths are never hardcoded: `path(var)` reads the environment variable `var` (one of the LAYOUT keys: FBT_CKPT,
STD_CKPT, DRAFTER_CKPT, FBT_STATS, GSM8K_PARQUET, CLIMBMIX_VAL, FBT_BASE_CKPTS, FBT_VAL_ROWS), and otherwise the
layout under $FBT_ROOT given by LAYOUT.
"""
import os

__all__ = ["path", "gsm_rows", "prompt_ids", "terminal_ids"]

# relative locations under $FBT_ROOT, used when the variable itself is unset
LAYOUT = {
    "FBT_CKPT": "nanochat_base/chatsft_checkpoints/d20-fbt-gate-openmath-k3",
    "STD_CKPT": "nanochat_base/chatsft_checkpoints/d20-standard-openmath-k1",
    "DRAFTER_CKPT": "nanochat_base/chatsft_checkpoints/drafter-d6-openmath",
    "FBT_STATS": "work/runG/states_ft4407.pt",
    "GSM8K_PARQUET": "data/gsm8k/test.parquet",
    "CLIMBMIX_VAL": "nanochat_base/base_data_climbmix/shard_06542.parquet",
    "FBT_BASE_CKPTS": "ckpt/base_checkpoints",
    "FBT_VAL_ROWS": "work/val_rows.pt",
}


def path(var):
    """The path named by environment variable `var`, else $FBT_ROOT/<release layout>. Raises if neither is set."""
    if os.environ.get(var):
        return os.environ[var]
    root = os.environ.get("FBT_ROOT")
    if root and var in LAYOUT:
        return os.path.join(root, LAYOUT[var])
    raise KeyError(f"set {var} (or FBT_ROOT for the release layout)")


def gsm_rows(parquet=None):
    """The GSM8K test split as a list of {"question", "answer"} dicts (1,319 rows), from `parquet` or
    path("GSM8K_PARQUET")."""
    import pyarrow.parquet as pq
    return pq.read_table(parquet or path("GSM8K_PARQUET")).to_pylist()


def prompt_ids(tok, row):
    """Token ids of the zero-shot chat prompt for one GSM8K row (the user turn and an open assistant turn), as in
    every run of the post."""
    conv = {"messages": [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": ""}]}
    return tok.render_for_completion(conv)


def terminal_ids(tok):
    """Tokens that end a generation: BOS and <|assistant_end|> (neither is emitted)."""
    return {tok.get_bos_token_id(), tok.encode_special("<|assistant_end|>")}

