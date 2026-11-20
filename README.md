# <img src="assets/logo.svg" alt="Manthan" width="320">

Fast inference for looped architectures: transformers that feed a state back into the network, across depth or across
tokens.

**News.** [2026/11] Manthan v0.1: feedback caching for FBT-style transformers, with our FBT fine-tune, its standard
twin and a drafter on Hugging Face.

## Methods

### Feedback caching (v0.1)

Feedback caching revisits the algorithm-system co-design of feedback transformers in the style of the full-bandwidth
transformer (FBT) [1], which feeds each token's top-layer state into the next token's input. A KV cache keeps each
token's keys and values; a feedback cache keeps the state each token feeds back, and how many refinements that state has had.

**Speculative decoding 1.62× as fast as plain decoding on GSM8K, up from 1.25×.** A drafted token's second refinement runs inside the
next round's verification call. A standard transformer reaches 1.74×.

[Post](https://people.eecs.berkeley.edu/~krishna/kernel/fbt.html) · [FBT paper](https://arxiv.org/abs/2608.08888)

## Install

```bash
git clone --recursive https://github.com/skylight-org/manthan && cd manthan && uv sync
```

[uv](https://docs.astral.sh/uv/) installs Python 3.11, PyTorch 2.6 and FlashAttention 2.7 from `uv.lock`; the FBT
model code is the authors' repository, a git submodule in `third_party/fbt`. Manthan runs on x86-64 Linux with an
NVIDIA GPU of the Ampere generation or newer and a CUDA 12 driver.

## Quickstart

```python
from manthan import LLM, SamplingParams

llm = LLM("kumarkrishna/d20-fbt-gate-openmath-k3",
          speculative_config={"model": "kumarkrishna/drafter-d6-openmath", "num_speculative_tokens": 7})
out = llm.chat([{"role": "user", "content": "A pen costs $3 and a notebook costs twice as much. What do both cost?"}],
               SamplingParams(max_tokens=192))
print(out[0].outputs[0].text)
```

The full API is documented in [`manthan/llm.py`](manthan/llm.py).

## Acknowledgments

Manthan builds on the [full-bandwidth transformer](https://arxiv.org/abs/2608.08888) and
[its code](https://github.com/xidulu/Full-bandwidth-transformer), [nanochat](https://github.com/karpathy/nanochat),
[FlashAttention](https://github.com/Dao-AILab/flash-attention) and [vLLM](https://github.com/vllm-project/vllm).
The FBT model code is their repository, included unchanged as a git submodule.

## Citation

[1] Xi Wang, Ziyang Cai, Zheng Zhan, Harry Dong, Ying Fan, Gustavo de Rosa, Tim Pearce, John Langford.
Full-bandwidth transformer. arXiv:2608.08888, 2026. https://arxiv.org/abs/2608.08888

```bibtex
@misc{agrawal2026feedback,
  author       = {Kumar Krishna Agrawal},
  title        = {Running Transformer Feedback Loops in Parallel},
  howpublished = {Blog post},
  year         = {2026},
  url          = {https://people.eecs.berkeley.edu/~krishna/kernel/fbt.html}
}
```

MIT license ([LICENSE](LICENSE)).
