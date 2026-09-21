# LLM from Scratch
Build a large language model from scratch, including pretraining, instruction fine-tuning, and RLHF reward modeling.

## Files
- `tokenizer.py` — Simple tokenizer implementation
- `models/gpt.py` — GPT model, TransformerBase (shared base for LM and reward models)
- `models/rewards.py` — BradleyTerry reward model
- `dataset/gutenberg_dataset.py` — Sliding window dataset for pretraining (no padding; every sample is exactly context_length tokens)
- `dataset/instructionft_dataset.py` — Instruction fine-tuning dataset with variable-length sequences, padded to batch max length using token 50256 (`<|endoftext|>`)
- `inference.py` — Training loops, inference, save/load utilities

## Architecture

### Model structure
- `TransformerBase`: token embedding + positional embedding + dropout + transformer blocks. Shared base for all model heads.
- `GPT` / `LanguageModel`: TransformerBase + LayerNorm + Linear(dim, vocab_size). Generative, predicts next token.
- `BradleyTerry`: TransformerBase + Linear(dim, 1). Discriminative, outputs a scalar reward per sequence.

### Reward model
- Input: `x + y + <eos>` (prompt + completion + end token), padded with 50256 to the longest sequence in the batch.
- The reward is read from the hidden state at the last real token position (the `<eos>`), identified using an attention mask.
- No LayerNorm before the reward head (unlike the LM head) — standard practice.
- Reward models are discriminative: they score a sequence rather than generating tokens. Used in RLHF to elicit qualities like helpfulness and harmlessness.

### Bradley-Terry loss
- Trained on preference pairs (chosen, rejected).
- Loss: `-log(sigmoid(reward_chosen - reward_rejected))`
- No target tensors or `-100` masking needed — only the reward scalars are compared.

### Padding conventions
- Input tokens are padded with `50256` (`<|endoftext|>`).
- Target tokens in SFT use `-100` as the ignore index for `CrossEntropyLoss` (pads beyond the first `<eos>` are masked out).
- `-100` is never used in input tokens.

## Misconceptions
Things I got wrong while building this, with the corrections.
- [Q/K/V projection vs. the split into heads](notes/attention.md) — which comes first, and why.

## Dataset
- For preference reward model training, using https://huggingface.co/datasets/Anthropic/hh-rlhf


## TODO
- [ ] Time per-token latency across a 100-token generation and watch it climb — every step re-attends over the full prefix from scratch. That curve is the motivation for a KV cache.
- [ ] Add a KV cache.
- [ ] Answer the open KL questions in [`notes/distillation.md`](notes/distillation.md) — forward vs backward KL, and why distillation and RLHF pick opposite directions.
- [ ] Implement [nanochat](https://github.com/karpathy/nanochat) from scratch. Should teach a BPE tokenizer, a modern transformer (RoPE, Muon), pretrain → midtrain → SFT → GRPO with tool use, an eval harness, and a KV-cached inference engine — but not LoRA, not reward models, and nothing past a basic KV cache on the serving side.

## Requirements
- Python 3.x
- Conda (Miniforge recommended for Apple Silicon)

## Installation

```bash
~/miniforge3/bin/conda run -n ml-env pip install -r requirements.txt
```
