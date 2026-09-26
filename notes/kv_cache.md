# KV Cache

## Intuition: Why do we need this?

Imagine we inferenced an input: "I am a", this generation would have created (3, 3*768) [Q,K,V] vectors for all 12 decoder blocks. Imagine the model generated "cat". Now, for the follow on generation with input "I am a cat". Is there anything we could reuse/cache instead of recomputing?

For that to be worth anything, we first have to prove the vectors for "I", "am" and "a" don't change when "cat" is appended (at *every* block, not just the first one!).

## Worked example

Collapsing the model to `d = 1` so every number is checkable by hand. One head, no scaling (`sqrt(1) = 1`), weights:

`W_q = 1.0`, `W_k = 0.5`, `W_v = 2.0`

### Step 1: input is "I am a"

Token + position embedding gives each token a scalar `x`:

| token | `x` | `q = 1.0x` | `k = 0.5x` | `v = 2.0x` |
|---|---|---|---|---|
| I | 1.0 | 1.0 | 0.5 | 2.0 |
| am | 2.0 | 2.0 | 1.0 | 4.0 |
| a | 3.0 | 3.0 | 1.5 | 6.0 |

Scores are `q_i * k_j`, masked to `j <= i`, then softmaxed across the row:

| | vs I | vs am | vs a | softmax | output |
|---|---|---|---|---|---|
| **I** | 0.5 | — | — | `[1.0]` | `1.0(2.0)` = **2.000** |
| **am** | 1.0 | 2.0 | — | `[0.269, 0.731]` | `0.269(2) + 0.731(4)` = **3.462** |
| **a** | 1.5 | 3.0 | 4.5 | `[0.039, 0.175, 0.786]` | `0.039(2) + 0.175(4) + 0.786(6)` = **5.493** |

### Step 2: input is "I am a cat"

`x_cat = 4.0`, so `q = 4.0`, `k = 2.0`, `v = 8.0`. Recompute the whole thing:

| | vs I | vs am | vs a | vs cat | softmax | output |
|---|---|---|---|---|---|---|
| **I** | 0.5 | — | — | — | `[1.0]` | **2.000** ← same |
| **am** | 1.0 | 2.0 | — | — | `[0.269, 0.731]` | **3.462** ← same |
| **a** | 1.5 | 3.0 | 4.5 | — | `[0.039, 0.175, 0.786]` | **5.493** ← same |
| **cat** | 2.0 | 4.0 | 6.0 | 8.0 | `[0.002, 0.016, 0.117, 0.865]` | **7.690** ← new |

Nothing in the first three rows changed across consequent generations. Reasons:

1. `q_i`, `k_i`, `v_i` depend only on `x_i` — each token's projection reads its own vector and nothing else. Adding a fourth token can't touch the first three.
2. The "causal mask" means row "a" never has a `vs cat` column at all. The new token is invisible to every position before it.

### Step 3: the next block

Residual add: `z_i = x_i + out_i`.

| token | `x` | `out` | `z` |
|---|---|---|---|
| I | 1.0 | 2.000 | 3.000 |
| am | 2.0 | 3.462 | 5.462 |
| a | 3.0 | 5.493 | 8.493 |
| cat | 4.0 | 7.690 | 11.690 |

`z_1`, `z_2`, `z_3` are unchanged because both terms feeding them are unchanged. LayerNorm and the FFN are *per-token operations*, so block 2's *input* for the old tokens is identical, which means block 2's `k` and `v` for the old tokens are identical too. By induction, it holds for all 12 blocks.

## The general proof

**Claim:** when "cat" is appended to "I am a", the hidden states of "I", "am", "a" are identical at every layer.

**Base case (input layer).** Each old token's input is its token embedding + its position embedding. Neither depends on "cat", and the old positions don't shift because we append at the end.

**Inductive step.** Assume the old tokens' inputs to block `l` are unchanged. Walk the block:

- **Q, K, V projections** — `k_i = W_k · x_i`, computed from that token's own vector alone. Unchanged in, unchanged out.
- **Attention** — the output at position `i` uses `q_i` and the `k_j, v_j` for `j <= i` only. "cat" sits at a later position, so the mask hides it from every old token.
- **Residual add, LayerNorm/RMSNorm, FFN** — all per-token, no mixing across positions.

So the block's outputs for old tokens are unchanged, and by induction this holds for every layer. ∎

Attention is the *only* operation in the whole stack that mixes information across tokens, and the causal mask is what stops that mixing from reaching backwards. Which is also why none of this works for BERT — with no mask, "I" attends to "cat" and every vector shifts.

## Why cache K and V but not Q

First, Q stays unchanged as well. Reason we don't cache Q for prior tokens is because it is not needed.

For generating next token for "I am a cat", you only need the enriched context vector at the final layer for "cat" because that's what predicts the next token. So we inference just "cat" through the network. So, for each layer:
- "cat" computes its own `q`, `k`, `v`.
- `k` and `v` are added to the KV cache.
- "cat"'s `q` attends over the `k`s and `v`s of all tokens (including its own).

Per-step work drops from reprocessing the whole sequence to processing one token plus one attention read over the cache. That read is what makes decode memory-bandwidth-bound rather than compute-bound.

## Serving many requests

Aha! This means you can hold 1000s of requests' Ks and Vs in cache and just batch the *next* token for all of them through the network together. "1000s of requests" is technically nothing but batch size.

Per decode step, for each request:
- one token in, so the batch going into the block is `(1000, 768)`
- that token's `q`, `k`, `v` are computed, and its `k` and `v` are appended to *that request's own* cache
- its `q` (`1 x 768`) attends across its cached `K` generates an attention weight vector of length `context_length_i`
- those weights multiply the cached `V` generating a context-enriched vector corresponding to the token
- repeat through all the layers

The catch: each request is at a different point in its own sequence generation, so the cache lengths differ. `req0` holds `K = (context_length_0, 768)` and `V = (context_length_0, 768)`, `req1` holds something else. Stack them and you get a **ragged** batch rather than a rectangle. Apparently this problem is resolved by PagedAttention and continuous batching (deep-dived in the follow on section).

## Where it breaks: memory

### Per-request cost for GPT-3

```
per token = 2 (K & V) × 96 layers × 12288 dims × 2 bytes
          ≈ 4.7 MB per token
× 2,000-token context ≈ 9.4 GB per request

total KV memory servable (- model weights themselves) = batch size × context length × (size per token)
```

A top-end GPU roughly has 80–190 GB, and the weights take a big chunk of that. So with a vanilla KV cache, we might practically fit a few dozen requests, not 1000! For a given GPU, we'd have to reduce our per-request KV cache memory footprint to process more batches of requests. Currently, every decode step has to stream every request's whole cache through memory, therefore decode with a vanilla KV cache is memory-bandwidth-bound.

### Reducing the footprint

- **MLA (DeepSeek)** — store a small compressed latent instead of full K and V, which cuts the cache many times over.
- **Local / sliding-window layers** — keep only the last N tokens' K and V in some layers.
- **Cross-layer sharing** — layers reuse each other's K and V.
- **PagedAttention (vLLM)** — store the cache in fixed-size pages, like OS virtual memory, so ragged lengths don't waste GPU memory.

## Open threads

- **Memory-bandwidth-bound vs compute-bound.** Deep dive on <https://horace.io/brrr_intro.html> to understand compute-bound vs HBM-bound and how to improve GPU utilization.
- **Continuous batching.** Deep dive on <https://huggingface.co/blog/continuous_batching>. Current understanding is handwavy — there is more to it.
