# Parameter-Efficient Fine-Tuning (PEFT)

This explores: *LoRA, LoRA-FA, VeRA, Delta-LoRA, LoRA+* techniques

## Low-Rank Adaptation (LoRA)

**Intuition:** `Wx + (alpha / r) * BAx`, where `B` is `768 x 16` and `A` is `16 x 768`.

`BAx` reads right to left, so `x` meets `A` first: `A` compresses 768 down to 16, and `B` expands 16 back up to 768.

For fine-tuning, we don't have to train the weights of the huge `W` matrix. We aren't relearning the structure of the language — we're only learning the *preferences* and *alignment* for the task at hand. For that, a rank of 16 should be enough.

So we project the input down into a lower rank (16 here) and then project it back up to the original dim. The goal was never to relearn `W`; instead we let the `delta` percolate into the learnable `A` and `B`. Everything else stays frozen — `W` never receives a gradient.

Applied to the `Q` and `V` matrices by convention, but in general it can go on any learnable matrix.

## LoRA-FA (Frozen `A`)

Standard LoRA trains both matrices — the compression `A` and the expansion `B`. LoRA-FA initializes `A` with random weights and freezes it immediately. The input gets compressed through that static, random projection, and only `B` receives gradients, learning to unpack it into the new task distribution.

So the compression isn't learned — only what to do with the compressed vector is.

**Note regarding trainable parameters:** `A` is `16 x 768` and `B` is `768 x 16`, the same size, so freezing `A` halves the trainable count. That only holds for a square matrix. On `c_attn` (`768 -> 2304`), `A` is still `16 x 768` = 12,288 params but `B` is `2304 x 16` = 36,864, so freezing `A` drops 12,288 out of 49,152 — a quarter, not a half.

**Activation memory.** To compute the gradient of a weight matrix, you need the input that was fed into it on the forward pass, so that input has to stay in memory until backward runs.

- `A`'s input is `x` — 768 floats per token.
- `B`'s input is `Ax` — 16 floats per token.

Training both means holding onto the 768-dim `x`. With `A` frozen there's no `A` gradient to compute, so only the 16-dim `Ax` needs to be kept.

## VeRA (Vector-based Random Matrix Adaptation)

Pushes the parameter reduction to the extreme. Both `A` and `B` are initialized randomly and completely frozen — neither is trained. What gets trained instead is two scaling vectors, `b` and `d`:

`delta_W = S_b · B · S_d · A`, where `S_b` and `S_d` are diagonal matrices built from `b` and `d`.

Reading right to left: `A` compresses 768 → 16, `S_d` rescales the 16 dims, `B` expands 16 → 768, `S_b` rescales the 768 outputs. So the projections are fixed and only their per-dimension gains are learned.

`b` has length 768 and `d` has length 16, so that's 784 trainable params per adapted matrix against LoRA's 24,576.

The other half of the saving: `A` and `B` are **shared across all layers** — generated once from a fixed seed and reused everywhere. Only `b` and `d` are per-layer.

## Delta-LoRA
***best of them all. Personal favourite!*

Standard LoRA leaves `W` frozen forever, so the total adaptation is permanently capped at rank `r`. Delta-LoRA updates `W` as well, without ever giving it a gradient or an optimizer state.

`W_{t+1} = W_t + c * (B_{t+1} A_{t+1} - B_t A_t)`

Per step: backprop as usual with only `A` and `B` in the optimizer. Once the optimizer has stepped them, take the difference between the new `BA` product and the old one, scale it by `c`, and add it to `W` in place. That addition is a detached tensor op — autograd never sees it, so the moment buffers are never allocated for `W`.

**Why the delta of `BA` is a valid direction for `W`.** Since `h = Wx + BAx`, the two paths sit in parallel on the same input, so `dL/dW` and `dL/d(BA)` are the same up to a constant. The change in `BA` already *is* the update `W` would have received.

**Where the extra capacity comes from.** Each step adds a rank-`r` change, but rank is subadditive. This means, **theoritically**, two different rank-16 updates sum to something up to rank 32. Thousands of accumulated rank-16 shifts drift `W` through a high-rank subspace, so the total adaptation escapes the rank ceiling even though every individual step respects it.

**Memory footprint is identical to standard LoRA.** Delta-LoRA still saves the first and second moments in the optimizer like the LoRA so from parameteric perspective both are economically similar. The main win is being able to represent more than what the rank can theoretically represent over the given training epochs. This means, you can do SFT longer and on complex tasks and see meaningful gains.

**What it costs.** `W` is permanently modified, so the base model is no longer shared. LoRA's practical selling point is: "one frozen base plus many swappable few-MB adapters". This will no longer be the case and each run produces a full fine-tuned model.

Two implementation details from the paper: dropout is removed from the adapter (the `dL/dW = dL/d(BA)` equivalence needs both paths to see the same input), and the `W` updates only start after a warmup of `K` steps.

## LoRA+
Same as standard LoRA but uses a faster learning rate for B (usually 2x to 16x higher than A). Apparently I using the same learning rate is inefficient and the above set up leads to faster convergence. I couldn't get to the underlying intuition.

## Follow-on questions

### Why not factor every `n x n` matrix into `n x r` @ `r x n` at train time? Would the encode/decode through a lower rank lose information?

Yes. When you multiply an `n x r` matrix by an `r x n` matrix, the result has the *shape* `n x n`, but its mathematical rank can never exceed `r`. With `n = 768` and `r = 16`, a dense matrix can represent 768 independent, orthogonal feature directions. The `B @ A` product, despite being `768 x 768` in shape, is fundamentally bottlenecked to 16 independent features.

That distinction is exactly the difference between pre-training and fine-tuning:

- **Pre-training (building the world model).** Starting from random weights, the network has to chew through terabytes of data to learn grammar, logic, facts, and reasoning. It needs the full rank — all 768 degrees of freedom — to store that distribution of world knowledge. Pre-train GPT-2 from scratch with low-rank matrices and it hits a severe information bottleneck and massively underfits.
- **Fine-tuning (shifting preferences).** LoRA works *because* the frozen `W` already holds that full-rank knowledge. The low-rank bypass only has to learn a highly compressed delta — just enough capacity to steer output format, tone, or alignment without degrading the base intelligence.

Full matrices build the foundational intelligence; low-rank matrices are plenty for steering it.

### In that case, what does the FFN fundamentally do? It looks like the opposite of LoRA. The GPT paper calls it the *thinking layer*, but what actually happens when we blow up to 4x dim and come back down?

Geometrically, it *is* the inverse of LoRA. LoRA is an hourglass that forces compression; the FFN is an inverted hourglass that forces expansion.

The intuition starts with separating the two layers' jobs:

- **Attention** mixes information *across* tokens — which words in the sequence matter to each other.
- **FFN** processes information strictly *within* a single token. It takes the freshly contextualized token and retrieves facts, logic, and concepts about it.

Why it expands 768 → 3072 and comes back down (usually framed in research as a **key-value memory bank**):

1. **The "key" layer (expansion to 3072).** The first matrix (`768 x 3072`) acts as 3072 pattern-matching keys. The 768-dim embedding is dense and entangled; projecting it into a much larger space disentangles those features. It gives specific, granular concepts enough room to get their own dedicated neurons — one might detect "this token is a verb," another "this is related to Paris."
2. **The activation (GELU).** The threshold mechanism. Without a non-linearity, projecting up to 3072 and straight back down to 768 would collapse into a single `768 x 768` linear transform, neutralizing the expansion entirely. GELU acts as a gate: if a key pattern matches the input, the neuron fires; otherwise it's suppressed.
3. **The "value" layer (compression to 768).** The second matrix (`3072 x 768`) holds the values. For every neuron that fired, it maps that concept back down into 768-dim and writes the retrieved knowledge into the token's residual stream.

So the 4x expansion isn't about compressing data. It's about opening the token into a high-resolution workspace where specific concepts can be isolated, thresholded through a non-linearity, and written back into the token's core representation.

### Is this how distillation works too? Isn't it just `W = BzA`, where `z` is the smaller model?

No — different axis entirely. LoRA and SVD are *structural*: they open up the model and modify the weight matrices themselves. Distillation never looks at the teacher's weights at all. It's a *behavioral* transfer that happens entirely in the loss function — the student is trained to match the teacher's output distribution.

Three things that all produce "a smaller model," which is why they blur together:

| | what it touches | what comes out |
|---|---|---|
| **LoRA** | weight matrices, structurally | same model — frozen `W` plus a trainable low-rank bypass |
| **SVD / bottleneck** (`W = BzA`) | weight matrices, structurally | a genuinely smaller model |
| **Distillation** | only the loss function | a separate student model; the teacher is never opened |

The `W = BzA` intuition isn't wrong about anything — it's a real technique (autoencoders, ALBERT's factorized embeddings). It just belongs under architectural compression, not distillation.

Full note: [`distillation.md`](distillation.md).

---

## Glossary to revisit

- Intuition behind the rank of a matrix.
- Intuition behind the `alpha / r` scaling in LoRA.
- Why Delta-LoRA has to remove dropout from the adapter — what dropout does to the `dL/dW = dL/d(BA)` equivalence.
- Intuition behind different learning rates in LoRA+  
