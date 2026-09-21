# Parameter-Efficient Fine-Tuning (PEFT)

This explores: *LoRA, LoRA-FA, VeRA, Delta-LoRA, LoRA+* techniques

## Low-Rank Adaptation (LoRA)

**Intuition:** `Wx + BAx`, where `B` is `768 x 16` and `A` is `16 x 768`.

For fine-tuning, we don't have to train the weights of the huge `W` matrix. We aren't relearning the structure of the language — we're only learning the *preferences* and *alignment* for the task at hand. For that, a rank of 16 should be enough.

So we project the input down into a lower rank (16 here) and then project it back up to the original dim. The goal was never to relearn `W`; instead we let the `delta` percolate into the learnable `A` and `B`. Everything else stays frozen — `W` never receives a gradient.

Applied to the `Q` and `V` matrices by convention, but in general it can go on any learnable matrix.

---

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
