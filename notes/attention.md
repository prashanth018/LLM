# Attention — misconceptions

## The split into heads happens *after* the Q/K/V projection, not before

**Misconception:** The 768 dims coming out of the embeddings get carved into 12 blocks of 64 first, and then each head projects its own block. Under this picture the pipeline is:

1. Embeddings out: `(b, T, 768)`.
2. Slice the last dim into 12 blocks. Head 0 gets dims `[0:64]`, head 1 gets `[64:128]`, and so on — each head now holds `(b, T, 64)`.
3. Head 0 owns three learnable `64 x 64` matrices `W_q, W_k, W_v` and pushes its block through them, getting Q, K, V each of shape `(b, T, 64)`.
4. Scores, mask, scale, softmax, multiply by V.
5. Head 0 pushes its result through its own `64 x 64` output matrix.

**GT:** Nothing is sliced until *after* Q, K and V have been computed from the full-width token. The real pipeline is:

1. Embeddings out: `(b, T, 768)`.
2. One single linear layer `c_attn`, weight shape `768 x 2304`, applied to the whole thing. Out: `(b, T, 2304)`.
3. `.chunk(3, dim=-1)` splits that into Q, K, V — each `(b, T, 768)`. Still full width. No head exists yet.
4. *Now* the heads appear, and they appear by **reshaping**, not by projecting: `(b, T, 768) -> (b, T, 12, 64) -> transpose -> (b, 12, T, 64)`, for each of Q, K, V. This step has no weights in it at all.
5. `Q @ K.transpose(-2, -1)` gives `(b, 12, T, T)`. Mask the upper triangle with `-inf`, divide by `sqrt(64)`, softmax, multiply by V. Out: `(b, 12, T, 64)`.
6. Transpose and reshape the heads back together into `(b, T, 768)`.
7. One output projection `c_proj`, a single `768 x 768` matrix — not a `64 x 64` per head.

The one-sentence version: **the head boundary is a slice through the *output columns* of the projection weight, never a slice through the *input dims* of the token.** Head 0's query vector is a projection of all 768 input dims — it's `x @ W[:, 0:64]`, using every row of `W`. In the misconception it would have been `x[0:64] @ W`, using only the first 64 rows.

### Why it has to be this way (the modeling reason)

If you sliced the input first, head 0 would spend the entire life of the model looking at dims 0–63 of a token and *nothing else*. Head 1 would only ever see dims 64–127. The heads would be permanently reading disjoint chunks of the representation.

That defeats the point of multi-head attention. The premise is that each head looks at the **complete** token and extracts a different relationship from it — one head learns to track syntactic agreement, another tracks coreference, another tracks position-ish patterns — and all of them need the whole vector to do that. The embedding was never trained to partition its 768 dims into 12 self-contained semantic groups, so there's no reason dims 0–63 alone would carry enough signal for a head to work with.

So each head needs its own `768 -> 64` projection, three times over (Q, K, V). That's `12 heads x 3 x 64 = 2304` output dims.

### Why it's one fat 2304 matrix (the hardware reason)

The `768 x 2304` shape looks like the model is blowing the representation up to 3x width. It isn't. There's no extra capacity here — 2304 is just **36 separate `768 -> 64` projections parked side by side in one matrix**.

Writing it as 36 small matmuls and writing it as one big matmul produce mathematically identical results. The difference is purely execution: GPUs are built to run one large GEMM efficiently, and 36 small ones means 36 kernel launches, each one hitting memory-bandwidth overhead and leaving the GPU stalled between them. So the weights get fused into one layer.

A way to check you've got the right mental model: count the parameters. Per-head `64 x 64` matrices would be `12 x 3 x 64 x 64 = 147K` params. The real `c_attn` is `768 x 2304 = 1.77M`, ~12x more — because every head gets a full-width `768 x 64` projection instead of a narrow `64 x 64` one.

### Why chunking into thirds isn't arbitrary

It's tempting to ask how the model knows the first chunk should be Q. It doesn't know, and at init the whole matrix is just random noise. What fixes the assignment is that the *code* always routes the first 768 columns into the Q slot, the second into K, the third into V — a fixed structural contract that never changes.

Training does the rest. Backprop flows back from `Q @ K.T`, and the gradient that reaches the first chunk's weights is the gradient of "being a query." Those columns become query-like because that's the only role the loss ever evaluates them in.

### The output projection mixes across heads

Last thing, since the misconception also puts a `64 x 64` output matrix inside each head: the real `c_proj` is a single `768 x 768` applied after the heads are concatenated. That matters — a per-head `64 x 64` could only ever recombine a head's own output, whereas one `768 x 768` lets every output dim read from **all 12 heads at once**. Merging the heads' findings is the job of that layer.

---

## The causal mask kills the upper triangle, not the lower

**Misconception:** Null out the lower triangle of the `(T, T)` score matrix.

**GT:** Everything strictly *above* the diagonal gets set to `-inf` before the softmax.

**Why:** Read the score matrix by its indices. Row `i` is the token doing the querying; column `j` is the token being attended to. So entry `(i, j)` means "how much token `i` attends to token `j`."

A token is allowed to look at itself and everything before it, and forbidden from looking ahead. "Ahead" means `j > i` — and `j > i` is exactly the region above the diagonal. For `T=4`, `.` is allowed and `x` is masked:

```
        j=0  j=1  j=2  j=3
i=0      .    x    x    x
i=1      .    .    x    x
i=2      .    .    .    x
i=3      .    .    .    .
```

Row 0 is the first token, which can only see itself. Row 3 is the last token, which can see everything. The diagonal always stays.

Two details worth holding onto:
- The mask is `-inf`, not `0`, and it's applied **before** softmax. `exp(-inf) = 0`, so those positions come out of the softmax as exactly zero and contribute nothing to the weighted sum of V. Writing `0` into the scores instead would give `exp(0) = 1` — a *uniform* weight on every future token, the opposite of what you want.
- Because the masked entries are zeroed after softmax, the remaining weights in each row still sum to 1. Row `i` distributes its full attention budget over the `i+1` tokens it's allowed to see.
