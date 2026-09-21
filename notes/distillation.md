# Knowledge Distillation

Training a smaller **student** model using a large trained **teacher** model. Came up while working through [`PEFT.md`](PEFT.md) — the two get confused because both end up with something smaller, but they work on completely different axes. LoRA modifies weight matrices; distillation only ever touches the loss function. The teacher's weights are never opened.

## The core idea: soft labels instead of hard ones

Normal training uses **hard labels** — one-hot. For `"The capital of France is"`, the target is `1` for Paris and `0` for every other token in the vocab.

A trained teacher doesn't output a hard label. Its final projection gives logits, softmax turns them into a distribution, and that distribution carries far more than the answer. Divide the logits by a temperature `T` before the softmax and it becomes visible (vocab shrunk to 5 for illustration):

| | `T = 1` | `T = 4` |
|---|---|---|
| Paris | 0.94 | 0.46 |
| London | 0.05 | 0.22 |
| Rome | 0.010 | 0.15 |
| Berlin | 0.006 | 0.13 |
| Mars | 0.00004 | 0.04 |

Under the hard label, London and Mars are *equally wrong*. The teacher disagrees: London is a very plausible mistake, Berlin slightly less so, Mars is grammatically and factually absurd. That ranking is structural knowledge about how the vocabulary relates — Hinton called it **dark knowledge**.

Training the student to reproduce that whole spread teaches it the relationships between words and concepts much faster than learning from raw text alone.

## Temperature is what makes it work

At `T = 1` the teacher is nearly one-hot — Paris takes 94% of the mass and the interesting ratios sit at 0.05 and 0.00004, too small to move the loss. Dividing the logits by `T` (typically 2–5) flattens the distribution so those ratios actually register. Both models use the same `T`.

Without temperature, distillation degenerates back toward ordinary hard-label training.

## The loss

Both models run forward on the same input. The teacher is frozen and in eval mode; the gradient flows only into the student.

```python
teacher.eval()
opt = torch.optim.AdamW(student.parameters())   # teacher not in the optimizer

for x, y in loader:
    with torch.no_grad():                # teacher builds no autograd graph
        t_logits = teacher(x)

    s_logits = student(x)                # student does

    soft = F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),
        F.softmax(t_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T ** 2)

    hard = F.cross_entropy(s_logits, y)  # ordinary training on the real label

    loss = alpha * soft + (1 - alpha) * hard
    loss.backward()                      # reaches student weights only
    opt.step()
```

- **KL divergence** between the two distributions is the soft term; plain cross-entropy on the real label is the hard term. `alpha` trades them off.
- **`T ** 2`** is bookkeeping. The soft term's gradients scale as `1/T²`, so multiplying it back keeps the two terms comparable in magnitude.
- **The teacher can't be updated**, structurally, for two independent reasons: `no_grad` keeps its ops out of the autograd graph, and the optimizer was built over `student.parameters()`, so `opt.step()` has nothing to touch even if a gradient existed.


## Notes

- If the dataset is fixed, run the teacher **once**, dump its soft distributions to disk, delete the teacher, and train the student off the cached file. Distillation still works. That's how thoroughly the teacher isn't a participant in training — it's a data-generating step. It only needs to be co-resident when inputs are produced on the fly (on-policy distillation, where the student generates and the teacher scores), and even then it's only ever scoring.
- The student can have a completely different architecture from the teacher — fewer layers, narrower, different block design. Nothing in the loss requires them to match; only the output vocabulary has to.

## Open questions

- **The KL direction — is distillation forward or backward KL, and why does that choice matter?**
  - First pin the direction from the code: `F.kl_div(input=log_softmax(student), target=softmax(teacher))` computes `KL(target || input) = KL(teacher || student)`. So the *teacher* is the reference `p`, the *student* is the model `q` → this is **forward KL**, `KL(p || q)`. (Double-check PyTorch's arg convention yourself — it's a classic footgun.)
  - Now the intuition to build: forward KL is **mass-covering / mean-seeking** (zero-*avoiding*) — `q` is punished hard for putting ~0 probability where `p` has mass. Work out *why* that's exactly what distillation wants: we're trying to transfer the whole "dark knowledge" spread (London/Rome/Berlin), so the student must **cover** the teacher's full distribution, not collapse onto Paris. What would go wrong if you used backward KL here?
  - **Deep-dive cross-link (this is the unlock for the Nathan Lambert KL section you're stuck on):** RLHF uses the *opposite* direction — **backward KL** `KL(q || p)`, which is **mode-seeking** (zero-*forcing*). Reconcile it: why does distillation want mass-covering but RLHF-as-regularizer want mode-seeking? Same math, opposite goals. Get this and both click at once.
  - Threads to pull for the geometry: forward vs backward KL asymmetry, why KL isn't a distance, and what each direction does to a multi-modal `p` (spread across all modes vs. lock onto one).
