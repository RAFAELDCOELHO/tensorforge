# tensorforge

A deep learning framework built from scratch. The final proof: retraining the
entire PersonaCore running on this engine alone, with no PyTorch underneath.

## Status

- [x] **M0 — Scalar autograd** — `Value`, computation graph, `.backward()`,
      ops: `+ * ** neg sub truediv relu tanh exp`.
- [x] **M1 — Tensor autograd** — `Tensor` over NumPy, broadcasting via
      `_unbroadcast`, `matmul` with arbitrary batch dimensions.
- [x] **M2 — Modules and optimizers** — `Linear`, `LayerNorm`, `SGD`, `Adam`.
- [x] **M3 — Transformer** — `softmax`, `reshape`/`transpose`, SDPA with causal
      mask, `Embedding`, `tanh`, `gelu`, `CausalSelfAttention`, `MLP`, `Block`,
      `GPT` with weight tying.
- [x] **M4 — PersonaCore training on this engine** — `log_softmax`, `pick`,
      `cross_entropy`, `AdamW`, `lr_at_step`, `clip_grad_norm_`, `train_step`.
      Numerical parity against the real checkpoint, validated below.
- [ ] M5 — Stretch: MLX backend (native GPU on the M3 Pro, no cloud cost)

**449 tests.** Every cycle followed the same protocol: the derivation is written
before the test, TDD red→green, and a mutation check at the end — mutations are
applied line by line and the suite has to catch each one. Surviving mutants are
investigated and accepted only when provably equivalent.

## Parity with PersonaCore

The engine is validated against an external oracle, not only against references
written here. `tests/test_parity.py` loads the 13.9M weights from
`checkpoints/best.pt` (trained for 49k steps) and runs a real window from
`data/val.bin`:

| | relative error vs PyTorch (float64) |
|---|---|
| logits (256×8192) | 2.7e-15 |
| loss | 0.0 (exact) |
| sampled gradients | 3.7e-15 to 9.6e-15 |
| one AdamW step from the real optimizer state | 0.0 to 2.5e-16 |
| `clip_grad_norm_` on real gradients | ~1e-15 |

The argmax is identical token for token across all 256 positions. The
tolerances are bounds derived in writing (`sqrt(K)·eps` accumulated over the
depth), not values found by trial; the measured errors came in 90x to 400x
below them.

What is **not** reproduced, by principle rather than by bug: the training
trajectory. PersonaCore trained in float32 on MPS; this is float64 in NumPy.
The difference at the first step is ~1e-7, and training amplifies that
exponentially. The validated target is **the step**, not the curve.

## Decisions and findings

### Backward-pass determinism bug (found in M4)

`Tensor._prev` (and `Value._prev`) held a node's children in a `set`. Iterating
a `set` of Python objects follows hashes derived from `id()`, which move between
processes — so the reverse topological order changed from run to run, and with
it the order in which the `+=` accumulations landed in `.grad`. Floating-point
addition is not associative: **two identical runs diverged by ~1 ulp per
parameter.**

How it was caught: the test comparing `train_step` against the manual
composition of the pieces demands **byte-for-byte** equality. It failed. Before
touching anything, the hypothesis was tested by running the *same* route twice
— which also diverged, ruling out "train_step is wrong" and pointing at the
engine. Printing the topological order on two runs confirmed it.

Why it matters here rather than being harmless noise: PersonaCore's resume
contract is a bit-for-bit identical trajectory, and the goal of this repository
is to retrain that model. A non-deterministic gradient makes any bit-level
reproducibility impossible.

Fix: `_prev` became a tuple in both files. Duplicates are harmless —
`build_topo` already de-duplicates via `visited`, so `x + x` still accumulates
twice. Pinned by tests in `tests/test_train.py` and `tests/test_engine.py`.

The methodological lesson: a test for **exact** equality found a defect no
`allclose` would have found. A loose tolerance hides ordering bugs.

### Other recorded decisions

- **Causal mask uses `-1e9`, not `float("-inf")`.** A fully masked row (padding,
  not used yet) would give `-inf - (-inf) = nan` in the softmax stability shift,
  and the `nan` spreads through the batch in the backward pass. With `-1e9` the
  row degrades to uniform instead. Numerically identical to PersonaCore's `-inf`
  in the causal case, where the diagonal is never masked.
- **`Linear` stores `W` as `(in, out)`**, unlike PyTorch, which stores
  `(out, in)` and transposes in the forward pass. The weight loader transposes.
  Since the attention projections are square, a forgotten transpose there changes
  no shape at all — only comparing values against torch catches it.
- **`weight_decay` is applied to every parameter**, including biases and
  LayerNorm gains. The nanoGPT convention is to exclude 1-D tensors; PersonaCore
  does not (a single param group, 100 tensors), and the target is the real
  PersonaCore.
- **`lr_at_step(step)` with a 0-based `step`.** In PersonaCore the
  `scheduler.step()` comes *after* the `optimizer.step()`, and `LambdaLR` already
  advances once during construction, so iteration *k* uses `λ(k−1)`. The lr
  recorded in the checkpoint at `step=49000` was never applied by any step: it is
  the value staged for the following iteration.
- **Adam's `eps` sits outside the square root**, `lr·m̂/(√v̂ + eps)`, following
  PyTorch.
- **No AMP.** The reference run was on MPS, where PersonaCore's `RuntimeConfig`
  disables AMP; the checkpoint's `GradScaler` state is empty. Pure fp32 is the
  faithful replica, not a simplification.
- **No gradient accumulation.** The real run uses `grad_accum_steps=1`.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

The parity tests skip themselves when the fixtures are absent, so the suite runs
on a clean clone — just without the parity layer. Generating them requires the
PersonaCore repository and its venv (with torch): see
[`scripts/README.md`](scripts/README.md).

## End-to-end demonstration

```bash
python scripts/demo_train.py
```

Loads the real weights and trains on this engine, in two regimes. Overfitting a
single batch (the classic gate, which proves the engine *learns*): loss from
0.756 to 0.006 in 15 steps, ~1.5s/step for 13.9M parameters in NumPy. And a
faithful continuation at the real learning rate from step 49000 (which proves
the engine *runs* in the real regime): the loss oscillates with the batch and
does not descend, exactly as expected from an already-converged model.

## Usage

```python
import numpy as np
from core.nn import GPT
from core.optim import AdamW
from core.train import train_step

model = GPT(vocab_size=8192, n_embd=384, n_head=6, n_layer=6, block_size=256)
opt = AdamW(model.parameters(), weight_decay=0.1)

loss, grad_norm = train_step(model, opt, x, y, step=0, max_norm=1.0,
                             base_lr=3e-4, warmup_steps=100, max_steps=50000)
```
