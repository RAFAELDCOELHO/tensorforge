# scripts/

## Fixture generators

The fixtures in `fixtures/` are this repository's external oracle: weights,
inputs, and reference results frozen from a **real, trained** PersonaCore. They
are what `tests/test_parity.py`, `tests/test_optim.py`, and
`tests/test_train.py` check against to prove this engine produces the same
numbers PyTorch does.

They are not versioned: they total ~157 MB and derive from a checkpoint that is
gitignored inside PersonaCore itself. The tests that depend on them **skip
themselves** when they are absent (`pytest.mark.skipif`), so the suite runs on a
clean clone — just without the parity layer.

### Why they need PersonaCore's venv

tensorforge has no PyTorch installed, and that is the point of the project, not
an oversight: the goal is for nothing here to depend on it at runtime. But
generating the reference requires doing exactly that — running PyTorch and
reading a `.pt`, which is a torch pickle.

The resolution is separation in time: PyTorch runs **once**, inside
PersonaCore's venv, writes plain-NumPy `.npz` files, and from then on the tests
only ever read NumPy. Three of the four scripts below must be executed with that
venv's interpreter.

### How to run them

The PersonaCore repository is looked up in `$PERSONACORE_ROOT` and, failing
that, in the sibling `../PersonaCore`. The scripts fail with a clear message if
they cannot find the repository or if they are run without torch.

```bash
export PERSONACORE_ROOT=/path/to/PersonaCore   # optional if it is a sibling of this repo
PC=$PERSONACORE_ROOT/.venv/bin/python

$PC scripts/gen_parity_fixture.py     # FIRST — the other two depend on it
$PC scripts/gen_adamw_fixture.py
$PC scripts/gen_clip_fixture.py

python scripts/gen_val_windows_fixture.py   # this one does NOT need torch
```

The order matters: `gen_parity_fixture.py` is what computes the real gradients,
and the AdamW and clipping generators reuse those same gradients rather than
recomputing them — that is what makes the three tests speak about the same data.

The scripts are deterministic: regenerating produces **byte-for-byte identical**
files (verified).

### What each one produces

| script | fixture | needs torch | consumed by |
|---|---|---|---|
| `gen_parity_fixture.py` | `personacore_parity.npz` (91 MB) | yes | `test_parity.py`, `test_train.py`, `demo_train.py` |
| `gen_adamw_fixture.py` | `personacore_adamw.npz` (43 MB) | yes | `test_optim.py` |
| `gen_clip_fixture.py` | `personacore_clip.npz` (24 MB) | yes | `test_optim.py` |
| `gen_val_windows_fixture.py` | `val_windows.npz` (34 KB) | no | `demo_train.py` |

- **parity** — the 101 tensors of the `state_dict` (float32, losslessly cast to
  float64 on read), a real 257-token window from `data/val.bin`, and the logits,
  the loss, and sampled gradients that PyTorch produces in float64.
- **adamw** — the real optimizer state at step 49000 (`exp_avg`, `exp_avg_sq`,
  `step`) for four parameters, plus the result of one `torch.optim.AdamW` step,
  plus a control run with **coupled** weight decay for the discriminating test.
- **clip** — the output of `torch.nn.utils.clip_grad_norm_` across three
  `max_norm` regimes (clips / no-op / aggressive clip) and on the boundary case
  where the norm is exactly equal to the limit.
- **val_windows** — 16 windows spaced across the corpus, for the demonstration.
  `val.bin` is a plain `uint16` memmap, which NumPy reads without any torch.

`_fixture_paths.py` is just the shared path helper — not an executable.

## Demonstration

`demo_train.py` generates nothing; it consumes `personacore_parity.npz` and
`val_windows.npz` and trains the real model on this engine. It runs with this
repository's Python, without torch. See the root README.
