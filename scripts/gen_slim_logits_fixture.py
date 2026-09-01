"""Freeze public-slim forward logits. No private checkpoint, no optimizer.

Needs torch + the public m1-demo-v1 ``model_slim.pt`` + PersonaCore source
(for the PyTorch GPT). Does NOT read ``best.pt``, ``val.bin``, or any
optimizer state — those are not in the slim and are not claimed here.

The committed fixture is small (~1 MB of logits, not the 55 MB weights).
Tests compare the frozen PyTorch logits to the frozen tensorforge logits
from the same public weights and input, so a fresh clone can check the
measurement without downloading the slim. Re-running this script
re-derives the same arrays from the public release.
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fixture_paths import out, personacore_root, report, require_torch  # noqa: E402
from slim_zip import (  # noqa: E402
    PUBLISHED_CONFIG,
    find_slim,
    gpt_from_slim,
    load_slim_zip,
)

require_torch()
import torch  # noqa: E402

ROOT = personacore_root()
sys.path.insert(0, os.path.join(ROOT, "src"))
from personacore.checkpoint import load_slim  # noqa: E402
from personacore.config import ModelConfig  # noqa: E402
from personacore.model.gpt import GPT  # noqa: E402

# Documented public token ids — not a window from private val.bin.
# 8184 is the published eos_id; 1..15 are in-vocab.
INPUT_X = np.array(
    [8184, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    dtype=np.int64,
)

slim_path = os.environ.get("PERSONACORE_SLIM") or find_slim()
if not slim_path or not os.path.isfile(slim_path):
    sys.exit(
        "public model_slim.pt not found. Download m1-demo-v1 and set "
        "PERSONACORE_SLIM, or place it at checkpoints/model_slim.pt"
    )

slim_sha = hashlib.sha256(open(slim_path, "rb").read()).hexdigest()
loaded = load_slim(slim_path)
for dropped in ("optimizer", "scheduler", "scaler", "rng", "train_config"):
    if dropped in loaded:
        sys.exit(f"refusing to treat this file as slim: has {dropped!r}")
if loaded["model_config"] != PUBLISHED_CONFIG:
    sys.exit(f"unexpected slim model_config: {loaded['model_config']}")

cfg = ModelConfig(**loaded["model_config"])
pt_model = GPT(cfg, attn_impl="manual")
pt_model.load_state_dict(loaded["model"])
pt_model.double().eval()
x = torch.tensor(INPUT_X).unsqueeze(0)
with torch.no_grad():
    pt_logits, pt_loss = pt_model(x, None)
if pt_loss is not None:
    sys.exit("forward without targets must return loss=None")
ref = pt_logits.detach().cpu().numpy()[0].astype(np.float64)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tf_model = gpt_from_slim(load_slim_zip(slim_path))
engine = tf_model(INPUT_X[None, :]).data[0].astype(np.float64)

rel = float(np.abs(engine - ref).max() / np.abs(ref).max())
print("measured relative error vs PyTorch: %.6e" % rel)
print("argmax identical:", bool(np.array_equal(engine.argmax(-1), ref.argmax(-1))))

path = out("slim_logits_public.npz")
np.savez_compressed(
    path,
    input_x=INPUT_X,
    ref_logits=ref,
    engine_logits=engine,
    measured_rel_error=np.float64(rel),
    git_sha=np.array(loaded["git_sha"]),
    step=np.int64(loaded["step"]),
    slim_sha256=np.array(slim_sha),
    source=np.array("m1-demo-v1/model_slim.pt"),
)
report(path)
