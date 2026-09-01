"""What the public PersonaCore slim actually allows on a fresh clone.

The v1.0 parity table is a private-checkpoint proof. This file pins the
weaker, public fact: ``m1-demo-v1`` / ``model_slim.pt`` is inference weights
plus ``model_config``. It matches this engine's GPT (6 / 6 / 384 / 256, tied
embed, 13,891,584 params). It does not carry optimizer state, a corpus window,
or published PyTorch grads, so backward-vs-PyTorch parity from this artifact
is impossible and is not claimed.

CI never downloads the ~55 MB file. Schema / param-count gates run on a
synthetic zip or on the engine alone. The real slim is skip-gated with a
loud reason when the file is absent.
"""
import os
import sys

import numpy as np
import pytest

from core.nn import GPT

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
from slim_zip import (  # noqa: E402
    DROPPED_TRAINING_KEYS,
    PUBLISHED_CONFIG,
    PUBLISHED_GIT_SHA_PREFIX,
    PUBLISHED_PARAM_COUNT,
    PUBLISHED_STEP,
    SLIM_KEYS,
    find_slim,
    gpt_from_slim,
    load_slim_zip,
    write_schema_only_zip,
)

LOGITS_FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir,
                              "fixtures", "slim_logits_public.npz")
# Same derived logits bound as tests/test_parity.py (sqrt(K)·eps over depth).
LOGITS_RTOL = 1e-11
LOGITS_CANARY = 1e-12
PUBLIC_INPUT = np.array(
    [8184, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    dtype=np.int64,
)
PUBLISHED_SLIM_SHA256 = (
    "dd3bbb8f772e0b9556a0a31d535a1673d55f0d61d6d669c58a9aab6bb6247e24"
)

_SLIM = find_slim()
_SKIP_REAL = pytest.mark.skipif(
    _SLIM is None,
    reason=(
        "public m1-demo-v1 model_slim.pt is not on this machine; CI must not "
        "download it. Set PERSONACORE_SLIM or place the file at "
        "checkpoints/model_slim.pt to run the real-artifact forward. "
        "The file is inference weights only — no published PyTorch grads."
    ),
)


def test_engine_gpt_matches_published_slim_architecture():
    """Failing-closed gate: this GPT is the published 13.9M tied PersonaCore.

    No slim file required. If layer/head/width/block/tying drift, this fails
    before anyone can pretend the public slim still maps onto the engine.
    """
    cfg = PUBLISHED_CONFIG
    model = GPT(cfg["vocab_size"], cfg["n_embd"], cfg["n_head"],
                cfg["n_layer"], cfg["block_size"],
                rng=np.random.default_rng(0))
    n = sum(int(p.data.size) for p in model.parameters())
    assert n == PUBLISHED_PARAM_COUNT
    assert not hasattr(model, "lm_head")
    assert model.block_size == cfg["block_size"]
    assert len(model.blocks) == cfg["n_layer"]
    assert model.blocks[0].attn.n_head == cfg["n_head"]
    assert model.wte.weight.shape == (cfg["vocab_size"], cfg["n_embd"])
    assert model.wpe.weight.shape == (cfg["block_size"], cfg["n_embd"])


def test_published_slim_schema_cannot_supply_pytorch_grad_oracle():
    """The published key set is inference-only. Do not invent backward parity."""
    assert DROPPED_TRAINING_KEYS.isdisjoint(SLIM_KEYS)
    assert "optimizer" not in SLIM_KEYS
    assert "ref_logits" not in SLIM_KEYS
    assert "ref_loss" not in SLIM_KEYS
    assert not any(k.startswith("g::") for k in SLIM_KEYS)
    # A corpus window would be extra keys; the slim does not have them.
    assert "input_x" not in SLIM_KEYS and "input_y" not in SLIM_KEYS


def _schema_payload(**overrides):
    payload = {
        "schema_version": 1,
        "model": {},
        "model_config": dict(PUBLISHED_CONFIG),
        "git_sha": "deadbeef0",
        "step": 1,
        "val_loss": 0.0,
    }
    payload.update(overrides)
    return payload


def test_synthetic_slim_zip_loads_schema_without_downloading_weights(tmp_path):
    """CI fixture: same key set as m1-demo-v1, no 55 MB file, no torch."""
    path = tmp_path / "synthetic_slim.pt"
    write_schema_only_zip(path, _schema_payload())
    loaded = load_slim_zip(path)
    assert set(loaded) == SLIM_KEYS
    assert loaded["model_config"] == PUBLISHED_CONFIG
    assert loaded["model"] == {}


def test_synthetic_slim_zip_rejects_training_state(tmp_path):
    path = tmp_path / "leaky.pt"
    write_schema_only_zip(path, _schema_payload(optimizer={"state": {}}))
    with pytest.raises(ValueError, match="published inference"):
        load_slim_zip(path)


def test_synthetic_slim_zip_rejects_missing_keys(tmp_path):
    path = tmp_path / "truncated.pt"
    write_schema_only_zip(path, {"schema_version": 1, "model": {}})
    with pytest.raises(ValueError, match="published inference"):
        load_slim_zip(path)


@_SKIP_REAL
def test_public_slim_config_matches_this_engine():
    slim = load_slim_zip(_SLIM)
    assert slim["schema_version"] == 1
    assert slim["model_config"] == PUBLISHED_CONFIG
    assert slim["git_sha"].startswith(PUBLISHED_GIT_SHA_PREFIX)
    assert slim["step"] == PUBLISHED_STEP
    assert DROPPED_TRAINING_KEYS.isdisjoint(slim)
    n = sum(v.size for k, v in slim["model"].items() if k != "lm_head.weight")
    assert n == PUBLISHED_PARAM_COUNT
    assert np.array_equal(slim["model"]["lm_head.weight"],
                          slim["model"]["wte.weight"])


def _relative_error(got, expected):
    return np.abs(got - expected).max() / np.abs(expected).max()


def test_public_slim_logits_match_pytorch_on_committed_fixture():
    """Forward/logits parity from m1-demo-v1, no 55 MB file, no private ckpt.

    The fixture was frozen by scripts/gen_slim_logits_fixture.py: PyTorch
    (attn_impl=manual, float64) and this engine, same public slim, same
    documented 16-token input. The arrays are the measurement; this test
    recomputes the relative error. It does not invent a number.
    """
    assert os.path.exists(LOGITS_FIXTURE), (
        "fixtures/slim_logits_public.npz must be committed — it is the "
        "public logits proof, not the 55 MB slim"
    )
    ref = np.load(LOGITS_FIXTURE)
    x, pt, tf = ref["input_x"], ref["ref_logits"], ref["engine_logits"]
    assert np.array_equal(x, PUBLIC_INPUT)
    vocab = PUBLISHED_CONFIG["vocab_size"]
    assert pt.shape == tf.shape == (16, vocab)
    assert pt.dtype == tf.dtype == np.float64
    assert np.isfinite(pt).all() and np.isfinite(tf).all()
    assert str(ref["slim_sha256"]) == PUBLISHED_SLIM_SHA256
    assert str(ref["git_sha"]).startswith(PUBLISHED_GIT_SHA_PREFIX)
    assert int(ref["step"]) == PUBLISHED_STEP
    err = _relative_error(tf, pt)
    stored = float(ref["measured_rel_error"])
    print(f"\n[public slim] logits rel. error vs PyTorch: {err:.6e} "
          f"(stored {stored:.6e}; rtol {LOGITS_RTOL:.0e})")
    assert err < LOGITS_RTOL, f"erro {err:.3e} — investigar arquitetura, não tolerância"
    assert err < LOGITS_CANARY, f"erro {err:.3e} above the derived 2.4e-13 canary"
    assert abs(err - stored) / max(stored, np.finfo(np.float64).tiny) < 1e-12
    assert np.array_equal(tf.argmax(-1), pt.argmax(-1))


def test_public_slim_cannot_support_m4_adamw():
    """Passing record: the slim has no optimizer, so M4 is not claimed."""
    assert DROPPED_TRAINING_KEYS.isdisjoint(SLIM_KEYS)
    assert os.path.exists(LOGITS_FIXTURE)
    ref = np.load(LOGITS_FIXTURE)
    assert "optimizer" not in ref.files
    assert "exp_avg" not in ref.files
    assert not any(name.startswith("g::") for name in ref.files)


@pytest.mark.skip(reason=(
    "public m1-demo-v1 model_slim.pt has no optimizer/scheduler state; "
    "M4/AdamW parity cannot be claimed from the slim. The private "
    "fixtures remain the only AdamW/grad oracle."
))
def test_m4_adamw_from_public_slim_is_not_claimed():
    raise AssertionError("unreachable: slim cannot support M4")


@_SKIP_REAL
def test_public_slim_drives_a_forward_not_a_grad_oracle():
    """Load the slim into this GPT and run a short forward.

    Tokens are synthetic. There is no published reference window or PyTorch
    grad tensor in the slim, so this is not numerical parity — only: the
    public weights are the right shape and produce finite logits.
    """
    slim = load_slim_zip(_SLIM)
    model = gpt_from_slim(slim)
    assert sum(int(p.data.size) for p in model.parameters()) == PUBLISHED_PARAM_COUNT
    vocab = PUBLISHED_CONFIG["vocab_size"]
    ref = np.load(LOGITS_FIXTURE)
    logits = model(ref["input_x"][None, :])
    assert logits.shape == (1, 16, vocab)
    assert np.isfinite(logits.data).all()
    err = _relative_error(logits.data[0], ref["ref_logits"])
    print(f"[public slim live] logits rel. error vs fixture PyTorch: {err:.6e}")
    assert err < LOGITS_RTOL
    assert err < LOGITS_CANARY
    assert np.array_equal(logits.data[0].argmax(-1), ref["ref_logits"].argmax(-1))
    # Explicit: we do not backward against a published oracle. The slim
    # has no g::* tensors. Autograd on these weights would be this
    # engine talking to itself, not a proof.
    assert "ref_logits" not in slim
    assert not any(k.startswith("g::") for k in slim)
    assert not any(k.startswith("g::") for k in slim["model"])
