"""Read a PersonaCore slim ``.pt`` (torch zip) without importing torch.

The public ``m1-demo-v1`` artifact is a torch zip of tensors + primitives only
(``weights_only=True``). This repo does not depend on torch, so the restricted
unpickler below reconstructs those tensors as NumPy arrays.

It is a loader, not an oracle: the slim has inference weights and
``model_config``, not published PyTorch grads or a reference window.
"""
from __future__ import annotations

import os
import pickle
import zipfile
from collections import OrderedDict

import numpy as np

SLIM_KEYS = frozenset(
    {"schema_version", "model", "model_config", "git_sha", "step", "val_loss"}
)
DROPPED_TRAINING_KEYS = frozenset(
    {"optimizer", "scheduler", "scaler", "rng", "train_config"}
)
# Architecture the public slim advertises, and that this engine's GPT matches.
PUBLISHED_CONFIG = {
    "vocab_size": 8192,
    "eos_id": 8184,
    "block_size": 256,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "dropout": 0.0,
}
PUBLISHED_PARAM_COUNT = 13_891_584
PUBLISHED_GIT_SHA_PREFIX = "3a46815"
PUBLISHED_STEP = 49000

_ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"): OrderedDict,
}


class _Storage:
    def __init__(self, data, dtype):
        self.data = data
        self.dtype = dtype


def _is_contiguous(size, stride):
    expected = 1
    for s, st in zip(reversed(size), reversed(stride)):
        if s == 0:
            return True
        if st != expected:
            return False
        expected *= s
    return True


def _rebuild_tensor_v2_safe(storage, storage_offset, size, stride, requires_grad,
                            backward_hooks, metadata=None):
    size = tuple(int(s) for s in size)
    stride = tuple(int(s) for s in stride)
    offset = int(storage_offset)
    n = 1
    for s in size:
        n *= s
    if n == 0:
        return np.zeros(size, dtype=storage.dtype)
    if _is_contiguous(size, stride):
        return storage.data[offset:offset + n].reshape(size).copy()
    last = offset + sum((s - 1) * st for s, st in zip(size, stride))
    flat = storage.data[offset:last + 1]
    return np.lib.stride_tricks.as_strided(
        flat, shape=size,
        strides=tuple(st * flat.itemsize for st in stride),
        writeable=False,
    ).copy()


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, file, storages):
        super().__init__(file)
        self._storages = storages

    def find_class(self, module, name):
        if (module, name) in _ALLOWED_GLOBALS:
            return _ALLOWED_GLOBALS[(module, name)]
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2_safe
        if module == "torch" and name in {"FloatStorage", "HalfStorage",
                                          "DoubleStorage", "BFloat16Storage"}:
            return name
        raise pickle.UnpicklingError(f"refused global {module}.{name}")

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or pid[0] != "storage":
            raise pickle.UnpicklingError(f"refused persistent id {pid!r}")
        _kind, storage_type, storage_id, _location, numel = pid[:5]
        dtype = {
            "FloatStorage": np.float32,
            "HalfStorage": np.float16,
            "DoubleStorage": np.float64,
            "BFloat16Storage": np.dtype("bfloat16") if hasattr(np, "bfloat16")
            else np.float32,
        }.get(storage_type if isinstance(storage_type, str) else storage_type)
        if dtype is None:
            raise pickle.UnpicklingError(f"unsupported storage {storage_type!r}")
        raw = self._storages[str(storage_id)]
        expected = int(numel) * np.dtype(dtype).itemsize
        if len(raw) < expected:
            raise pickle.UnpicklingError(
                f"storage {storage_id} is {len(raw)} bytes, expected {expected}"
            )
        data = np.frombuffer(raw[:expected], dtype=dtype)
        return _Storage(data, dtype)


def _zip_prefix(names):
    for name in names:
        if name.endswith("data.pkl"):
            return name[:-len("data.pkl")]
    raise ValueError("not a torch zip: no data.pkl")


def _check_slim_keys(loaded):
    if not isinstance(loaded, dict):
        raise ValueError(f"slim root is {type(loaded).__name__}, not dict")
    keys = set(loaded)
    missing = SLIM_KEYS - keys
    extra = keys - SLIM_KEYS
    if missing or extra:
        raise ValueError(
            f"slim key set {sorted(keys)} is not the published inference "
            f"schema {sorted(SLIM_KEYS)} (missing={sorted(missing)}, "
            f"extra={sorted(extra)})"
        )
    leaked = DROPPED_TRAINING_KEYS & keys
    if leaked:
        raise ValueError(f"training state leaked into slim: {sorted(leaked)}")


def load_slim_zip(path):
    """Load a slim ``.pt`` into a dict of primitives + NumPy arrays.

    Fails closed if the top-level key set is not the published slim schema.
    """
    with zipfile.ZipFile(path) as z:
        prefix = _zip_prefix(z.namelist())
        storages = {}
        data_prefix = prefix + "data/"
        for name in z.namelist():
            if name.startswith(data_prefix) and not name.endswith("/"):
                storages[name[len(data_prefix):]] = z.read(name)
        with z.open(prefix + "data.pkl") as fh:
            loaded = _RestrictedUnpickler(fh, storages).load()
    _check_slim_keys(loaded)
    return loaded


def write_schema_only_zip(path, payload):
    """Write a zip whose ``data.pkl`` is a plain pickle (no torch tensors).

    Used by tests as a synthetic fixture: CI never downloads the 55 MB slim.
    """
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("synthetic/data.pkl", pickle.dumps(payload, protocol=2))


def slim_search_paths():
    """Local-only locations. Never downloaded by tests or CI."""
    env = os.environ.get("PERSONACORE_SLIM")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    paths = []
    if env:
        paths.append(env)
    paths.extend([
        os.path.join(root, "checkpoints", "model_slim.pt"),
        os.path.join(os.path.dirname(root), "PersonaCore",
                     "checkpoints", "model_slim.pt"),
    ])
    return paths


def find_slim():
    for path in slim_search_paths():
        if path and os.path.isfile(path):
            return path
    return None


def assert_published_config(cfg):
    """Fail closed if the slim's embedded config is not this engine's GPT."""
    for key, expected in PUBLISHED_CONFIG.items():
        got = cfg.get(key)
        if got != expected:
            raise AssertionError(
                f"slim model_config[{key!r}]={got!r} != published {expected!r}"
            )


def gpt_from_slim(slim):
    """Build this engine's GPT and overwrite every weight from the slim.

    Linear weights are transposed: torch stores ``(out, in)``, this engine
    stores ``(in, out)``. ``lm_head.weight`` is not loaded as a second table
    — tying is the engine's only head, and the slim values are asserted equal
    to ``wte.weight``.
    """
    # Imported here so metadata-only uses of this module stay engine-free.
    from core.nn import GPT

    assert_published_config(slim["model_config"])
    weights = slim["model"]
    if not np.array_equal(weights["lm_head.weight"], weights["wte.weight"]):
        raise AssertionError(
            "slim lm_head.weight is not equal to wte.weight — tying broken"
        )

    cfg = slim["model_config"]
    model = GPT(cfg["vocab_size"], cfg["n_embd"], cfg["n_head"],
                cfg["n_layer"], cfg["block_size"])

    def put(tensor, value):
        value = np.asarray(value, dtype=np.float64)
        if tensor.shape != value.shape:
            raise AssertionError(f"{tensor.shape} != {value.shape}")
        tensor.data = value

    put(model.wte.weight, weights["wte.weight"])
    put(model.wpe.weight, weights["wpe.weight"])
    put(model.ln_f.gamma, weights["ln_f.weight"])
    put(model.ln_f.beta, weights["ln_f.bias"])
    for i, block in enumerate(model.blocks):
        p = f"blocks.{i}."
        put(block.ln_1.gamma, weights[p + "ln_1.weight"])
        put(block.ln_1.beta, weights[p + "ln_1.bias"])
        put(block.ln_2.gamma, weights[p + "ln_2.weight"])
        put(block.ln_2.beta, weights[p + "ln_2.bias"])
        for name in ("q_proj", "k_proj", "v_proj", "c_proj"):
            proj = getattr(block.attn, name)
            put(proj.weight, weights[f"{p}attn.{name}.weight"].T)
            put(proj.bias, weights[f"{p}attn.{name}.bias"])
        put(block.mlp.fc_in.weight, weights[p + "mlp.fc_in.weight"].T)
        put(block.mlp.fc_in.bias, weights[p + "mlp.fc_in.bias"])
        put(block.mlp.fc_out.weight, weights[p + "mlp.fc_out.weight"].T)
        put(block.mlp.fc_out.bias, weights[p + "mlp.fc_out.bias"])
    return model
