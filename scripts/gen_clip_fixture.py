"""Congela a saída de torch.nn.utils.clip_grad_norm_ em gradientes reais.

PRECISA do venv do PersonaCore. Rode gen_parity_fixture.py antes.

Três regimes de max_norm (corta / no-op / corte agressivo) mais o caso de
fronteira construído, em que a norma é EXATAMENTE igual a max_norm — onde o
eps de 1e-6 no denominador faz clipar mesmo assim.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fixture_paths import FIXTURES, out, report, require_torch  # noqa: E402

require_torch()

import numpy as np  # noqa: E402
import torch  # noqa: E402

KEYS = ["blocks.0.attn.q_proj.weight", "blocks.0.ln_1.bias",
        "blocks.5.mlp.fc_out.weight", "ln_f.weight"]

parity_path = os.path.join(FIXTURES, "personacore_parity.npz")
if not os.path.exists(parity_path):
    sys.exit("rode scripts/gen_parity_fixture.py primeiro")
parity = np.load(parity_path)
grads = [torch.tensor(parity[f"g::{k}"], dtype=torch.float64) for k in KEYS]

nat = torch.linalg.vector_norm(
    torch.stack([torch.linalg.vector_norm(g, 2.0) for g in grads]), 2.0)
print("norma total dos 4 grads reais: %.17g" % nat.item())

blob = {f"g::{k}": parity[f"g::{k}"] for k in KEYS}
blob["keys"] = np.array(KEYS)
for tag, mx in [("clips", 0.4), ("noop", 10.0), ("tight", 1e-3)]:
    ps = [torch.nn.Parameter(g.clone()) for g in grads]
    for p, g in zip(ps, grads):
        p.grad = g.clone()
    tn = torch.nn.utils.clip_grad_norm_(ps, mx)
    blob[f"maxnorm::{tag}"] = np.float64(mx)
    blob[f"totalnorm::{tag}"] = np.float64(tn.item())
    for k, p in zip(KEYS, ps):
        blob[f"clipped::{tag}::{k}"] = p.grad.numpy()
    ch = max(float((p.grad - g).abs().max()) for p, g in zip(ps, grads))
    print(f"  max_norm={mx:<8} coef={min(1.0, mx / (tn.item() + 1e-6)):.10f}  mudou={ch:.3e}")

# fronteira construída: [3,4] tem norma 5 exata contra max_norm=5.0
b = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
b.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
tnb = torch.nn.utils.clip_grad_norm_([b], 5.0)
blob["boundary_grad_in"] = np.array([3.0, 4.0])
blob["boundary_grad_out"] = b.grad.numpy()
blob["boundary_total"] = np.float64(tnb.item())
print("fronteira: norma", tnb.item(), "-> grad", b.grad.numpy().tolist())

path = out("personacore_clip.npz")
np.savez(path, **blob)
report(path)
