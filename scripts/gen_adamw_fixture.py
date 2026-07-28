"""Congela UM passo de torch.optim.AdamW a partir do estado real do best.pt.

PRECISA do venv do PersonaCore. Rode gen_parity_fixture.py antes — os
gradientes vêm de lá.

Estado (exp_avg/exp_avg_sq/step) do checkpoint no passo 49000; gradientes já
congelados e validados na fixture de paridade. Tudo em float64 dos dois lados,
então a comparação mede ordem de operação, não precisão.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fixture_paths import FIXTURES, out, personacore_root, report, require_torch  # noqa: E402

require_torch()
ROOT = personacore_root()
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from personacore.config import ModelConfig  # noqa: E402
from personacore.model.gpt import GPT  # noqa: E402

LR = 3.0267460081032337e-05          # o valor gravado no param_group em 49000
BETAS, EPS, WD = (0.9, 0.999), 1e-8, 0.1
KEYS = ["blocks.0.attn.q_proj.weight",   # 2-D, fundo no backward
        "blocks.0.ln_1.bias",            # 1-D — o caso do decay sem filtro de ndim
        "blocks.5.mlp.fc_out.weight",    # 2-D, raso
        "ln_f.weight"]                   # 1-D, ganho de LayerNorm

ck = torch.load(os.path.join(ROOT, "checkpoints", "best.pt"),
                map_location="cpu", weights_only=False)
model = GPT(ModelConfig(**ck["model_config"]), attn_impl="manual")
model.load_state_dict(ck["model"])

# índice no param_groups -> nome, pela ORDEM de model.parameters() (o contrato do torch)
names = [n for n, _ in model.named_parameters()]
assert len(names) == len(ck["optimizer"]["param_groups"][0]["params"]) == 100

parity_path = os.path.join(FIXTURES, "personacore_parity.npz")
if not os.path.exists(parity_path):
    sys.exit("rode scripts/gen_parity_fixture.py primeiro")
parity = np.load(parity_path)

state = ck["optimizer"]["state"]
blob = {}
for key in KEYS:
    st = state[names.index(key)]
    p0 = ck["model"][key].double().clone()
    g = torch.tensor(parity[f"g::{key}"], dtype=torch.float64)
    assert p0.shape == g.shape
    m0, v0 = st["exp_avg"].double().clone(), st["exp_avg_sq"].double().clone()
    t0 = int(st["step"].item())

    p = torch.nn.Parameter(p0.clone())
    p.grad = g.clone()
    opt = torch.optim.AdamW([p], lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    opt.state[p] = {"step": torch.tensor(float(t0)),
                    "exp_avg": m0.clone(), "exp_avg_sq": v0.clone()}
    opt.step()

    blob[f"p0::{key}"], blob[f"g::{key}"] = p0.numpy(), g.numpy()
    blob[f"m0::{key}"], blob[f"v0::{key}"] = m0.numpy(), v0.numpy()
    blob[f"t0::{key}"] = np.int64(t0)
    blob[f"p1::{key}"] = p.detach().numpy()
    blob[f"m1::{key}"] = opt.state[p]["exp_avg"].numpy()
    blob[f"v1::{key}"] = opt.state[p]["exp_avg_sq"].numpy()
    print(f"{key:35s} ndim={p0.ndim} t0={t0} |dp|max={(p.detach() - p0).abs().max():.3e}")

# controle: o MESMO passo com decay ACOPLADO (Adam clássico + wd*p no gradiente)
key = KEYS[0]
st = state[names.index(key)]
p = torch.nn.Parameter(ck["model"][key].double().clone())
p.grad = torch.tensor(parity[f"g::{key}"], dtype=torch.float64) + WD * p.detach()
opt = torch.optim.Adam([p], lr=LR, betas=BETAS, eps=EPS, weight_decay=0.0)
opt.state[p] = {"step": torch.tensor(float(st["step"].item())),
                "exp_avg": st["exp_avg"].double().clone(),
                "exp_avg_sq": st["exp_avg_sq"].double().clone()}
opt.step()
blob[f"coupled_p1::{key}"] = p.detach().numpy()

blob["lr"], blob["eps"], blob["wd"] = np.float64(LR), np.float64(EPS), np.float64(WD)
blob["betas"] = np.array(BETAS)

path = out("personacore_adamw.npz")
np.savez(path, **blob)
report(path)
