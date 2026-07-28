"""Congela pesos + entrada real + logits/loss/grads de referência do PersonaCore.

PRECISA do venv do PersonaCore (é o único com torch). Ver scripts/README.md.

Tudo em float64 explícito: o checkpoint é float32, então o cast é sem perda, e
a partir daí as duas implementações fazem a MESMA matemática em float64 — as
diferenças que sobrarem são ordem de operação, não precisão de entrada.

Consumido por tests/test_parity.py, e é pré-requisito dos outros dois
geradores (ambos reusam os gradientes daqui).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fixture_paths import out, personacore_root, report, require_torch  # noqa: E402

require_torch()
ROOT = personacore_root()
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from personacore.config import ModelConfig  # noqa: E402
from personacore.model.gpt import GPT  # noqa: E402

T = 256

ck = torch.load(os.path.join(ROOT, "checkpoints", "best.pt"),
                map_location="cpu", weights_only=False)
cfg = ModelConfig(**ck["model_config"])
model = GPT(cfg, attn_impl="manual")      # manual = o caminho que o tensorforge espelha
model.load_state_dict(ck["model"])
model.double().eval()                     # float64, sem dropout (dropout=0.0 de qualquer jeito)

data = np.memmap(os.path.join(ROOT, "data", "val.bin"), dtype=np.uint16, mode="r")
window = np.array(data[: T + 1], dtype=np.int64)
x = torch.tensor(window[:T]).unsqueeze(0)          # (1, T)
y = torch.tensor(window[1:]).unsqueeze(0)

logits, loss = model(x, y)
loss.backward()

blob = {f"w::{k}": v.detach().numpy().astype(np.float32)
        for k, v in model.state_dict().items()}
blob["input_x"] = window[:T]
blob["input_y"] = window[1:]
blob["ref_logits"] = logits.detach().numpy()[0]    # (T, V) float64
blob["ref_loss"] = np.float64(loss.item())

named = dict(model.named_parameters())
for k in ["blocks.0.attn.q_proj.weight", "blocks.0.ln_1.bias",
          "blocks.5.mlp.fc_out.weight", "ln_f.weight"]:
    blob[f"g::{k}"] = named[k].grad.numpy()

# wte é o parâmetro tied — o grad soma as duas rotas. Guarda linhas escolhidas:
# presentes na janela (as duas rotas) e ausentes (só a rota lm_head).
present = np.unique(window[:T])[:8]
absent = np.array([i for i in range(cfg.vocab_size) if i not in set(window.tolist())][:4])
rows = np.concatenate([present, absent])
blob["g::wte.rows"] = rows
blob["g::wte.weight"] = named["wte.weight"].grad.numpy()[rows]
blob["g::wte.full_abs_sum"] = np.float64(np.abs(named["wte.weight"].grad.numpy()).sum())

path = out("personacore_parity.npz")
np.savez(path, **blob)
print("loss = %.17g" % loss.item())
print("linhas wte amostradas:", rows.tolist())
report(path)
