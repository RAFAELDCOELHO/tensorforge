"""Congela uma geração GREEDY que REALMENTE PARA POR EOS.

PRECISA do venv do PersonaCore (é o único com torch). Ver scripts/README.md.

Por que esta fixture existe: `personacore_generate_fixture.npz` tem duas gerações
que pararam por `max_new_tokens`, nenhuma por EOS. Logo o caminho de parada por
EOS do motor NumPy nunca foi comparado contra o PyTorch em nenhum milestone — só
contra stub sintético. Esta fixture fecha esse buraco.

O `eos_id` real do checkpoint (8184) não serve: ele nunca é o argmax numa janela
curta, então nenhuma geração de 30 passos pararia. Usamos um EOS ARTIFICIAL: um
token comum que o modelo de fato emite cedo. `generate` trata `eos_id` como
parâmetro puro (só a comparação de parada), então trocá-lo não muda logit nenhum.

Token escolhido: 261.
  - Scan dos 30 primeiros tokens gerados pelo mesmo prompt de 10 tokens (sem EOS,
    já congelados em personacore_generate_fixture.npz): 261 aparece pela primeira
    vez no passo 16 e 14 vezes no run de 280.
  - Alta frequência (o critério da spec) E parada no meio do loop. O token 46 —
    o candidato inicial — pararia no passo 4, emitindo só 4 tokens: teste magro
    demais, mal exercita o loop.

Como greedy é determinístico e o EOS só INTERROMPE (nunca altera um logit), o
passo de parada é exatamente o índice da primeira ocorrência de 261 na geração
sem EOS. Este script não assume isso: mede e falha se não parar.

Contrato de captura (o mesmo do M1): `generate` dá `return` ANTES do `yield` e do
append, então o EOS NÃO aparece em `generated_ids`, e `len(generated_ids)` É o
passo em que parou (0-indexado).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fixture_paths import out, personacore_root, report, require_torch  # noqa: E402

require_torch()
ROOT = personacore_root()
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from personacore.config import ModelConfig  # noqa: E402
from personacore.generation.core import generate  # noqa: E402
from personacore.model.gpt import GPT  # noqa: E402

PROMPT_LEN = 10          # mesmo prompt do M1: os 10 primeiros tokens de val.bin
MAX_NEW = 30
ARTIFICIAL_EOS = 261

ck = torch.load(os.path.join(ROOT, "checkpoints", "best.pt"),
                map_location="cpu", weights_only=False)
cfg = ModelConfig(**ck["model_config"])
model = GPT(cfg, attn_impl="manual")      # manual = o caminho que o tensorforge espelha
model.load_state_dict(ck["model"])
model.double().eval()                     # float64, sem dropout (dropout=0.0 de qualquer jeito)

data = np.memmap(os.path.join(ROOT, "data", "val.bin"), dtype=np.uint16, mode="r")
prompt = np.array(data[:PROMPT_LEN], dtype=np.int64)
idx = torch.tensor(prompt).unsqueeze(0)   # (1, PROMPT_LEN)

print(f"prompt_ids ({PROMPT_LEN}) = {prompt.tolist()}")
print(f"block_size = {cfg.block_size}   eos_id real = {cfg.eos_id}   "
      f"eos_id artificial = {ARTIFICIAL_EOS}")

t0 = time.perf_counter()
toks = list(generate(model, idx, max_new_tokens=MAX_NEW,
                     greedy=True, eos_id=ARTIFICIAL_EOS))
dt = time.perf_counter() - t0

eos_step = len(toks) if len(toks) < MAX_NEW else -1
print(f"\nemitidos = {len(toks)}  em {dt:.1f}s")
print(f"eos_step = {eos_step}   ({'PAROU POR EOS' if eos_step >= 0 else 'NÃO parou por EOS'})")
print(f"tokens = {toks}")

if eos_step < 0:
    sys.exit(f"token {ARTIFICIAL_EOS} não foi emitido em {MAX_NEW} passos — "
             "escolha outro token e rode de novo. Fixture NÃO escrita.")
if ARTIFICIAL_EOS in toks:
    sys.exit(f"token {ARTIFICIAL_EOS} apareceu na saída — o contrato "
             "'return antes do yield' foi quebrado. Fixture NÃO escrita.")

path = out("personacore_eos_fixture.npz")
np.savez(
    path,
    prompt_ids=prompt,
    generated_ids=np.array(toks, dtype=np.int64),
    artificial_eos_id=np.int64(ARTIFICIAL_EOS),
    eos_step=np.int64(eos_step),
    block_size=np.int64(cfg.block_size),
    max_new_tokens=np.int64(MAX_NEW),
)
report(path)
