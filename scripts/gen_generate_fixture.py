"""Congela duas gerações GREEDY do PersonaCore para a paridade do M1.

PRECISA do venv do PersonaCore (é o único com torch). Ver scripts/README.md.

Mesma origem de dados da fixture de paridade: o checkpoint best.pt e o começo de
data/val.bin. O prompt são os 10 PRIMEIROS tokens de val.bin — ids crus, sem
tokenizer, sem texto novo.

Duas gerações, ambas greedy (determinísticas, sem RNG), chamando
personacore.generation.core.generate DIRETAMENTE (não generate_text: sem
forbid_ids, sem prefixo de eos_id, sem decode):

  (a) curta: max_new_tokens=20  -> total ~30 tokens, nunca cruza block_size=256,
      então o crop `idx[:, -bs:]` nunca dispara.
  (b) longa: max_new_tokens=280 -> total ~290 tokens, cruza block_size e força o
      crop a disparar repetidamente. É esta que pega um bug de posição absoluta
      vs relativa, que a curta é estruturalmente incapaz de ver.

A sequência é capturada exatamente como `generate` faz yield: se o EOS for
atingido, ele NÃO aparece (a função dá return ANTES do yield e do append). Logo
`len(gerados) < max_new_tokens` é a assinatura de uma parada por EOS, e o passo
em que parou é o próprio `len(gerados)` (0-indexado).
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

PROMPT_LEN = 10
SHORT_NEW = 20
LONG_NEW = 280

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
print(f"block_size = {cfg.block_size}   eos_id = {cfg.eos_id}")


def run(max_new_tokens, label):
    t0 = time.perf_counter()
    toks = list(generate(model, idx, max_new_tokens=max_new_tokens, greedy=True))
    dt = time.perf_counter() - t0
    # generate() dá return ANTES de emitir o EOS, então uma lista curta == parou por EOS.
    eos_step = len(toks) if len(toks) < max_new_tokens else -1
    print(f"\n[{label}] max_new_tokens={max_new_tokens}  emitidos={len(toks)}  "
          f"{dt:.1f}s ({dt / max(len(toks), 1) * 1000:.0f} ms/token)")
    print(f"[{label}] eos_step = {eos_step}   ({'PAROU POR EOS' if eos_step >= 0 else 'sem EOS'})")
    print(f"[{label}] tokens = {toks}")
    return np.array(toks, dtype=np.int64), eos_step


short_ids, short_eos = run(SHORT_NEW, "curta")
long_ids, long_eos = run(LONG_NEW, "longa")

path = out("personacore_generate_fixture.npz")
np.savez(
    path,
    prompt_ids=prompt,
    short_generated_ids=short_ids,
    long_generated_ids=long_ids,
    short_max_new_tokens=np.int64(SHORT_NEW),
    long_max_new_tokens=np.int64(LONG_NEW),
    short_eos_step=np.int64(short_eos),
    long_eos_step=np.int64(long_eos),
    block_size=np.int64(cfg.block_size),
    eos_id=np.int64(cfg.eos_id),
)
report(path)
