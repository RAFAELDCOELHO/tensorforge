"""Demonstração fim a fim: treina o GPT do PersonaCore neste motor.

Ponto de partida são os pesos reais de checkpoints/best.pt (step 49000) e o
dado são janelas reais de data/val.bin. Nada de PyTorch em lugar nenhum daqui
para baixo — só NumPy.

Dois regimes, porque provam coisas diferentes:

  A. OVERFIT de um batch. O mesmo batch em todo passo, com lr alto. É o gate
     clássico (o próprio PersonaCore tem um, TRAIN-05): se a loss não despenca
     aqui, alguma peça entre forward, loss, backward e optimizer está morta.
     É o que prova que o motor APRENDE.

  B. CONTINUAÇÃO fiel. Janelas novas a cada passo, lr = lr_at_step(49000),
     exatamente o schedule que a corrida real teria usado no passo seguinte.
     Aqui a loss NÃO deve despencar: o modelo já convergiu neste corpus depois
     de 49k passos. É o que prova que o motor RODA no regime real.

Não é para reproduzir a trajetória histórica — isso está descartado por
princípio (fp32/MPS lá, float64/NumPy aqui; ~1e-7 no primeiro passo, e o treino
amplifica isso exponencialmente).

    python scripts/demo_train.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from core.optim import AdamW                       # noqa: E402
from core.train import train_step                  # noqa: E402
from tests.test_parity import load_personacore     # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures")
REAL = dict(max_norm=1.0, base_lr=3e-4, warmup_steps=100,
            max_steps=50000, min_ratio=0.1)
CKPT_STEP = 49000


def load():
    parity = np.load(os.path.join(FIXTURES, "personacore_parity.npz"))
    windows = np.load(os.path.join(FIXTURES, "val_windows.npz"))["windows"]
    return load_personacore(parity), windows


def as_batch(windows):
    """(B, 257) -> x (B, 256), y (B*256,) — o mesmo shift do get_batch."""
    return windows[:, :256], windows[:, 1:].reshape(-1)


def run(label, model, batches, steps, hp, step_index):
    print(f"\n=== {label}")
    print(f"    lr base = {hp['base_lr']:.2e}   max_norm = {hp['max_norm']}")
    opt = AdamW(model.parameters(), weight_decay=0.1)
    losses = []
    t0 = time.time()
    for s in range(steps):
        x, y = as_batch(batches[s % len(batches)])
        loss, gnorm = train_step(model, opt, x, y, step_index(s), **hp)
        losses.append(loss)
        flag = "clip" if gnorm > hp["max_norm"] else "    "
        print(f"    passo {s:2d}  loss {loss:8.5f}   |g| {gnorm:8.4f} {flag}")
    dt = time.time() - t0
    print(f"    {steps} passos em {dt:.1f}s ({dt / steps:.2f}s/passo)")
    return losses


def main():
    model, windows = load()
    n_params = sum(p.data.size for p in model.parameters())
    print(f"modelo: {n_params:,} parâmetros carregados de best.pt (step {CKPT_STEP})")
    print(f"dado:   {len(windows)} janelas reais de val.bin, 256 tokens cada")

    # --- A: overfit de um batch ---
    one = [windows[:4]]
    losses = run("A. OVERFIT — mesmo batch de 4 janelas, lr alto",
                 model, one, 15, {**REAL, "base_lr": 1e-3, "warmup_steps": 1},
                 step_index=lambda s: s)
    drop = (losses[0] - losses[-1]) / losses[0]
    print(f"    loss {losses[0]:.5f} -> {losses[-1]:.5f}  ({drop:+.1%})")
    print(f"    monotônica em {sum(b < a for a, b in zip(losses, losses[1:]))}/"
          f"{len(losses) - 1} passos")

    # --- B: continuação fiel ---
    model, _ = load()                              # pesos frescos do checkpoint
    losses = run("B. CONTINUAÇÃO — janelas novas a cada passo, lr real do step 49000",
                 model, [windows[i:i + 4] for i in range(0, 16, 4)], 12, REAL,
                 step_index=lambda s: CKPT_STEP + s)
    print(f"    loss média {np.mean(losses):.5f}  (desvio {np.std(losses):.5f})")
    print(f"    faixa [{min(losses):.5f}, {max(losses):.5f}] — batches diferentes, "
          "não é uma curva de aprendizado")


if __name__ == "__main__":
    main()
