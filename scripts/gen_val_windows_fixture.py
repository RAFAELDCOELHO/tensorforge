"""Extrai 16 janelas reais de PersonaCore/data/val.bin para a demonstração.

NÃO precisa de torch: val.bin é um memmap uint16 puro, o NumPy lê direto. Roda
com o Python deste repositório.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fixture_paths import out, personacore_root, report  # noqa: E402

import numpy as np  # noqa: E402

d = np.memmap(os.path.join(personacore_root(), "data", "val.bin"),
              dtype=np.uint16, mode="r")
# espaçadas pelo corpus (não contíguas) para variar o conteúdo
starts = np.linspace(0, len(d) - 258, 16).astype(int)
windows = np.stack([np.array(d[s:s + 257], dtype=np.int64) for s in starts])

path = out("val_windows.npz")
np.savez(path, windows=windows, starts=starts)
print("janelas:", windows.shape, "| tokens únicos:", len(np.unique(windows)))
report(path)
