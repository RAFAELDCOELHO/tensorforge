"""Caminhos compartilhados pelos geradores de fixture.

O repo do PersonaCore é resolvido nesta ordem: $PERSONACORE_ROOT, depois o
irmão ../PersonaCore ao lado deste repositório.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TENSORFORGE = os.path.dirname(HERE)
FIXTURES = os.path.join(TENSORFORGE, "fixtures")


def personacore_root():
    root = os.environ.get("PERSONACORE_ROOT") or os.path.join(
        os.path.dirname(TENSORFORGE), "PersonaCore")
    if not os.path.isdir(os.path.join(root, "src", "personacore")):
        sys.exit(f"PersonaCore não encontrado em {root!r}. "
                 "Defina PERSONACORE_ROOT para o repo.")
    return root


def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("Este script precisa do venv do PersonaCore (com torch). Rode:\n"
                 "  $PERSONACORE_ROOT/.venv/bin/python scripts/<este-script>.py")


def out(name):
    os.makedirs(FIXTURES, exist_ok=True)
    return os.path.join(FIXTURES, name)


def report(path):
    print(f"escrito: {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
