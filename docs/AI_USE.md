# Use of AI tools: the full account (recorded 2026-11-24)

The README of this repository has a short version of this text. Paths are relative to the
repository root, and "README line N" refers to `README.md` as of this commit.

AI coding agents worked on this repository: see the session record and the Cursor commits below.
Each statement names the record that supports it. Claude Code drafted this text on 2026-11-24
from the records it cites.

`AUTHORSHIP.md` means `docs/AUTHORSHIP.md` on the branch `docs/authorship` of the PersonaCore
repository (commit `86cda17`). Its §4.4 reconstructs this repository's one surviving session.

- **What the git history does not show.**
  - The history was rewritten on 2026-07-28 to remove the private fixtures (README section "Reproduction limits"; `AUTHORSHIP.md` lines 265–268).
  - The first commit, `372dbb8`, already holds the whole v1.0 engine (`AUTHORSHIP.md` line 266), so
    no commit shows how the engine was developed.
  - 4 of the 13 commits are by a Cursor agent (below). The other 9 carry the author's identity: 6 on
    2026-07-28 and 3 GitHub-side commits on 2026-09-01 (`git log --format='%an %ad'`). That identity
    does not say whether a person or a model wrote a line.
  - Who produced the first version of `core/engine.py`, the scalar engine, is not recoverable from
    any artifact (`AUTHORSHIP.md` §9, item 3, lines 480–486).
- **What the surviving session record shows.** One Claude Code (Anthropic) session from 2026-07-28
  survives as a private copy on the author's machine and is not published. `AUTHORSHIP.md` §4.4
  reconstructs it (lines 269–270 and 278–293).
  - In it, the agent wrote `core/tensor.py`, `core/nn.py`, `core/optim.py` and `core/train.py`,
    their tests, `tests/test_parity.py`, `scripts/demo_train.py` and the first version of the README
    (Write calls in the session's private copy,
    `rollout-2026-08-01T11-19-16-019fbdb1-86da-7cd3-a1ee-863ebc200a0a.jsonl`).
  - The author required that `train_step` match the manual composition of its pieces byte for byte.
  - The agent saw that test fail, diagnosed the `set`-ordering bug, changed `_prev` to a tuple in
    `core/tensor.py`, wrote a determinism regression test and ran a mutation check (`AUTHORSHIP.md`
    line 282).
  - The author then ordered the same fix in `core/engine.py` and asked for the finding to be
    recorded in the README (`AUTHORSHIP.md` line 283). He accepted that attribution (§9, item 4).
  - README lines 13 and 55 first appear as the agent's output in that session (private session
    record). They do not mean that a person typed the code or ran the mutants: in that session the
    agent wrote the modules and ran the mutation check (`AUTHORSHIP.md` line 282).
- **Cursor commits.** The Cursor agent's 4 commits on 2026-09-01 (`git log --author="Cursor Agent"`)
  added:
  - the MIT license (`12cc94c`);
  - the public-slim parity test and `scripts/slim_zip.py` (`a3c5823`);
  - the public-slim fixture, its generator `scripts/gen_slim_logits_fixture.py` and the CI workflow
    (`3f6db55`);
  - the citation file (`31cae8a`).

  They also wrote the README section "What the public slim allows" (`git blame README.md`).
- **What does not depend on who wrote a line.**
  - The parity tests compare this engine with a checkpoint it did not produce (README line 60), and
    the test writes out how its tolerances were derived (`tests/test_parity.py:79-92`).
  - The determinism fix is pinned by a test that demands bit-identical weights over repeated runs
    (`tests/test_train.py:353`).
- **Authorship.** No AI system is listed as an author (`CITATION.cff:5-7`).
