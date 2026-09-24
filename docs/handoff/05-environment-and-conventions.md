# 05 — Environment, tooling and how Luke wants work done

## Working conventions (important — these are standing instructions)

- **Commits must carry no trace of Claude.** No `Co-Authored-By: Claude`, no "Generated with
  Claude Code", nothing in the message or the authorship. Commits are authored
  `themrluke <themrlukejohnson@gmail.com>`. This overrides any default attribution guidance the
  session is given. Verify with `git log --format='%an <%ae>|%s%n%b' | grep -i claude` before
  pushing — it should print nothing.
- **Do not commit the Obsidian notes to working branches.** `notes_for_obsidian/` is Luke's
  personal working directory and is gitignored on `OR-amplification`. The one exception is the
  `notes-and-presentation` branch, where they were committed deliberately, and force-added past
  the ignore rule, to move them between machines. They must stay off `main`,
  `OR-amplification` and `colliderml-baseline`.
- **Commit messages** are lowercase, plain, and describe the change, e.g. "handle padded hits in
  the per-head orderings", "pick the compiled or uncompiled flex kernel per call, from the
  tensor device".
- **Working style.** Luke often works in a teaching mode: he writes the code and asks for the
  steps, explanations and line-level guidance, and will say so ("walk me through", "explain
  simply"). At other times he asks for the code to be written directly. Follow whichever he
  asks for in the moment rather than assuming. He checks the work carefully and pushes back when
  an explanation is hand-waved, so state what was measured versus what is inferred.
- **Do not change the presentation's content** — layout fixes only, nothing deleted or reworded,
  unless he asks.

## Environment

- **pixi** manages the environment: `pixi shell` from the repo root, or the apptainer image
  `pixi.sif` (`apptainer shell --nv --bind <data dir> pixi.sif`). Python 3.12, PyTorch with
  flex attention.
- The old cluster used Slurm (`srun --pty --cpus-per-task 15 --gres gpu:l40s:1 --mem=100G -p GPU
  bash`) and had A100s, some 40 GB and some 80 GB. Pick a GPU explicitly with
  `CUDA_VISIBLE_DEVICES=N`, after checking `nvidia-smi` — the machine is shared.
- Training entry points: `src/hepattn/experiments/colliderml/main.py`, and for TrackML
  `run_tracking.py` / `run_filtering.py`. All use the Lightning CLI:
  `python main.py fit --config configs/pu0.yaml`.

## Comet

- Workspace `themrluke`, project `colliderml`. Configured through `~/.comet.config`, which on
  the old machine held the API key with `chmod 600`.
- **The API key was pasted into a chat session and should be rotated.** It was flagged at the
  time; it is not known whether that was done. Rotate it on the new machine and set it via
  `COMET_API_KEY` or `~/.comet.config` — do not paste it into a prompt. To hand a credential to
  a session, use the `!` prefix so the command runs locally and only its output enters the
  conversation.
- The project CLI injects Comet-specific arguments into whatever logger is configured, so
  `--trainer.logger false` and CSVLogger overrides fail on the command line. For a logger-free
  smoke test, copy the config and set `logger: false` while removing the `LearningRateMonitor`
  callback.

## Tests

```bash
CUDA_VISIBLE_DEVICES=2 pytest tests/models/ tests/flex/
```

Expect **2 failed, 455 passed, 69 skipped**. The two failures are
`tests/flex/test_flex_basic.py` and `tests/flex/test_nested.py` running out of memory, and they
are pre-existing: they only appear when `tests/models/` and `tests/flex/` share one process on a
40 GB card, `tests/flex/` alone passes 48/48, and both reproduce on unmodified `main`. Do not
attribute them to new work, and do not "fix" them by changing the OR code.

The OR tests worth knowing: `tests/models/test_encoder_or.py` (identity invariance, head-major
layout, `masked` vs `sorted` agreement, padding, cache staleness across events),
`tests/models/test_maskformer_or_coords.py`, `tests/models/test_or_merge.py`,
`tests/models/test_encoder_set_backend.py`.

## Repo and branches

- `origin` is `git@github.com:themrluke/hepattn.git`; `upstream` is
  `git@github.com:samvanstroud/hepattn.git`.
- Branches: `main`, `luke_dev`, `OR-amplification`, `colliderml-baseline`, `handoff/context`,
  `notes-and-presentation`. See `HANDOFF.md` for what each holds.
- `tests/outputs/` accumulates diagnostic plots written by the test suite. It is ignored on
  `OR-amplification` and, since the `.gitignore` fix on `colliderml-baseline`, there too —
  along with `*.jpg` / `*.jpeg`, which the old rules missed (only `*.png` was covered). If a
  branch off `main` shows `tests/outputs/` as untracked, it is missing that fix.

## Notes

`src/hepattn/experiments/colliderml/notes_for_obsidian/` (committed on the
`notes-and-presentation` branch only — use `git worktree add ../hepattn-notes
notes-and-presentation` to keep it visible while working on another branch) is Luke's Obsidian
vault content for this project:

| File | Contents |
|---|---|
| `1. Per-Head Ordering.md` | The main design note for the OR work |
| `step4-5-plan.md` | The implementation plan, step by step, with all measurements |
| `test_encoder_or.md`, `test_maskformer_or_coords.md` | Standalone explanations of the test files |
| `ColliderML vs TrackML.md` | Dataset comparison, and the original download recipe |
| `presentation/` | The 33-slide deck, its figure scripts and the speaker script |

They use `[[wiki links]]` between notes. When updating them, keep the cross-links working and do
not drop sections — a section was once deleted by a careless edit and had to be restored.
