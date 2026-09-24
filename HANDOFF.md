# Handoff — read this first

This branch (`handoff/context`) exists to move the whole state of this project from one
machine to another. It carries no code changes of its own: it is branched from
`OR-amplification`, plus a `docs/handoff/` directory.

The Obsidian notes and the presentation live on a **separate** branch,
`notes-and-presentation`, so that neither they nor the 2.7 MB `.pptx` ever land on a working
branch. See [Getting at the notes](#getting-at-the-notes) below.

**If you are a new Claude Code session: read `docs/handoff/` in order before doing anything.**
Start with `01-orientation.md`, which says where the work stands and what comes next.

| Document | What it covers |
|---|---|
| [`docs/handoff/01-orientation.md`](docs/handoff/01-orientation.md) | Project, branches, where the work stands, what comes next |
| [`docs/handoff/02-or-amplification.md`](docs/handoff/02-or-amplification.md) | The OR-amplification work: design, code, results, open questions |
| [`docs/handoff/03-colliderml-baseline.md`](docs/handoff/03-colliderml-baseline.md) | Current focus: getting the ColliderML baseline training well, starting at pu0 |
| [`docs/handoff/04-datasets.md`](docs/handoff/04-datasets.md) | How to download ColliderML and TrackML onto the new machine |
| [`docs/handoff/05-environment-and-conventions.md`](docs/handoff/05-environment-and-conventions.md) | Environment, Comet, tests, and how Luke wants work done |
| [`docs/handoff/memory/`](docs/handoff/memory/) | Copy of the previous session's persistent memory files |

## The short version

Two strands of work:

1. **OR amplification (done, not yet judged).** Per-head LSH ordering with OR amplification is
   implemented, tested and committed on `OR-amplification`, which is pushed. Two implementations
   exist (`sorted`, the default, and `masked`); they produce identical output. `sorted` costs
   about 5.6x a plain sliding window per layer, and about 1.44x per training step. Whether that
   cost buys any physics is **still unanswered**, because there is no working baseline to
   compare against.

2. **ColliderML baseline (current focus, in progress).** ColliderML training metrics were poor,
   and the likely cause is the event selection rather than the model: the config used pixel
   barrel only with |eta| < 1, which leaves about 4.2 hits per particle. The plan is to get the
   baseline working on a simpler no-pile-up (pu0) sample with a wider selection, then return to
   pu200, and only then run the OR comparison. Branch: `colliderml-baseline`.

## Branches

| Branch | State |
|---|---|
| `main` | Upstream-tracking; in sync with `origin/main`. |
| `luke_dev` | Older work; in sync with `origin/luke_dev`. |
| `OR-amplification` | All OR-amplification work. Pushed, working tree clean. |
| `colliderml-baseline` | Branched from `main`. The `pileup` dataset argument, `configs/pu0.yaml`, and a `.gitignore` fix for test plots. |
| `handoff/context` | This branch: handoff docs only. Branched from `OR-amplification`. |
| `notes-and-presentation` | This branch plus the Obsidian notes and the presentation. Branched from `handoff/context`. |

## Getting at the notes

The notes are deliberately on their own branch, which means that **checking out any other branch
removes them from the working tree** — they are committed, not lost, but they disappear from
disk. To keep them available alongside the code, use a second worktree rather than switching
back and forth:

```bash
git worktree add ../hepattn-notes notes-and-presentation
```

That leaves the notes permanently visible in `../hepattn-notes/src/hepattn/experiments/colliderml/notes_for_obsidian/`
while you work on any branch in the main checkout. Point Obsidian at that directory, or copy the
notes into your vault.

The presentation's `.venv/` (python-pptx, pypdf) and `__pycache__` were **not** committed;
recreate the virtualenv if you need to rebuild the deck.

## First thing to do on the new machine

The data does not travel with the repo. Follow
[`docs/handoff/04-datasets.md`](docs/handoff/04-datasets.md) to download ColliderML and TrackML
to their new locations, then update the `train_dir` / `val_dir` / `test_dir` paths in the
configs, which still point at the old cluster.
