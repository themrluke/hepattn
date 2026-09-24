# Memory snapshot

These two files are a copy of the previous session's persistent memory directory
(`~/.claude/projects/<project>/memory/`), which lives outside the repo and does not travel with
it. They are included so the next session can pick up the same background.

**They are a snapshot, and parts are out of date.** Where they disagree with the other documents
in `docs/handoff/`, those documents win — they were written at the end of the work, the memory
files during it.

Known stale points in `route-b-per-head-ordering.md`:

- **Teaching mode.** It says steps 4–5 are being done in teaching mode and that the
  implementation must not be written for Luke. That was true when written; he later asked for
  code to be written directly. Follow what he asks for now.
- **"Next step is 4.8".** All steps, including step 5 (padding), are finished, tested and
  committed.
- **The design fork is resolved.** The "third option" it flags as open — sorting each head's
  q/k/v into its own ordering with one shared banded window — was implemented, measured to be
  about 6.8x cheaper per event than Route B, and is now the **default** (`or_impl: "sorted"`).
  Route B survives as `or_impl: "masked"`, the reference implementation.
- **Note paths.** It refers to `src/hepattn/experiments/colliderml/notes/`; the directory is now
  `notes_for_obsidian/`.

If you want to re-create the memory on a new machine, the useful long-lived facts are: the
commit conventions in `05-environment-and-conventions.md`, the branch layout in `HANDOFF.md`, and
the design summary in `02-or-amplification.md`.
