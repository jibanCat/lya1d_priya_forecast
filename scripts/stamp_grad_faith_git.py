#!/usr/bin/env python
"""One-off back-fill: add a ``git=<hash>`` field to grad_faith sidecar headers that
predate the provenance convention (see ``priya_forecast.provenance``). Idempotent —
skips sidecars that already carry a ``git=`` stamp. Going forward,
``grad_faith_io.write_grad_faith_sidecar`` stamps automatically, so this script is
only for pre-convention artifacts.

Usage:
    scripts/stamp_grad_faith_git.py [--dir RESULTS_DIR] [--git HASH] [--dry-run]

The default ``--git`` is the git the committed production run documents in its
README.md / RUN_MANIFEST.md; pass ``--git`` explicitly for other runs.
"""
import argparse
from pathlib import Path

DEFAULT_DIR = "results/paper_production_20260630_perz_sobolev_z2.6-4.2"
DEFAULT_GIT = "7aa26af"  # this run's code git (its README.md; RUN_MANIFEST records the earlier submit git fc914fc+uncommitted)


def stamp_header(line: str, git: str) -> str | None:
    """Return the stamped header line, or None if it already has git= / isn't a header."""
    if not line.startswith("#"):
        return None
    if "git=" in line:
        return None  # already stamped
    nl = "\n" if line.endswith("\n") else ""
    body = line.rstrip("\n")
    if "source=" in body:
        return body.replace("source=", f"git={git} source=", 1) + nl
    return body + f" git={git}" + nl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--git", default=DEFAULT_GIT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.dir).rglob("grad_faith_*.csv"))
    stamped = skipped = 0
    for f in files:
        with open(f) as fh:
            lines = fh.readlines()
        if not lines:
            skipped += 1
            continue
        new_first = stamp_header(lines[0], args.git)
        if new_first is None:
            skipped += 1
            continue
        lines[0] = new_first
        if not args.dry_run:
            with open(f, "w") as fh:
                fh.writelines(lines)
        stamped += 1
    verb = "would stamp" if args.dry_run else "stamped"
    print(f"{verb} {stamped} sidecar(s) git={args.git}; skipped {skipped} "
          f"(already stamped / headerless) of {len(files)} under {args.dir}")


if __name__ == "__main__":
    main()
