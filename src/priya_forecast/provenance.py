"""Provenance helpers — git stamping of generated artifacts.

**Convention (all artifact writers should follow this):** every generated data
artifact carries the short git hash of the code that produced it, embedded in the
file itself (a ``# ... git=<hash> ...`` comment header for CSVs, a ``"git"`` field
for JSON), so a committed data file is traceable to a code state without relying
only on a run-level README. Call :func:`git_stamp` in the writer and include its
value in the header/payload. Readers that use ``pd.read_csv(comment="#")`` or that
read named JSON keys are unaffected by the added stamp.

Artifact writers wired to this convention (all emit a ``# git=<hash> source=...``
header, or a JSON ``git`` field):

* ``grad_faith_io.write_grad_faith_sidecar`` — grad_faith sidecar CSVs
* ``refit_1d_pysr`` pareto output — the PySR Pareto-front CSVs
* ``scripts/regen_maxsize_sensitivity.py`` / ``scripts/regen_multid.py`` — the figure-data CSVs
* ``scripts/aggregate_seed_band.py`` — ``seed_band_summary.json`` (``git`` field)

Readers already tolerate the header: ``pd.read_csv(comment="#")`` in
``grad_faith_io``, ``pareto_diag.load_front``, ``models.pysr_model.load_pareto_csv``,
and ``paper_figures`` (maxsize/multid); JSON reads named keys. The one-off
``scripts/stamp_grad_faith_git.py`` back-fills pre-convention grad_faith sidecars; the
committed ``paper_production_20260630…`` run's grad_faith + maxsize + multid CSVs and
JSON were back-filled to ``git=7aa26af``. The large pareto fileset is stamped **going
forward**; its pre-convention committed files stay traceable via their sibling
grad_faith sidecar (whose ``source=`` names the pareto and whose ``git=`` dates it).
"""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

# src/priya_forecast/provenance.py -> repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def git_stamp() -> str:
    """Short git hash of the working tree, with a ``+dirty`` suffix when the tree
    has uncommitted changes.

    Returns ``"nogit"`` if git or the repo is unavailable (e.g. an installed
    wheel or a source tarball rather than a clone), so writers never fail on
    provenance. Cached per process (the head does not move mid-run).
    """
    try:
        rev = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if rev.returncode != 0 or not rev.stdout.strip():
            return "nogit"
        h = rev.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            h += "+dirty"
        return h
    except Exception:
        return "nogit"
