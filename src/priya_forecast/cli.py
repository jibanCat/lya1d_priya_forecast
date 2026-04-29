"""Unified `priya-forecast` CLI.

Subcommands:

  priya-forecast forecast   — score a PySR equation YAML vs GP + perfect_1D
  priya-forecast multid     — train real multi-D PySR + score
  priya-forecast coupling   — produce the headline coupling-matrix heatmap
  priya-forecast hpo        — sweep PySR hyperparameters
  priya-forecast diagnose   — full reward-loop bundle (forecast + multid + coupling)

Each subcommand re-exports the corresponding `scripts/*.py` driver so the
student can either run `priya-forecast <sub> ...` (the entry point in
pyproject.toml) OR call the script directly.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Auto-set Julia paths so PySR works without the user remembering them.
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(Path.home() / ".julia_env"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(Path.home() / ".julia"))


def _delegate(script_name: str, argv: list[str]) -> int:
    """Run a `scripts/<script_name>.py` driver with `argv` as its sys.argv."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    script_path = repo_root / "scripts" / f"{script_name}.py"
    if not script_path.exists():
        raise SystemExit(f"Driver not found: {script_path}")
    # Import the module under a non-__main__ name and then call its main()
    # explicitly so the driver's `if __name__ == "__main__"` guard is bypassed.
    spec = importlib.util.spec_from_file_location(f"_drv_{script_name}", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [str(script_path), *argv]
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        mod.main()
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="priya-forecast",
        description="PRIYA P1D forecast + PySR diagnostics + reusable PySR HPO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  priya-forecast forecast --equations published --params dtau0 Ap ns alphaq --output results/r1\n"
            "  priya-forecast coupling --params ns Ap hub omegamh2 herei alphaq --output results/r2\n"
            "  priya-forecast multid   --params dtau0 Ap --niter 200 --output results/r3\n"
            "  priya-forecast hpo --param ns --space configs/hpo/quick.yaml --n-trials 6 --output results/r4\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("forecast", help="Score a PySR equation YAML vs GP + perfect_1D references.",
                   add_help=False)
    sub.add_parser("multid",   help="Train multi-D real PySR + score the equation.",
                   add_help=False)
    sub.add_parser("coupling", help="Sweep all parameter pairs; produce coupling-matrix heatmap.",
                   add_help=False)
    sub.add_parser("hpo",      help="Sweep PySR hyperparameters on one parameter.",
                   add_help=False)
    sub.add_parser("diagnose", help="Run forecast + multid + coupling end-to-end.",
                   add_help=False)

    args, rest = parser.parse_known_args()
    name_map = {
        "forecast": "train_and_forecast",
        "multid":   "run_multid_pysr",
        "coupling": "run_coupling_matrix",
        "hpo":      "run_pysr_hpo",
    }
    if args.cmd in name_map:
        return _delegate(name_map[args.cmd], rest)
    if args.cmd == "diagnose":
        # Run forecast → coupling → hpo on a sensible default subset.
        # The student can pass --output and --params; everything else uses
        # quick.yaml-style budgets.
        raise SystemExit(
            "`diagnose` subcommand chains forecast + coupling + hpo. "
            "Run them individually for now — combining them needs explicit "
            "budget choices that vary by use case."
        )
    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
