#!/usr/bin/env python
"""Aggregate the §5.1 joint multi-D maxsize sweep into the summary table.

Every column is derived from the per-seed ``joint_rank_diagnostic.json`` files, so
``SWEEP_SUMMARY.md`` can be regenerated rather than hand-maintained. An earlier
hand-typed version of that table carried a ``#singular`` column that no metric in
these files reproduces; the ``rank-deficient`` column below is the honest count.

Two ranks are reported because they answer different questions and were previously
conflated:

``front-max``
    max numerical rank over *every* Pareto equation. Biased upward: the front grows
    with maxsize, so a max over it can climb from extra draws alone.
``selected``
    rank of the loss-minimizing equation -- the one a forecast would actually deploy.

Usage::

    python scripts/aggregate_joint_sweep.py                    # markdown to stdout
    python scripts/aggregate_joint_sweep.py --write-summary    # rewrite SWEEP_SUMMARY.md table
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "results" / "joint_multid_sweep"
MAXSIZES = (25, 40, 60, 100)
SEEDS = range(5)


def load(maxsize: int, seed: int) -> dict:
    path = SWEEP_DIR / f"ms{maxsize}_seed{seed}" / "joint_rank_diagnostic.json"
    with path.open() as fh:
        return json.load(fh)


def _fmt_small(x: float) -> str:
    """Eigenvalues below ~1e-12 are machine zero; don't print false precision."""
    return "~0" if abs(x) < 1e-12 else f"{x:.2e}"


def rows() -> list[dict]:
    out = []
    for ms in MAXSIZES:
        runs = [load(ms, s) for s in SEEDS]
        front = [r["front_scan"]["front_max_rank_1e8"] for r in runs]
        sel = [r["preregistered"]["idxmin_rank_whitened_1e8"] for r in runs]
        eig = [r["joint_pysr"]["whitened"]["eigenvalues"] for r in runs]
        out.append({
            "maxsize": ms,
            "front": front,
            "selected": sel,
            "detached": sum(not r["preregistered"]["pinned_at_cap"] for r in runs),
            "all_inputs": sum(r["n_inputs_present"] == r["n_params"] for r in runs),
            "lam5": st.median(e[4] for e in eig),
            "lam6": st.median(e[5] for e in eig),
            "rank_deficient": sum(r["joint_pysr"]["rank_deficient_vs_nparams"] for r in runs),
            "offfid": st.median(r["accuracy_offfid"]["student_vs_gp"] for r in runs),
        })
    return out


def table(rs: list[dict]) -> str:
    head = (
        "| maxsize | front-max rank (med, range) | selected rank (med, range) | "
        "detached / all-6-inputs | median λ5 | median λ6 | rank-deficient | off-fid (med) |\n"
        "|--------:|:--------------------------:|:--------------------------:|"
        ":-----------------------:|:---------:|:---------:|:--------------:|:-------------:|\n"
    )
    body = ""
    for r in rs:
        f, s = r["front"], r["selected"]
        body += (
            f"| {r['maxsize']:<7} | {st.median(f):.0f} ({min(f)}-{max(f)}) "
            f"| {st.median(s):.0f} ({min(s)}-{max(s)}) "
            f"| {r['detached']}/5 · {r['all_inputs']}/5 "
            f"| {_fmt_small(r['lam5'])} | {_fmt_small(r['lam6'])} "
            f"| {r['rank_deficient']}/5 | {r['offfid']:.3f} |\n"
        )
    return head + body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-summary", action="store_true",
                    help="rewrite the table block inside SWEEP_SUMMARY.md in place")
    args = ap.parse_args()

    rs = rows()
    md = table(rs)
    o25, o100 = rs[0]["offfid"], rs[-1]["offfid"]

    if not args.write_summary:
        print(md)
        print(f"off-fid ratio ms25/ms100 = {o25 / o100:.2f}x")
        gp = load(100, 0)["gp_reference"]["whitened"]
        ev = gp["eigenvalues"]
        print(f"GP whitened: rank 6/6, cond {gp['condition_number']:.0f}, "
              f"sloppiest/stiffest = {ev[-1] / ev[0]:.2e}")
        return

    summary = SWEEP_DIR / "SWEEP_SUMMARY.md"
    text = summary.read_text()
    start = text.index("| maxsize |")
    end = text.index("\n\n", start)
    summary.write_text(text[:start] + md.rstrip("\n") + text[end:])
    print(f"rewrote table in {summary}")


if __name__ == "__main__":
    main()
