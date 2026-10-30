"""Per-k export of `scripts/regen_multid.py` (the multid_bestworst_perk.csv curves).

The summary CSV (`tab:multid`) and the per-k CSV (the rel-err-vs-k figure) are
reductions of the same Sobol-sample error array, so the per-k curves must
reproduce the summary's mean / max / max_k exactly. These tests pin that.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def _load_regen_multid():
    """Load scripts/regen_multid.py by path (it's a script, not a package)."""
    scr = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "regen_multid.py"
    spec = importlib.util.spec_from_file_location("regen_multid", scr)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rm = _load_regen_multid()


def _rows_and_summaries(rels: dict[str, np.ndarray], k_grid: np.ndarray):
    """Mimic main(): one summary row per combo, best/worst tagged by mean."""
    rows, summaries = [], {}
    for params, rel in rels.items():
        s = rm._summarize(rel, k_grid)
        summaries[params] = s
        rows.append({"dim": f"{params.count('-') + 1}D", "params": params,
                     "mean_pct": s["mean_pct"], "role": ""})
    for dim in {r["dim"] for r in rows}:
        sub = sorted((r for r in rows if r["dim"] == dim), key=lambda r: r["mean_pct"])
        sub[0]["role"], sub[-1]["role"] = "best", "worst"
    return rows, summaries


# ---- unit tests ----------------------------------------------------------

def test_summarize_exposes_per_k_max():
    k = np.geomspace(0.001, 0.04, 5)
    rel = np.array([[0.01, 0.02, 0.03, 0.04, 0.05],
                    [0.03, 0.01, 0.06, 0.02, 0.01]])
    s = rm._summarize(rel, k)
    np.testing.assert_allclose(s["per_k_max_pct"], [3.0, 2.0, 6.0, 4.0, 5.0])
    assert s["max_pct"] == 6.0 and s["max_k"] == k[2]


def test_perk_csv_layout_and_header(tmp_path):
    k = np.geomspace(0.001, 0.04, 6)
    rng = np.random.default_rng(0)
    rels = {p: rng.uniform(0, 0.05, (16, k.size))
            for p in ("a-b", "c-d", "a-b-c", "b-c-d")}
    rows, summaries = _rows_and_summaries(rels, k)
    out = tmp_path / "perk.csv"
    rm._write_perk_csv(out, rows, summaries, k, z=3.6, n_sobol=16, seed=7)

    header = out.read_text().splitlines()[0]
    assert header.startswith("# git=")
    assert "source=multid_combine_perk" in header
    assert "z=3.6 n_sobol=16 seed=7" in header

    df = pd.read_csv(out, comment="#", float_precision="round_trip")
    assert list(df.columns) == rm.PERK_COLS
    assert len(df) == len(rels) * k.size
    for params, g in df.groupby("params", sort=False):
        np.testing.assert_array_equal(g["k"].to_numpy(), k)  # full-precision k
        assert g["role"].nunique(dropna=False) == 1
    roles = df.drop_duplicates("params").set_index("params")["role"].fillna("")
    assert sorted(roles[roles != ""]) == ["best", "best", "worst", "worst"]


def test_both_writers_emit_lf_only(tmp_path):
    """The committed CSVs are LF; a CRLF writer makes every regenerated row
    differ byte-wise from them even when the values agree."""
    k = np.geomspace(0.001, 0.04, 4)
    rel = np.random.default_rng(1).uniform(0, 0.05, (8, k.size))
    rows, summaries = _rows_and_summaries({"a-b": rel, "c-d": rel / 2}, k)
    for r in rows:
        s = summaries[r["params"]]
        r.update(n_varied=2, p90_pct=s["p90_pct"], max_pct=s["max_pct"],
                 max_k=s["max_k"], gp_sliced="")
    rm._write_csv(tmp_path / "summary.csv", rows)
    rm._write_perk_csv(tmp_path / "perk.csv", rows, summaries, k,
                       z=3.6, n_sobol=8, seed=0)
    for name in ("summary.csv", "perk.csv"):
        assert b"\r" not in (tmp_path / name).read_bytes(), name


# ---- hypothesis property tests -------------------------------------------

_rel = arrays(
    np.float64,
    shape=st.tuples(st.integers(1, 40), st.integers(2, 12)),
    elements=st.floats(0.0, 0.5, allow_nan=False, allow_subnormal=False),
)


@settings(max_examples=30, deadline=None)
@given(rel=_rel)
def test_per_k_curves_reduce_to_summary_scalars(rel):
    """mean = k-average of per-k mean; max/max_k = max/argmax of per-k max."""
    k = np.geomspace(0.001, 0.04, rel.shape[1])
    s = rm._summarize(rel, k)
    assert np.isclose(s["per_k_mean_pct"].mean(), s["mean_pct"], rtol=1e-12, atol=1e-12)
    assert s["per_k_max_pct"].max() == s["max_pct"]
    assert k[int(np.argmax(s["per_k_max_pct"]))] == s["max_k"]
    lo, hi = 100.0 * rel.min(axis=0), s["per_k_max_pct"]
    assert np.all(s["per_k_p90_pct"] >= lo - 1e-12)
    assert np.all(s["per_k_p90_pct"] <= hi + 1e-12)
    assert np.all(s["per_k_mean_pct"] <= hi + 1e-12)


@settings(max_examples=15, deadline=None)
@given(rel_a=_rel, seed=st.integers(0, 2**16))
def test_perk_csv_roundtrips_curves(tmp_path_factory, rel_a, seed):
    """Reading the CSV back recovers every curve to the written 1e-6 precision."""
    k = np.geomspace(0.001, 0.04, rel_a.shape[1])
    rel_b = np.random.default_rng(seed).uniform(0, 0.1, rel_a.shape)
    rows, summaries = _rows_and_summaries({"a-b": rel_a, "c-d": rel_b}, k)
    out = tmp_path_factory.mktemp("perk") / "perk.csv"
    rm._write_perk_csv(out, rows, summaries, k, z=3.6, n_sobol=rel_a.shape[0], seed=seed)
    df = pd.read_csv(out, comment="#")
    for params, g in df.groupby("params"):
        s = summaries[params]
        for col in ("mean_pct", "p90_pct", "max_pct"):
            np.testing.assert_allclose(g[col].to_numpy(), s[f"per_k_{col}"],
                                       rtol=0, atol=5e-7)


# ---- the committed artifact ----------------------------------------------

_MULTID = (pathlib.Path(__file__).resolve().parents[1] / "results"
           / "paper_production_20260630_perz_sobolev_z2.6-4.2" / "figures" / "multid_z3.6")


def test_committed_perk_curves_reduce_to_committed_summary():
    """The committed per-k curves and the committed `tab:multid` summary come
    from the same Sobol sample: mean / max / max_k / role agree for all ten
    combos (the summary is rounded to 4 decimals, max_k to 5)."""
    perk_csv = _MULTID / "multid_bestworst_perk.csv"
    header = perk_csv.read_text().splitlines()[0]
    assert header.startswith("# git=")
    assert "source=multid_combine_perk z=3.6 n_sobol=256 seed=1234" in header

    perk = pd.read_csv(perk_csv, comment="#", float_precision="round_trip")
    summ = pd.read_csv(_MULTID / "multid_bestworst.csv", comment="#")
    assert len(summ) == 10
    assert list(perk["params"].drop_duplicates()) == list(summ["params"])

    k_default = np.geomspace(0.001, 0.04, 48)  # regen_multid's default k-grid
    for _, row in summ.iterrows():
        g = perk[perk["params"] == row["params"]]
        np.testing.assert_allclose(g["k"].to_numpy(), k_default, rtol=1e-12, atol=0)
        assert g["dim"].unique().tolist() == [row["dim"]]
        role = row["role"] if isinstance(row["role"], str) else ""
        assert g["role"].fillna("").unique().tolist() == [role]
        assert abs(g["mean_pct"].mean() - row["mean_pct"]) <= 5e-5 + 1e-6
        assert abs(g["max_pct"].max() - row["max_pct"]) <= 5e-5 + 1e-6
        k_at_max = g["k"].to_numpy()[int(np.argmax(g["max_pct"].to_numpy()))]
        assert round(k_at_max, 5) == row["max_k"]
        assert np.all(g["p90_pct"] <= g["max_pct"]) and np.all(g["p90_pct"] >= 0)
