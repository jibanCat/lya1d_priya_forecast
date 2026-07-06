from pathlib import Path
import pandas as pd
import pytest
from priya_forecast.rerun import RerunConfig, run_grid


def _fake_gp_loader(basedir, k_grid):
    return ("gp_lf", "gp_hf")


def _fake_regen_fn(gp_lf, gp_hf, params, zs, k_grid, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)   # scorer is faked; contents unused


def _make_refit_fn(record, fail_param=None):
    def refit_fn(*, param_name, z, cfg, gp_lf, gp_hf, k_grid, out_dir):
        record.append((param_name, z, str(out_dir)))
        if param_name == fail_param:
            raise RuntimeError(f"boom {param_name}")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Complexity": [5], "Loss": [0.1], "Equation": ["x0"]}
                     ).to_csv(Path(out_dir) / f"pareto_{param_name}.csv", index=False)
        return object()
    return refit_fn


def _fake_score_fn(pareto_csv, param, z, out_csv, basedir, data_1pvar,
                   fid_ov=None, prior_ov=None):
    pd.DataFrame({"Complexity": [5], "Loss": [0.1], "grad_err": [0.2],
                  "value_mse": [1e-4], "n_keep": [40], "gate_pass": [True],
                  "x0_enters": [True]}).to_csv(out_csv, index=False)


def test_run_grid_writes_production_layout(tmp_path):
    rec = []
    cfg = RerunConfig.quick(params=["ns", "tau0"])
    cfg.out_root = tmp_path
    run_dir = run_grid(cfg, gp_loader=_fake_gp_loader, regen_fn=_fake_regen_fn,
                       refit_fn=_make_refit_fn(rec), score_fn=_fake_score_fn,
                       progress=lambda *_: None)
    assert run_dir == tmp_path / "rerun_quick"
    for arm in ("value", "sobolev"):
        d = run_dir / arm / "refit" / "z3.6"
        assert (d / "pareto_ns.csv").exists()
        assert (d / "grad_faith_ns.csv").exists()
        assert (d / "pareto_tau0.csv").exists()
    assert (run_dir / "RUN_MANIFEST.md").exists()
    assert len(rec) == 4                               # 2 params x 2 arms x 1 z


def test_run_grid_refuses_production_dir(tmp_path):
    cfg = RerunConfig.quick()
    cfg.out_root = Path("results/paper_production_20260630_perz_sobolev_z2.6-4.2")
    with pytest.raises(ValueError, match="production"):
        run_grid(cfg, gp_loader=_fake_gp_loader, regen_fn=_fake_regen_fn,
                 refit_fn=_make_refit_fn([]), score_fn=_fake_score_fn)


def test_run_grid_applies_overrides(tmp_path):
    seen = {}
    from priya_forecast import parameters as P

    def refit_fn(*, param_name, z, cfg, gp_lf, gp_hf, k_grid, out_dir):
        seen["ns_fid"] = P.get_param("ns").fid          # read under the context
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Complexity": [5], "Loss": [0.1], "Equation": ["x0"]}
                     ).to_csv(Path(out_dir) / f"pareto_{param_name}.csv", index=False)
        return object()

    cfg = RerunConfig.quick(params=["ns"], arms=["value"],
                            fiducial_overrides={"ns": 0.90})
    cfg.out_root = tmp_path
    run_grid(cfg, gp_loader=_fake_gp_loader, regen_fn=_fake_regen_fn,
             refit_fn=refit_fn, score_fn=_fake_score_fn, progress=lambda *_: None)
    assert seen["ns_fid"] == 0.90
    assert P.get_param("ns").fid == 0.983               # restored after


def test_run_grid_continues_on_one_failed_fit(tmp_path):
    rec = []
    cfg = RerunConfig.quick(params=["ns", "tau0"], arms=["value"])
    cfg.out_root = tmp_path
    run_dir = run_grid(cfg, gp_loader=_fake_gp_loader, regen_fn=_fake_regen_fn,
                       refit_fn=_make_refit_fn(rec, fail_param="ns"),
                       score_fn=_fake_score_fn, progress=lambda *_: None)
    d = run_dir / "value" / "refit" / "z3.6"
    assert not (d / "pareto_ns.csv").exists()           # failed param -> no CSV
    assert (d / "pareto_tau0.csv").exists()             # other param still ran
    assert (run_dir / "RUN_MANIFEST.md").exists()       # manifest still written


def test_score_fn_passes_override_env(monkeypatch, tmp_path):
    # _real_score_fn must serialize overrides into the subprocess env
    import priya_forecast.rerun as R
    captured = {}

    def fake_run(cmd, check, env):
        captured["env"] = env
        captured["cmd"] = cmd
        class R0:  # noqa
            returncode = 0
        return R0()

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    R._real_score_fn(tmp_path / "p.csv", "ns", 3.6, tmp_path / "gf.csv",
                     "data/kodiaq_gp", tmp_path / "_1pvar",
                     fid_ov={"ns": 0.9}, prior_ov=None)
    assert '"ns": 0.9' in captured["env"]["PRIYA_FIDUCIAL_OVERRIDES"]
    assert "PRIYA_PRIOR_OVERRIDES" not in captured["env"]
    assert "--data-1pvar" in captured["cmd"]
