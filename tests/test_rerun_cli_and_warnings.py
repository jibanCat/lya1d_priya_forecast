from priya_forecast.rerun import RerunConfig, cli_command_for, budget_warnings


def test_cli_command_sobolev_arm_has_flags():
    cmd = cli_command_for(RerunConfig.full(), "ns", 3.6, "sobolev")
    assert "scripts/refit_one_param_single_z.py" in cmd
    assert "--param ns" in cmd and "--z 3.6" in cmd
    assert "--target-space log" in cmd and "--use-sobolev" in cmd
    assert "--sobolev-lambda 5" in cmd
    assert "--niterations 200" in cmd and "--populations 48" in cmd
    assert "--kmin 0.001" in cmd and "--kmax 0.04" in cmd


def test_cli_command_value_arm_no_sobolev():
    cmd = cli_command_for(RerunConfig.full(), "tau0", 2.6, "value")
    assert "--use-sobolev" not in cmd
    assert "--target-space log" in cmd            # value baseline is log-target, plain MSE


def test_cli_command_notes_overrides_need_api():
    c = RerunConfig.quick(fiducial_overrides={"ns": 0.9})
    cmd = cli_command_for(c, "ns", 3.6, "value")
    assert "override" in cmd.lower()              # a comment noting CLI can't take overrides


def test_budget_warnings_flags_quick():
    w = budget_warnings(RerunConfig.quick())
    assert any("niter" in s.lower() for s in w)
    assert any("illustrative" in s.lower() or "production" in s.lower() for s in w)


def test_budget_warnings_silent_on_full():
    # full at production budget, all params, all z, both arms -> no budget shortfall
    w = budget_warnings(RerunConfig.full())
    assert w == []


# ---- command line: the same run_grid, usable from a cluster job --------------------

def test_cli_dry_run_full_preset_is_the_production_budget(capsys, tmp_path):
    from priya_forecast import rerun
    rc = rerun.main(["--preset", "full", "--label", "full_seed0",
                     "--out-root", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "budget: niter=200 pop=48 maxsize=20" in out
    assert "zs: [2.6, 3.6, 4.2]" in out and "sobolev_lambda: 5.0" in out
    assert "matches the production grid and budget" in out
    assert not (tmp_path / "rerun_full_seed0").exists()        # a dry run writes nothing


def test_cli_runs_the_grid_with_the_requested_settings(monkeypatch, tmp_path, capsys):
    from priya_forecast import rerun
    seen = {}

    def fake_run_grid(cfg, **kw):
        seen["cfg"] = cfg
        cfg.run_dir.mkdir(parents=True)
        return cfg.run_dir

    monkeypatch.setattr(rerun, "run_grid", fake_run_grid)
    rc = rerun.main(["--preset", "full", "--label", "s1", "--seed", "1", "--zs", "3.6",
                     "--params", "ns", "Ap", "--out-root", str(tmp_path)])
    cfg = seen["cfg"]
    assert rc == 0
    assert (cfg.label, cfg.seed, cfg.zs, cfg.params) == ("s1", 1, [3.6], ["ns", "Ap"])
    assert (cfg.niterations, cfg.populations, cfg.maxsize) == (200, 48, 20)
    assert (tmp_path / "rerun_s1" / "compare_to_production.csv").exists()


def test_cli_rejects_a_bad_override_before_running(tmp_path):
    import pytest
    from priya_forecast import rerun
    with pytest.raises(ValueError, match="unknown param"):
        rerun.main(["--params", "nope", "--out-root", str(tmp_path), "--dry-run"])
