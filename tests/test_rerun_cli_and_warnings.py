from priya_forecast.rerun import RerunConfig, cli_command_for, budget_warnings


def test_cli_command_sobolev_arm_has_flags():
    cmd = cli_command_for(RerunConfig.full(), "ns", 3.6, "sobolev")
    assert "scripts/refit_one_param_single_z.py" in cmd
    assert "--param ns" in cmd and "--z 3.6" in cmd
    assert "--target-space log" in cmd and "--use-sobolev" in cmd
    assert "--sobolev-lambda 5" in cmd
    assert "--niterations 200" in cmd and "--populations 48" in cmd


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
