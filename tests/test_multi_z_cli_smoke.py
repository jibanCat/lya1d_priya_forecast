# tests/test_multi_z_cli_smoke.py
import subprocess, sys, os


def _run(script, *args):
    env = dict(os.environ); env["PYTHONPATH"] = "src"
    return subprocess.run([sys.executable, script, *args],
                          capture_output=True, text=True, env=env)


def test_run_pipeline_multi_z_help():
    out = _run("scripts/run_pipeline_multi_z.py", "--help")
    assert out.returncode == 0
    assert "--config" in out.stdout


def test_refit_one_param_multi_z_help():
    out = _run("scripts/refit_one_param_multi_z.py", "--help")
    assert out.returncode == 0
    assert "--z-min" in out.stdout and "--z-max" in out.stdout
