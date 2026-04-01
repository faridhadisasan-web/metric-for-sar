from typer.testing import CliRunner
from rriq.cli import app

runner = CliRunner()


def test_init_project():
    result = runner.invoke(app, ["init-project"])
    assert result.exit_code == 0
    assert "initialized" in result.stdout


def test_run_all_dry_run():
    # Will use the default configs from the package
    result = runner.invoke(app, ["run-all", "--config", "configs/default.yaml"])
    assert result.exit_code == 0
    assert "Running full pipeline" in result.stdout
