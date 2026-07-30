from monitor_localization.config import ExperimentConfig
from monitor_localization.experiment import ExperimentRun
from monitor_localization.utils import read_json


def make_config(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(name="unit-test", output_dir=tmp_path / "results")


def test_run_creates_stage_directory(tmp_path):
    run = ExperimentRun(make_config(tmp_path), stage="global")
    assert run.root.is_dir()
    assert run.root.parts[-2:] == ("unit-test", "global")


def test_manifest_records_config_and_timestamps(tmp_path):
    run = ExperimentRun(make_config(tmp_path), stage="global")
    run.write_manifest(n_transcripts=3, n_failures=0)
    manifest = read_json(run.manifest_path)
    assert manifest["stage"] == "global"
    assert manifest["config"]["name"] == "unit-test"
    assert manifest["n_transcripts"] == 3
    assert manifest["started_at"] <= manifest["finished_at"]


def test_stages_do_not_collide(tmp_path):
    cfg = make_config(tmp_path)
    assert (
        ExperimentRun(cfg, stage="global").results_path
        != ExperimentRun(cfg, stage="chunk").results_path
    )
