import pytest
from pydantic import ValidationError

from monitor_localization.config import ChunkConfig, ExperimentConfig, ModelConfig
from monitor_localization.paths import CONFIGS_DIR


def test_default_yaml_loads():
    cfg = ExperimentConfig.from_yaml("default")
    assert cfg.name == "baseline"
    assert cfg.subset.n_samples == 500
    assert cfg.chunk.overlap == 0
    assert cfg.aggregation.method == "max"


def test_shipped_configs_all_valid():
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        ExperimentConfig.from_yaml(path)


def test_default_monitor_is_gpt_4_1_mini():
    cfg = ModelConfig()
    assert cfg.provider == "openai"
    assert cfg.name == "gpt-4.1-mini"
    assert cfg.max_tokens == 256
    assert cfg.temperature == 0.0


def test_temperature_must_be_an_openai_supported_value():
    with pytest.raises(ValidationError):
        ModelConfig(temperature=2.1)


def test_valid_temperature_is_accepted():
    assert ModelConfig(temperature=1.0).temperature == 1.0


def test_overlap_must_be_less_than_size():
    with pytest.raises(ValidationError):
        ChunkConfig(size=5, overlap=5)


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        ChunkConfig(size=5, overlap=0, unknown_knob=1)


def test_roundtrip_to_dict_is_json_safe():
    import json

    json.dumps(ExperimentConfig.from_yaml("default").to_dict())
