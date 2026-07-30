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


def test_temperature_unset_by_default():
    # Current Claude models reject `temperature`; it must not be sent implicitly.
    assert ModelConfig().temperature is None


def test_temperature_with_adaptive_thinking_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(temperature=0.0, thinking="adaptive")


def test_temperature_allowed_when_thinking_disabled():
    assert ModelConfig(temperature=0.0, thinking="disabled").temperature == 0.0


def test_overlap_must_be_less_than_size():
    with pytest.raises(ValidationError):
        ChunkConfig(size=5, overlap=5)


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        ChunkConfig(size=5, overlap=0, unknown_knob=1)


def test_roundtrip_to_dict_is_json_safe():
    import json

    json.dumps(ExperimentConfig.from_yaml("default").to_dict())
