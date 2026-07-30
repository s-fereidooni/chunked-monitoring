import os

from monitor_localization.utils import load_env, read_json, read_jsonl, write_json, write_jsonl


def test_load_env_parses_and_skips_comments(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "HF_TOKEN=hf_abc123",
                'QUOTED="value with spaces"',
                "EMPTY=",
                "not_a_pair",
            ]
        )
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    applied = load_env(env)
    assert applied["HF_TOKEN"] == "hf_abc123"
    assert applied["QUOTED"] == "value with spaces"
    assert "EMPTY" not in applied
    assert os.environ["HF_TOKEN"] == "hf_abc123"


def test_load_env_does_not_clobber_exported_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=from_file")
    monkeypatch.setenv("HF_TOKEN", "from_shell")
    # An explicitly exported value must win over a possibly stale .env.
    assert load_env(env) == {}
    assert os.environ["HF_TOKEN"] == "from_shell"


def test_load_env_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=from_file")
    monkeypatch.setenv("HF_TOKEN", "from_shell")
    load_env(env, override=True)
    assert os.environ["HF_TOKEN"] == "from_file"


def test_load_env_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") == {}


def test_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "a.json"
    write_json(path, {"run_ids": [1, 2, 3]})
    assert read_json(path)["run_ids"] == [1, 2, 3]


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "a.jsonl"
    write_jsonl(path, [{"i": 0}, {"i": 1}])
    assert [r["i"] for r in read_jsonl(path)] == [0, 1]


def test_write_json_leaves_no_temp_files(tmp_path):
    write_json(tmp_path / "a.json", {"x": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
