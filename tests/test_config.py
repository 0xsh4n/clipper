"""Configuration loading, layering and validation."""

from __future__ import annotations

import pytest

from app.config import load_config, parse_formats, parse_resolution
from app.errors import ConfigError


def test_defaults():
    cfg = load_config()
    assert cfg.ollama.model == "qwen2.5"
    assert cfg.ollama.base_url == "http://localhost:11434"
    assert cfg.whisper.model == "small"
    assert cfg.clip.target_duration == 30.0
    assert cfg.video.formats == ["16:9", "9:16"]


def test_yaml_then_env_then_cli_precedence(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "ollama:\n  model: from-yaml\nclip:\n  target_duration: 28\n", encoding="utf-8"
    )
    env_file = tmp_path / ".env"
    env_file.write_text('OLLAMA_MODEL="from-env"\nWHISPER_MODEL=base\n', encoding="utf-8")

    cfg = load_config(config_path=config_file, env_path=env_file, environ={})
    assert cfg.ollama.model == "from-env"      # env beats yaml
    assert cfg.whisper.model == "base"
    assert cfg.clip.target_duration == 28.0    # yaml still applies

    cfg = load_config(
        config_path=config_file,
        env_path=env_file,
        overrides={"ollama.model": "from-cli"},
        environ={},
    )
    assert cfg.ollama.model == "from-cli"      # cli beats env


def test_real_environment_beats_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_MODEL=from-env-file\n", encoding="utf-8")
    cfg = load_config(env_path=env_file, environ={"OLLAMA_MODEL": "from-process"})
    assert cfg.ollama.model == "from-process"


def test_env_type_coercion():
    cfg = load_config(
        environ={
            "KEEP_TEMP_FILES": "true",
            "TARGET_CLIP_DURATION": "20",
            "MIN_CLIP_DURATION": "18",
            "MAX_CLIP_DURATION": "22",
            "SUBTITLE_FONT_SIZE": "60",
            "VIDEO_FORMATS": "9:16",
        }
    )
    assert cfg.runtime.keep_temp_files is True
    assert cfg.clip.target_duration == 20.0
    assert cfg.subtitles.font_size == 60
    assert cfg.video.formats == ["9:16"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"clip.min_duration": 40.0},                     # min > target
        {"clip.max_duration": 10.0},                     # max < min
        {"clip.clips": 0},
        {"whisper.model": "enormous"},
        {"whisper.device": "tpu"},
        {"subtitles.position": "diagonal"},
        {"ollama.base_url": "localhost:11434"},          # missing scheme
        {"video.formats": "4:3"},
    ],
)
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ConfigError):
        load_config(overrides=overrides, environ={})


def test_parse_formats_aliases():
    assert parse_formats("both") == ["16:9", "9:16"]
    assert parse_formats("shorts") == ["9:16"]
    assert parse_formats("16:9,9:16,16:9") == ["16:9", "9:16"]
    assert parse_formats(["vertical"]) == ["9:16"]


def test_parse_resolution_rounds_to_even():
    assert parse_resolution("1080x1920") == (1080, 1920)
    assert parse_resolution("721x481") == (720, 480)
    with pytest.raises(ConfigError):
        parse_resolution("1080")


def test_config_file_with_bad_yaml(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("ollama:\n  model: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path=bad, environ={})
