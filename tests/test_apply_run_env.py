"""apply_run_env: a run's recorded settings replace the caller's environment (no noise / delay / dropout leaking in from the shell)."""
import json
import os

from terrain_memory_prosthesis.environment import myoassist as ma


def _run(tmp_path, perception_env):
    (tmp_path / "stage2_info.json").write_text(json.dumps({"perception_env": perception_env, "controller_env": {}}))
    return str(tmp_path)


def test_recorded_sensor_settings_are_applied(tmp_path, monkeypatch):
    monkeypatch.delenv("MYO_SENSOR_NOISE_STD", raising=False)
    ma.apply_run_env(_run(tmp_path, {"MYO_PERCEPTION": "none", "MYO_SENSOR_NOISE_STD": "0.05"}))
    assert os.environ["MYO_SENSOR_NOISE_STD"] == "0.05"
    monkeypatch.delenv("MYO_SENSOR_NOISE_STD")  # keep the test process clean


def test_run_without_sensor_settings_does_not_inherit_them_from_the_shell(tmp_path, monkeypatch):
    for k in ("MYO_SENSOR_NOISE_STD", "MYO_SENSOR_DELAY_STEPS", "MYO_SENSOR_DROPOUT_PROBABILITY"):
        monkeypatch.setenv(k, "9")
    ma.apply_run_env(_run(tmp_path, {"MYO_PERCEPTION": "none"}))
    assert not any(k.startswith("MYO_SENSOR_") for k in os.environ)
