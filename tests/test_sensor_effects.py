import numpy as np
import pytest

from terrain_memory_prosthesis.common.sensor_effects import ProprioceptiveSensorModel


def test_clean_sensor_preserves_values_and_marks_them_valid():
    sensor = ProprioceptiveSensorModel(3)
    sensor.reset(seed=10)
    x = np.array([0.2, -1.0, 3.0])
    sample = sensor.observe(x)
    assert np.allclose(sample.values, x)
    assert np.array_equal(sample.validity, np.ones(3))
    assert np.allclose(sensor.observe_vector(x), np.r_[x, np.ones(3)])


def test_noise_is_reproducible_for_same_seed():
    x = np.zeros(4)
    a = ProprioceptiveSensorModel(4, noise_std=0.1)
    b = ProprioceptiveSensorModel(4, noise_std=0.1)
    a.reset(seed=123)
    b.reset(seed=123)
    assert np.array_equal(a.observe(x).values, b.observe(x).values)
    assert np.array_equal(a.observe(x).values, b.observe(x).values)


def test_delay_returns_previous_measurement_and_marks_warmup_invalid():
    sensor = ProprioceptiveSensorModel(2, delay_steps=1)
    sensor.reset(seed=0)
    first = sensor.observe([1.0, 2.0])
    second = sensor.observe([3.0, 4.0])
    assert np.array_equal(first.values, [0.0, 0.0])
    assert np.array_equal(first.validity, [0.0, 0.0])
    assert np.array_equal(second.values, [1.0, 2.0])
    assert np.array_equal(second.validity, [1.0, 1.0])


def test_dropout_zeroes_values_and_exposes_validity():
    sensor = ProprioceptiveSensorModel(4, dropout_probability=1.0)
    sensor.reset(seed=5)
    sample = sensor.observe([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(sample.values, np.zeros(4))
    assert np.array_equal(sample.validity, np.zeros(4))


def test_channel_specific_noise_and_dropout():
    sensor = ProprioceptiveSensorModel(2, noise_std=[0.0, 0.0], dropout_probability=[0.0, 1.0])
    sensor.reset(seed=1)
    sample = sensor.observe([2.0, 9.0])
    assert np.array_equal(sample.values, [2.0, 0.0])
    assert np.array_equal(sample.validity, [1.0, 0.0])


@pytest.mark.parametrize("kwargs", [
    {"noise_std": -1.0},
    {"dropout_probability": 1.1},
    {"delay_steps": -1},
    {"noise_std": [0.1, 0.2, 0.3]},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ProprioceptiveSensorModel(2, **kwargs)


def test_bad_observation_shape_or_nonfinite_values_rejected():
    sensor = ProprioceptiveSensorModel(2)
    with pytest.raises(ValueError):
        sensor.observe([1.0])
    with pytest.raises(ValueError):
        sensor.observe([1.0, np.nan])


def test_reset_clears_delay_history_and_reseeds_rng():
    sensor = ProprioceptiveSensorModel(1, noise_std=0.5, delay_steps=1)
    sensor.reset(seed=42)
    first_a = sensor.observe([1.0])
    next_a = sensor.observe([1.0])
    sensor.reset(seed=42)
    first_b = sensor.observe([1.0])
    next_b = sensor.observe([1.0])
    assert np.array_equal(first_a.values, first_b.values)
    assert np.array_equal(next_a.values, next_b.values)
    assert np.array_equal(first_a.validity, [0.0])
