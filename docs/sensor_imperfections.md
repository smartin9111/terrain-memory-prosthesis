# Proprioceptive sensor imperfections (P1)

`common/sensor_effects.py` implements the reusable sensor model for the existing 40-channel proprioceptive observation. It deliberately does not change the Stage-1 observation definition or depend on MuJoCo/MyoAssist.

## Supported effects

- **Gaussian measurement noise**: per-channel standard deviation in that channel's native units.
- **Fixed-step latency**: return a measurement from `delay_steps` control steps earlier. The delay history is cleared on reset.
- **Dropout**: independently invalidate channels with a configurable probability each step.
- **Validity mask**: invalid values are set to zero and the corresponding validity value is `0`; valid values have validity `1`. The helper returns the values and mask separately, or `observe_vector()` concatenates them as `[values | validity]`.
- **Reproducibility**: all random draws come from a NumPy generator initialized by `reset(seed)`.

## Example

```python
from terrain_memory_prosthesis.common.sensor_effects import ProprioceptiveSensorModel

sensor = ProprioceptiveSensorModel(
    size=40,
    noise_std=0.01,                 # scalar or 40 per-channel values
    delay_steps=1,
    dropout_probability=0.01,       # scalar or 40 per-channel values
)
sensor.reset(seed=123)
measurement = sensor.observe(clean_proprioception)
actor_features = sensor.observe_vector(clean_proprioception)  # 80 values: 40 values + 40 validity bits
```

The noise scale is expressed in each input's native units, so one shared scalar is only appropriate if the observation channels have comparable scales. For realistic experiments, configure per-channel noise levels or disable noise for channels that should not be perturbed.

## Integration note

This helper is intentionally a tested building block, not yet wired into `preview_env.py` or the Stage-2 policy. The 40-value Stage-1 observation must remain unchanged for the frozen human actor. Therefore, do **not** corrupt the shared environment observation in place: that would also perturb the frozen human actor. The next integration step must route a corrupted 40-value block (and, if using dropout, its 40 validity bits) to the prosthesis actor only, while preserving the clean 40-value prefix for the human actor and the clean privileged preview for the critic. The actor's observation ranges and warm-start weight mapping must be updated together.


## Environment and policy interface

The preview environment now exposes the following observation layout:

`[clean proprioception (40) | sensor values (40) | validity (40) | perception (0 or N) | privileged terrain (5)]`

The clean first 40 values are unchanged. The human actor continues to read those clean values, and the critic continues to read its existing clean inputs plus the privileged terrain block. The prosthesis actor's original Stage-1 observation ranges are redirected to the corresponding channels in the duplicated sensor-values block, then the corresponding validity channels are appended. Thus it receives the same 22 semantic channels as before, plus 22 validity flags. For P2, terrain perception features are appended after those sensor features.

Sensor effects are configured through environment variables inherited by training workers:

- `MYO_SENSOR_NOISE_STD`: scalar or JSON array of 40 standard deviations, in each channel's native units (default `0`).
- `MYO_SENSOR_DELAY_STEPS`: non-negative integer number of control steps (default `0`).
- `MYO_SENSOR_DROPOUT_PROBABILITY`: scalar or JSON array of 40 per-channel probabilities (default `0`).

For example, `MYO_SENSOR_NOISE_STD='0.01'`, `MYO_SENSOR_DELAY_STEPS='1'`, and `MYO_SENSOR_DROPOUT_PROBABILITY='0.02'` enable simple uniform settings. For realistic experiments, prefer per-channel JSON arrays because the 40 signals have different units and scales.

At warm start, the sensor-values block is set equal to clean proprioception for the equivalence check. The new validity inputs and any perception inputs have zero-initialized first-layer weights, so the Stage-2 actor must reproduce Stage-1 actions before training.
