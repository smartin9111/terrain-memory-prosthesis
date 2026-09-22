# Terrain Preview vs. Controller Memory

Research project investigating the trade-off between exteroceptive terrain perception and temporal controller memory for powered ankle prosthesis locomotion in MyoAssist.

## Simulation

- MyoAssist / MuJoCo
- Human model: `myolegs22`
- Prosthesis: `OpenSourceLeg_A_L1`

## Experimental Design

|                     | C1: Feed-forward PPO | C2: Recurrent PPO + GRU |
|---------------------|----------------------|--------------------------|
| P1: Proprioception  | P1C1                 | P1C2                     |
| P2: Terrain Preview | P2C1                 | P2C2                     |

## Perception

- P1: proprioceptive observations only
- P2: proprioception + terrain-height preview

## Controllers

- C1: feed-forward PPO controller
- C2: GRU-based recurrent PPO controller

## Main Research Question

How much exteroceptive terrain preview is required for robust prosthetic locomotion, and can temporal controller memory compensate for incomplete or noisy environmental perception?

## Repository Structure

configs/ - Experiment configurations

src/terrain_memory_prosthesis/
- perception/ - Perception methods
- controllers/ - Controller architectures
- envs/ - MyoAssist integration
- evaluation/ - Evaluation metrics
- utils/ - Shared utilities

scripts/ - Training and evaluation scripts

tests/ - Unit tests

results/ - Experiment outputs

docs/ - Project documentation

## Development Responsibilities

### Perception
- Proprioceptive observation
- Terrain preview
- Noise / dropout / delay

### Control
- Feed-forward PPO
- GRU-based recurrent controller
- Controller evaluation

### Shared
- MyoAssist integration
- Experiments
- Results
- Report
