# Controllers: C1 (feed-forward) and C2 (GRU)

Both controllers are the **prosthesis actor** of MyoAssist's multi-actor policy (one number: the ankle torque command). The
human (muscle policy) is frozen in Stage 2 for both. They are chosen by name (`--controller mlp|gru`,
`controllers/registry.py`); neither knows about the perception axis except through the observation vector.

| | C1 `mlp` | C2 `gru` |
|---|---|---|
| network | the Stage-1 prosthesis MLP (64-64, tanh), zero-extended inputs for the perception block (`controllers/warm_start.py`) | **the same MLP** + a GRU branch (`controllers/gru_module.py`) |
| memory | none: the action depends on the current observation only | a window of the last **K = 16 frames** (0.53 s at 30 Hz, about half a gait cycle) |
| extra observation | - | the env appends the window to the observation (`environment/history.py`, variable `MYO_HISTORY`) |
| training | MyoAssist's default PPO | **the same PPO, the same recipe**, only another policy class (`GruExoActorCriticPolicy`) |

## How C2 is built

```
window of K frames (oldest ... now) --GRU(h=32)--> Linear(32->1) --> delta
current inputs --Stage-1 MLP (pre-tanh)--> pre
action = tanh(pre + delta)
```

* **Starts as C1.** The Linear head is initialised to exactly zero, so at step 0 `delta = 0`: C2 outputs the same torque as C1 and as
  Stage 1 (`check_equivalent`: max difference 0.0 on random observations). The same idea as the zero-initialised preview
  columns of P2. Any later difference between C2 and C1 therefore comes from what the GRU learned.
* **Same inputs.** A frame contains the numbers the MLP itself sees (22 proprioceptive inputs; +5 preview heights in P2), so the
  comparison is "same information, with or without memory" - not "more information".
* **Critic unchanged.** The critic is privileged (it sees the clean preview) and has no memory; the question under study is the
  actor's memory.
* **No state between steps.** The GRU is run over the window from a zero state at every step. This keeps PPO, the rollout
  buffer, evaluation and video tools identical to C1 (sb3-contrib's `RecurrentPPO` does not fit MyoAssist's multi-actor
  policy, and a custom recurrent PPO would put the two controllers on different code paths).

## Limits you must state when you report C2

1. **Finite memory.** The controller remembers K steps, nothing older. This is "a recurrent network over a sliding window" (it
   can integrate, smooth and detect rhythm within 0.5 s), not an unbounded hidden state. K is an experiment parameter:
   `--history 8|16|32` gives a memory-length ablation for free.
2. **Cost.** The GRU is recomputed over K frames for every PPO sample. Microbenchmark of the update (8192 samples, 30 epochs, 2
   minibatches; 1-2 CPU threads, sandbox): K=16/h=32 about 6-11 s extra per PPO iteration, K=16/h=64 18-27 s, K=32/h=64 32-54 s.
   Estimate (not measured on the big node): a C2 run is roughly 1.5x slower than a C1 run with the defaults. Measure with `scripts/ucloud/stage2.sh measure` first.
3. **No previous-action input** (the plan mentioned it). The torque command is not part of the observation; the ankle angle
   and velocity in the window carry its effect. Adding the last command to the frame is a small change in `environment/history.py`.

## Running it

```bash
# one run
python scripts/train_stage2.py --stage1-session $S1 --cond P2 --controller gru --history 16 --seed 0 --timesteps 20000000 --num-envs 128
# the same comparison pipeline as for C1 (training, evaluation, figures, videos): P1xC2 vs P2xC2
python scripts/compare_p1_p2.py --stage1-session $S1 --controller gru --seed 0 --num-envs 64 --timesteps 20000000 --out compare_c2_s0
# UCloud launcher
STAGE1=$S1 CONTROLLER=gru HISTORY=16 COND=P2 bash scripts/ucloud/stage2.sh full
```

`stage2_info.json` stores `controller_env` (`MYO_HISTORY`, `MYO_GRU_HIDDEN`); evaluation, videos and pilots read it
(`environment/myoassist.py::apply_run_env`), and `load_model` loads a C2 checkpoint with its own policy class. Analysis labels
the cells `P1xC2` / `P2xC2` (dashed lines).

To compare C1 and C2, evaluate both in one call (`eval_stage2.py --runs "<C1 runs>" "<C2 runs>"`) and analyse them together
(`analyze_stage2.py`): the figures show the four cells; `differences` is P2 - P1 within one controller, so the
**memory effect** is read as "does the P1xC2 curve beat P1xC1" and "does the P2 advantage shrink with memory".

## Open choices (agree before the long runs)

* Stateful alternative: carry a hidden state across steps and train on sequences (a custom recurrent rollout + minibatching).
  Unbounded memory and cheaper updates, but different code path for C2 and changes to every evaluation tool. Only worth it if
  the windowed C2 shows an effect that the horizon might be limiting.
* K and the GRU size: defaults 16 and 32. One seed of `--history 8/16/32` on P1 shows whether more memory matters at all.
* A critic with memory (only relevant if the actor's memory clearly helps and the value estimate becomes the bottleneck).
