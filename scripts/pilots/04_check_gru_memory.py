#!/usr/bin/env python
"""04 - Does recurrence buy anything under THIS training recipe? (synthetic memory test, no MuJoCo)

Question : The whole C1-vs-C2 comparison assumes (a) a feed-forward PPO policy cannot use information that is gone from the
           current observation, and (b) a recurrent PPO policy trained the same way can. If (b) fails in our hands
           (hyper-parameters, rollout length, truncation), a "no GRU effect" in the prosthesis experiment would be a bug,
           not a finding. This toy checks it in ~minutes on a laptop.
Task     : cue c in {-1,+1} is visible at t=0 only; then `delay` steps with zero observation; at the last step a query flag is
           shown and the agent must output the cue's sign. Reward 1 if correct, else 0. Delay is random in [dmin, dmax] so the
           moment of the query cannot be learned from a step counter. Chance level = 0.5.
Agents   : MLP   - SB3 PPO, MlpPolicy                         (stand-in for C1: no memory)
           RNN   - sb3-contrib RecurrentPPO, LSTM             (stand-in for C2; sb3-contrib has LSTM only, not GRU)
           PLUGIN- --agent your_module:your_function          (your own GRU controller / stage-2 code):
                   def your_function(make_env, timesteps, seed, n_eval) -> float   # returns accuracy in [0,1]
Verdict  : PASS  MLP <= chance+margin  AND  recurrent >= --rnn-min (0.9)
           WARN  recurrent between chance+margin and rnn-min (undertrained: raise --timesteps) or MLP also high (task leaks the cue)
           FAIL  recurrent at chance (recurrent pipeline cannot learn memory with these settings)
Exit code: 0 PASS, 1 WARN, 2 FAIL.

Example:
    python 04_check_gru_memory.py --timesteps 200000
    python 04_check_gru_memory.py --timesteps 200000 --agent my_gru_agent:train_and_eval
"""
import argparse
import importlib
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (no MyoAssist needed here; we only use its reporting helpers)

import numpy as np  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--dmin", type=int, default=5)
    ap.add_argument("--dmax", type=int, default=20)
    ap.add_argument("--eval-delays", default="", help="extra delays to evaluate generalisation, e.g. 40")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--rnn-min", type=float, default=0.9)
    ap.add_argument("--agent", default=None, help="module:function plugin (see docstring)")
    ap.add_argument("--skip", default="", help="comma list of agents to skip: mlp,rnn")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def make_env_cls():
    import gymnasium as gym
    from gymnasium import spaces

    class CueDelayQuery(gym.Env):
        """obs = [cue, query_flag]; action in {0,1} (0 <-> cue -1, 1 <-> cue +1)."""

        def __init__(self, dmin=5, dmax=20, fixed_delay=None):
            self.dmin, self.dmax, self.fixed = dmin, dmax, fixed_delay
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            self.action_space = spaces.Discrete(2)
            self._rng = np.random.default_rng()

        def reset(self, seed=None, options=None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self.cue = int(self._rng.integers(0, 2))
            self.delay = int(self.fixed if self.fixed is not None else self._rng.integers(self.dmin, self.dmax + 1))
            self.t = 0
            return np.array([2 * self.cue - 1, 0.0], dtype=np.float32), {}

        def step(self, a):
            self.t += 1
            if self.t <= self.delay:  # nothing visible while waiting
                return np.zeros(2, dtype=np.float32), 0.0, False, False, {}
            # query step
            if self.t == self.delay + 1:
                return np.array([0.0, 1.0], dtype=np.float32), 0.0, False, False, {}
            r = 1.0 if int(a) == self.cue else 0.0
            return np.zeros(2, dtype=np.float32), r, True, False, {}

    return CueDelayQuery


def evaluate(predict, Env, n, delay=None, seed=123):
    """predict(obs, state, episode_start) -> action, state. Accuracy over n episodes."""
    ok = 0
    for i in range(n):
        env = Env(fixed_delay=delay) if delay is not None else Env()
        obs, _ = env.reset(seed=seed + i)
        state, start, done, ret = None, np.array([True]), False, 0.0
        while not done:
            a, state = predict(obs, state, start)
            obs, r, done, _, _ = env.step(int(a))
            start = np.array([False])
            ret += r
        ok += ret > 0.5
    return ok / n


def run_agent(kind, args, Env, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    venv = make_vec_env(lambda: Env(args.dmin, args.dmax), n_envs=args.n_envs, seed=seed)
    common_kw = dict(n_steps=args.n_steps, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4, seed=seed, verbose=0, device="cpu")
    if kind == "mlp":
        model = PPO("MlpPolicy", venv, batch_size=256, n_epochs=10, **common_kw)
        pred = lambda obs, st, ep: (model.predict(obs, deterministic=True)[0], None)
    else:
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO("MlpLstmPolicy", venv, batch_size=256, n_epochs=10, policy_kwargs=dict(lstm_hidden_size=64, n_lstm_layers=1), **common_kw)
        pred = lambda obs, st, ep: model.predict(obs, state=st, episode_start=ep, deterministic=True)
    model.learn(total_timesteps=args.timesteps)
    res = {"acc": evaluate(pred, Env, args.n_eval)}
    for d in [int(x) for x in args.eval_delays.split(",") if x.strip()]:
        res[f"acc_delay{d}"] = evaluate(pred, Env, max(100, args.n_eval // 2), delay=d)
    return res


def main():
    args = parse()
    out = common.out_dir(args.out, "pilot_out/04")
    Env = make_env_cls()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    rows = []
    for kind in ("mlp", "rnn"):
        if kind in skip:
            continue
        for s in range(args.seeds):
            print(f"[{kind}] seed {s} ...", flush=True)
            r = run_agent(kind, args, Env, s)
            r.update(agent=kind, seed=s)
            rows.append(r)
            print("   ", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    if args.agent:
        mod, fn = args.agent.split(":")
        sys.path.insert(0, common.ORIG_CWD)
        f = getattr(importlib.import_module(mod), fn)
        for s in range(args.seeds):
            print(f"[plugin {args.agent}] seed {s} ...", flush=True)
            acc = float(f(lambda: Env(args.dmin, args.dmax), args.timesteps, s, args.n_eval))
            rows.append({"agent": "plugin", "seed": s, "acc": acc})
            print("    acc", round(acc, 3))
    common.save_rows(out, "04_memory", rows)

    def mean(kind):
        v = [r["acc"] for r in rows if r["agent"] == kind]
        return sum(v) / len(v) if v else None

    m_mlp, m_rnn, m_plug = mean("mlp"), mean("rnn"), mean("plugin")
    print()
    common.print_table([{"agent": k, "mean_acc": v} for k, v in (("mlp", m_mlp), ("rnn", m_rnn), ("plugin", m_plug)) if v is not None], ["agent", "mean_acc"], {"mean_acc": ".3f"})
    reasons, status = [], "PASS"
    if m_mlp is not None:
        reasons.append(f"MLP accuracy {m_mlp:.2f} (chance 0.50)")
        if m_mlp > 0.5 + args.margin:
            status = "WARN"
            reasons.append("MLP is clearly above chance: the task leaks the cue (or the env is wrong)")
    target = m_plug if m_plug is not None else m_rnn
    label = "plugin (yours)" if m_plug is not None else "recurrent (LSTM)"
    if target is not None:
        reasons.append(f"{label} accuracy {target:.2f}")
        if target < 0.5 + args.margin:
            status = "FAIL"
            reasons.append("recurrent policy is at chance: with this recipe it cannot learn memory (try more timesteps; if still chance, the pipeline is wrong)")
        elif target < args.rnn_min:
            status = "WARN" if status == "PASS" else status
            reasons.append(f"above chance but < {args.rnn_min}: probably undertrained -> raise --timesteps")
        elif status == "PASS":
            reasons.append("memory gap confirmed: recurrence solves what the feed-forward policy cannot")
    return common.verdict(out, "04_check_gru_memory", status, reasons)


if __name__ == "__main__":
    sys.exit(main())
