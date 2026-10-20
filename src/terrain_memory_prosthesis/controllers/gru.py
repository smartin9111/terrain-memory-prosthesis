"""C2: the Stage-1 prosthesis MLP plus a GRU branch over the last K observation frames (zero-initialised residual), human frozen.

Design (decision to review with the team, see docs/controllers.md):
  * memory = a GRU run over a WINDOW of the last K frames that the environment appends to the observation (environment/history.py).
    No hidden state is carried across steps, so SB3's PPO, the rollout buffer, evaluation and videos stay unchanged and C1 and C2
    are trained by the very same code path. The memory horizon is K / 30 s (--history, default 16 = 0.53 s, about half a gait cycle); hidden size 32 (--gru-hidden).
  * the GRU sees the same inputs the prosthesis MLP sees (the sensor values + validity bits it reads [+ the perception block in P2]),
    one frame per step, taken from the sensor block of the observation, never from the clean proprioception (common/obs_layout.py).
  * start: delta = 0 exactly, so C2 == C1 == Stage 1 at step 0 (checked numerically by check_equivalent).
  * the critic is unchanged: it is privileged (clean preview) and needs no memory for the question under study.

Settings travel in environment variables (MYO_HISTORY, MYO_GRU_HIDDEN) like the perception settings: the env, the policy class and
every evaluation / video tool read them, and stage2_info.json stores them (`controller_env`).
"""
from __future__ import annotations

import os

import torch
from rl_train.train.policies.rl_agent_exo import CustomNetworkHumanExo, HumanExoActorCriticPolicy

from terrain_memory_prosthesis.common.obs_layout import GRU_HIDDEN_ENV_VAR, HISTORY_ENV_VAR, ObsLayout
from terrain_memory_prosthesis.controllers.gru_module import HistoryGruResidual, n_percept_from
from terrain_memory_prosthesis.controllers.registry import register
from terrain_memory_prosthesis.controllers.warm_start import warm_start

DEFAULT_HISTORY = 16
DEFAULT_HIDDEN = 32


def _env_vars(a) -> dict:
    return {HISTORY_ENV_VAR: str(getattr(a, "history", None) or DEFAULT_HISTORY),
            GRU_HIDDEN_ENV_VAR: str(getattr(a, "gru_hidden", None) or DEFAULT_HIDDEN)}


def _expand(blocks) -> list[int]:
    out: list[int] = []
    for b in blocks:
        out += list(range(*b["range"])) if b["type"] == "range" else list(b["index"])
    return out


class GruCustomNetwork(CustomNetworkHumanExo):
    """MyoAssist's human + prosthesis network; the prosthesis action is tanh(MLP pre-activation + GRU residual)."""

    def reset_policy_networks(self):
        super().reset_policy_networks()
        if "exo_actor" not in self.net_indexing_info:
            raise NotImplementedError("C2 needs a single 'exo_actor' network (not the shared per-side exo_actor_r / exo_actor_l)")
        n_hist = int(os.environ.get(HISTORY_ENV_VAR, "0") or 0)
        if n_hist < 1:
            raise RuntimeError(f"C2 needs {HISTORY_ENV_VAR}=K (observation window length) in the environment; "
                               "stage2_info.json stores it as controller_env (environment/myoassist.py:apply_run_env)")
        hidden = int(os.environ.get(GRU_HIDDEN_ENV_VAR, "0") or DEFAULT_HIDDEN)
        lay = ObsLayout(n_percept=n_percept_from(self.observation_space.shape[0], n_hist), n_hist=n_hist)
        frame_idx = [i - lay.frame_start for i in _expand(self.net_indexing_info["exo_actor"]["observation"])]
        if min(frame_idx) < 0:
            raise ValueError("the prosthesis actor reads the clean proprioception obs[0:40]; C2 needs it to read the sensor block "
                             "(common/session_config.py routes it there)")
        if not isinstance(self.exo_policy_net[-1], torch.nn.Tanh):
            raise RuntimeError("expected the prosthesis MLP to end in Tanh")
        self.memory = HistoryGruResidual(
            frame_idx=frame_idx, frame_width=lay.frame_width, n_hist=n_hist,
            offset=lay.hist_range[0], hidden=hidden, n_out=self.network_index_handler.get_action_num("exo_actor"),
        )

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.network_index_handler
        pre = self.exo_policy_net[:-1](h.map_observation_to_network(obs, "exo_actor"))
        return h.map_network_to_action({
            "human_actor": self.human_policy_net(h.map_observation_to_network(obs, "human_actor")),
            "exo_actor": torch.tanh(pre + self.memory(obs)),
        })


class GruExoActorCriticPolicy(HumanExoActorCriticPolicy):
    def _build_policy_network(self, observation_space, action_space, custom_policy_params):
        return GruCustomNetwork(observation_space, action_space, custom_policy_params)


def build_model(cfg, env):
    """SB3 PPO with the C2 policy class (what EnvironmentHandler.get_stable_baselines3_model does for C1, minus the mirror variant)."""
    import stable_baselines3
    from rl_train.utils.data_types import DictionableDataclass

    kw = DictionableDataclass.to_dict(cfg.ppo_params)
    if kw.pop("mirror_coef", 0.0) > 0:
        raise NotImplementedError("C2 with the mirror penalty (MirrorPPO) is not supported")
    return stable_baselines3.PPO(policy=GruExoActorCriticPolicy, env=env, policy_kwargs=DictionableDataclass.to_dict(cfg.policy_params),
                                 verbose=2, **kw)


@register("gru", build_model=build_model, env_vars=_env_vars)
def prepare(model, stage1_model, n_muscle: int = 18) -> dict:
    info = warm_start(model, stage1_model, n_muscle=n_muscle)  # human frozen; MLP + critic <- Stage 1 (zero-extended)
    mem = model.policy.policy_network.memory
    mem.zero_head()  # SB3's init_weights has just re-initialised every Linear: restore delta == 0
    info.update(history=mem.n_hist, gru_hidden=mem.gru.hidden_size, gru_inputs=int(mem.frame_idx.numel()))
    return info
