"""Stage-2 warm start: build the prosthesis controller from the Stage-1 policy, with the human frozen.

Why a warm start: P1 and P2 must start from the SAME function. P1 = the Stage-1 prosthesis actor unchanged; P2 = the same
actor with extra input columns (the terrain preview) initialised to ZERO, so at step 0 it computes exactly the same torque
as P1 and learning decides whether it uses the preview at all. Any later difference between P1 and P2 is therefore caused
by the preview, not by a different initialisation. The critic gets the preview in both conditions (zero-initialised
columns), so the value baseline is the same too.

`warm_start` relies on the new input columns being the LAST ones of the first layer (the preview block is appended to the
observation lists - see common/session_config.py) and verifies it numerically with `check_equivalent`.
"""
from __future__ import annotations

import torch

from terrain_memory_prosthesis.common.obs_layout import N_PROPRIO, N_SENSOR_VALUES
from torch import nn

from terrain_memory_prosthesis.controllers.freeze import load_and_freeze_human


def copy_linear_stack(new: nn.Sequential, old: nn.Sequential) -> None:
    """Copy all Linear layers of `old` into `new`. The first layer of `new` may have EXTRA input columns (the last ones):
    they are set to zero, so `new` computes the same function as `old` while the extra inputs are ignored."""
    nl = [m for m in new if isinstance(m, nn.Linear)]
    ol = [m for m in old if isinstance(m, nn.Linear)]
    if len(nl) != len(ol):
        raise ValueError(f"layer count differs: new {len(nl)} vs old {len(ol)}")
    with torch.no_grad():
        for i, (n, o) in enumerate(zip(nl, ol)):
            if n.weight.shape == o.weight.shape:
                n.weight.copy_(o.weight)
            elif i == 0 and n.out_features == o.out_features and n.in_features > o.in_features:
                n.weight.zero_()
                n.weight[:, : o.in_features] = o.weight
            else:
                raise ValueError(f"layer {i}: cannot map {tuple(o.weight.shape)} -> {tuple(n.weight.shape)}")
            n.bias.copy_(o.bias)


def warm_start(model, stage1_model, n_muscle: int = 18) -> dict:
    """Human <- Stage 1 (frozen); prosthesis actor + critic <- Stage 1 (zero-extended); prosthesis log_std <- Stage 1."""
    load_and_freeze_human(model, stage1_model, n_muscle=n_muscle)
    npn, opn = model.policy.policy_network, stage1_model.policy.policy_network
    copy_linear_stack(npn.exo_policy_net, opn.exo_policy_net)
    copy_linear_stack(npn.value_net, opn.value_net)
    with torch.no_grad():
        model.policy.log_std[n_muscle:] = stage1_model.policy.log_std[n_muscle:].to(model.policy.log_std.device)
    return {
        "exo_in_new": npn.exo_policy_net[0].in_features, "exo_in_old": opn.exo_policy_net[0].in_features,
        "critic_in_new": npn.value_net[0].in_features, "critic_in_old": opn.value_net[0].in_features,
    }


@torch.no_grad()
def check_equivalent(model, stage1_model, n_obs_old: int, n: int = 256, seed: int = 0, atol: float = 1e-6) -> float:
    """Compare deterministic actions/values after making the duplicated sensor block equal to clean proprioception.

    This verifies the zero-imperfection Stage-2 actor reproduces Stage 1; validity/perception/privileged extras must be
    ignored initially because their added first-layer weights are zero.
    """
    g = torch.Generator().manual_seed(seed)
    n_obs_new = model.observation_space.shape[0]
    obs = torch.randn(n, n_obs_new, generator=g)
    # The Stage-2 exo actor now reads the sensor-values block, not the clean base block.
    # Set that duplicate to the exact same measurements so the zero-effect configuration
    # must reproduce Stage 1. Validity and any perception columns remain random; warm-start
    # zero columns must make them irrelevant at initialization.
    sensor_start = N_PROPRIO
    sensor_end = sensor_start + N_SENSOR_VALUES
    if n_obs_new >= sensor_end:
        obs[:, sensor_start:sensor_end] = obs[:, :N_PROPRIO]
    a_new = model.policy.policy_network.forward_actor(obs)
    a_old = stage1_model.policy.policy_network.forward_actor(obs[:, :n_obs_old])
    v_new = model.policy.policy_network.forward_critic(obs)
    v_old = stage1_model.policy.policy_network.forward_critic(obs[:, :n_obs_old])
    diff = float(max((a_new - a_old).abs().max(), (v_new - v_old).abs().max()))
    if diff > atol:
        raise RuntimeError(f"warm start is not equivalent to Stage 1 (max diff {diff:.2e}): the preview columns are probably not the last ones")
    return diff
