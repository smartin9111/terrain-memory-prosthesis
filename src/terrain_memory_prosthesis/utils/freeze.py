"""Stage-2 helper: copy ONLY the human actor (+ its muscle log_std) from a stage-1 policy into a new policy
whose prosthesis actor has a different input size (e.g. P2 = P1 + terrain preview), then freeze the human.

Usage (after the model for stage 2 has been built with EnvironmentHandler.get_stable_baselines3_model):
    stage1 = PPO.load(path, custom_objects={"policy_class": HumanExoActorCriticPolicy})
    load_and_freeze_human(model, stage1, n_muscle=18)
"""
import torch


def load_and_freeze_human(model, stage1_model, n_muscle: int = 18):
    new_pn = model.policy.policy_network
    old_pn = stage1_model.policy.policy_network

    # 1) human actor weights + muscle part of the shared log_std
    new_pn.human_policy_net.load_state_dict(old_pn.human_policy_net.state_dict())
    with torch.no_grad():
        model.policy.log_std[:n_muscle] = stage1_model.policy.log_std[:n_muscle].to(model.policy.log_std.device)

    # 2) freeze the human network
    for p in new_pn.human_policy_net.parameters():
        p.requires_grad_(False)

    # 3) log_std is ONE parameter for all actions: zero the gradient on the muscle entries only
    mask = torch.ones_like(model.policy.log_std)
    mask[:n_muscle] = 0.0
    model.policy.log_std.register_hook(lambda g: g * mask.to(g.device))
    return model
