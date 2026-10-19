"""refreeze_human: what a model loaded from a Stage-2 checkpoint needs again (the checkpoint stores weights, not requires_grad or hooks)."""
import types

import torch

from terrain_memory_prosthesis.controllers.freeze import refreeze_human


def _model(n_muscle=3, n_exo=1):
    human, exo = torch.nn.Linear(4, n_muscle), torch.nn.Linear(4, n_exo)
    log_std = torch.nn.Parameter(torch.zeros(n_muscle + n_exo))
    net = types.SimpleNamespace(human_policy_net=human, exo_policy_net=exo)
    return types.SimpleNamespace(policy=types.SimpleNamespace(policy_network=net, log_std=log_std)), human, exo, log_std


def test_human_is_frozen_and_the_prosthesis_is_not():
    m, human, exo, _ = _model()
    refreeze_human(m, n_muscle=3)
    assert not any(p.requires_grad for p in human.parameters())
    assert all(p.requires_grad for p in exo.parameters())


def test_log_std_gradient_only_reaches_the_prosthesis_entry():
    m, _, _, log_std = _model(n_muscle=3, n_exo=1)
    refreeze_human(m, n_muscle=3)
    (log_std * torch.arange(1.0, 5.0)).sum().backward()
    assert torch.equal(log_std.grad, torch.tensor([0.0, 0.0, 0.0, 4.0]))


def test_weights_are_not_touched():
    m, human, _, log_std = _model()
    w = human.weight.detach().clone()
    refreeze_human(m, n_muscle=3)
    assert torch.equal(human.weight, w) and float(log_std.abs().sum()) == 0.0
