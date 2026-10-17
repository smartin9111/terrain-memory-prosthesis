"""warm_start on a minimal stand-in with the attribute layout of HumanExoActorCriticPolicy (the real check is scripts/train_stage2.py)."""
import pytest
import torch
from torch import nn

from terrain_memory_prosthesis.controllers.warm_start import check_equivalent, copy_linear_stack, warm_start


def mlp(i, o, tanh_out=False):
    layers = [nn.Linear(i, 8), nn.Tanh(), nn.Linear(8, o)]
    return nn.Sequential(*layers, nn.Tanh()) if tanh_out else nn.Sequential(*layers)


class PN(nn.Module):
    def __init__(self, n_obs, exo_in, critic_in):
        super().__init__()
        self.n_obs = n_obs
        self.human_policy_net = mlp(4, 18, True)
        self.exo_policy_net = mlp(exo_in, 1, True)
        self.value_net = mlp(critic_in, 1)

    def forward_actor(self, obs):  # human sees obs[:4]; exo sees obs[4:4+exo_in] (preview = trailing columns)
        e = self.exo_policy_net[0].in_features
        return torch.cat([self.human_policy_net(obs[:, :4]), self.exo_policy_net(obs[:, 4:4 + e])], 1)

    def forward_critic(self, obs):
        return self.value_net(obs[:, : self.value_net[0].in_features])


class Policy(nn.Module):
    def __init__(self, n_obs, exo_in, critic_in):
        super().__init__()
        self.policy_network = PN(n_obs, exo_in, critic_in)
        self.log_std = nn.Parameter(torch.zeros(19))


class Space:
    def __init__(self, n):
        self.shape = (n,)


class Model:
    def __init__(self, n_obs, exo_in, critic_in):
        self.policy = Policy(n_obs, exo_in, critic_in)
        self.observation_space = Space(n_obs)


def _models(exo_new):
    torch.manual_seed(0)
    s1 = Model(14, 6, 14)  # stage 1: exo sees 6 inputs, critic 14
    torch.manual_seed(1)
    new = Model(19, exo_new, 19)  # + 5 preview columns (critic always, exo only for P2)
    return s1, new


@pytest.mark.parametrize("exo_new", [6, 11])  # P1 / P2
def test_warm_start_is_equivalent_to_stage1(exo_new):
    s1, new = _models(exo_new)
    with torch.no_grad():
        s1.policy.log_std[18] = -0.7
    info = warm_start(new, s1)
    assert info["exo_in_new"] == exo_new and info["critic_in_new"] == 19
    assert check_equivalent(new, s1, n_obs_old=14) < 1e-6
    assert float(new.policy.log_std[18]) == pytest.approx(-0.7)


def test_p2_extra_columns_are_zero_and_trainable():
    s1, new = _models(11)
    warm_start(new, s1)
    w = new.policy.policy_network.exo_policy_net[0].weight
    assert torch.count_nonzero(w[:, 6:]) == 0 and torch.count_nonzero(w[:, :6]) > 0
    assert w.requires_grad and not any(p.requires_grad for p in new.policy.policy_network.human_policy_net.parameters())


def test_wrong_column_order_is_detected():
    s1, new = _models(11)
    warm_start(new, s1)
    with torch.no_grad():  # simulate: preview columns placed FIRST instead of last
        w = new.policy.policy_network.exo_policy_net[0].weight
        w.copy_(torch.roll(w, 5, dims=1))
    with pytest.raises(RuntimeError):
        check_equivalent(new, s1, n_obs_old=14)


def test_copy_linear_stack_rejects_bad_shapes():
    with pytest.raises(ValueError):
        copy_linear_stack(mlp(4, 1), mlp(6, 1))
