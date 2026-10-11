"""Unit test of load_and_freeze_human on a minimal stand-in with the same attribute layout as HumanExoActorCriticPolicy.
The real end-to-end check (needs MyoAssist) is scripts/pilots/01_check_freeze.py."""
import torch
from torch import nn

from terrain_memory_prosthesis.utils.freeze import load_and_freeze_human


class _PN(nn.Module):
    def __init__(self, exo_in):
        super().__init__()
        self.human_policy_net = nn.Linear(4, 18)
        self.exo_policy_net = nn.Linear(exo_in, 1)


class _Policy(nn.Module):
    def __init__(self, exo_in):
        super().__init__()
        self.policy_network = _PN(exo_in)
        self.log_std = nn.Parameter(torch.zeros(19))


class _Model:
    def __init__(self, exo_in):
        self.policy = _Policy(exo_in)


def test_freeze_keeps_human_and_trains_exo():
    stage1, new = _Model(3), _Model(8)  # P2-like: bigger prosthesis input
    with torch.no_grad():
        stage1.policy.log_std[:18] = -1.0
    load_and_freeze_human(new, stage1, n_muscle=18)
    h0 = {k: v.clone() for k, v in new.policy.policy_network.human_policy_net.state_dict().items()}
    assert all(torch.equal(h0[k], v) for k, v in stage1.policy.policy_network.human_policy_net.state_dict().items())
    assert torch.allclose(new.policy.log_std[:18], torch.full((18,), -1.0))

    e0 = new.policy.policy_network.exo_policy_net.weight.clone()
    opt = torch.optim.Adam([p for p in new.policy.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        loss = new.policy.policy_network.human_policy_net(torch.ones(2, 4)).pow(2).sum() + new.policy.policy_network.exo_policy_net(torch.ones(2, 8)).pow(2).sum() + new.policy.log_std.sum()
        loss.backward()
        opt.step()
    assert all(torch.equal(h0[k], v) for k, v in new.policy.policy_network.human_policy_net.state_dict().items())
    assert torch.allclose(new.policy.log_std[:18], torch.full((18,), -1.0))  # muscle std frozen
    assert not torch.equal(e0, new.policy.policy_network.exo_policy_net.weight)  # prosthesis actor trained
    assert new.policy.log_std[18].item() != 0.0  # prosthesis std trained
