import numpy as np

from terrain_memory_prosthesis.evaluation.rollout import Perturbation, seeded_rng


def test_perturbation_scale_zero_delay():
    a = np.array([0.5, -0.5, 0.8])  # last entry = prosthesis command
    ids = [2]
    assert np.allclose(Perturbation("none", 0, ids)(a), a)
    assert np.allclose(Perturbation("scale", 0.5, ids)(a), [0.5, -0.5, 0.4])
    assert np.allclose(Perturbation("zero", 0, ids)(a), [0.5, -0.5, 0.0])
    assert np.allclose(Perturbation("scale", 5.0, ids)(a)[2], 1.0)  # clipped to [-1, 1]

    p = Perturbation("delay", 2, ids)
    out = [p(np.array([0, 0, v]))[2] for v in (0.1, 0.2, 0.3, 0.4)]
    assert np.allclose(out, [0.0, 0.0, 0.1, 0.2])  # command arrives 2 control steps late
    p.reset()
    assert p(np.array([0, 0, 0.9]))[2] == 0.0


def test_perturbation_only_touches_device_entries():
    a = np.array([0.3, 0.3, 0.3])
    out = Perturbation("noise", 0.2, [2], seed=1)(a)
    assert np.allclose(out[:2], a[:2]) and out[2] != a[2]


def test_seeded_rng_makes_default_rng_reproducible():
    with seeded_rng(5):
        a = np.random.default_rng().integers(0, 1000, 5)
    with seeded_rng(5):
        b = np.random.default_rng().integers(0, 1000, 5)
    assert (a == b).all()
    assert np.random.default_rng.__module__ != "builtins"  # restored afterwards
