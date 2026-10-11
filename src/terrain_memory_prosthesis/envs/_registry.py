"""Register an extra env id with gym AND with MyoAssist's EnvironmentHandler (exo policy + config type). Idempotent."""
from __future__ import annotations


def register_exo_env(env_id: str, entry_point: str, max_episode_steps: int = 1000):
    from rl_train.envs import register_env_myoassist
    from rl_train.envs import environment_handler as eh
    from myosuite.utils import gym

    if env_id not in gym.envs.registry:
        register_env_myoassist(id=env_id, entry_point=entry_point, max_episode_steps=max_episode_steps, kwargs={})
    if env_id not in eh._EXO_ENV_IDS:
        eh._EXO_ENV_IDS = tuple(eh._EXO_ENV_IDS) + (env_id,)
    H = eh.EnvironmentHandler
    patched = getattr(H, "_extra_env_ids", None)
    if patched is None:
        patched = H._extra_env_ids = set()
        orig = H.get_config_type_from_session_id

        def get_config_type_from_session_id(session_id, _orig=orig):
            if session_id in H._extra_env_ids:
                from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig

                return ExoImitationTrainSessionConfig
            return _orig(session_id)

        H.get_config_type_from_session_id = staticmethod(get_config_type_from_session_id)
    patched.add(env_id)
