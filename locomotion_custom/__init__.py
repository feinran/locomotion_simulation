"""Set up gym interface for locomotion environments."""
import gym
from gym.envs.registration import registry, make, spec


def register(env_id, *args, **kvargs):
    if env_id in registry.env_specs:
        return
    else:
        print("jajajakljkjjjaj", env_id)
        return gym.envs.registration.register(env_id, *args, **kvargs)


register(
    env_id='custom-v1',
    entry_point='locomotion_simulation.locomotion_custom.envs.gym_envs:A1GymEnvCustom',
    max_episode_steps=2000,
    reward_threshold=2000.0,
)
