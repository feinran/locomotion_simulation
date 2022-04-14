"""Setup such that environment can be created using gym.make()."""
# from locomotion_simulation.locomotion_custom.envs.gym_envs.a1_gym_env import A1GymEnvCustom

from importlib import import_module
A1GymEnvCustom = getattr(import_module('locomotion_simulation.locomotion_custom.envs.gym_envs.a1_gym_env'), 'A1GymEnvCustom')
