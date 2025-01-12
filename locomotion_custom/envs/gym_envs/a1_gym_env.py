"""Wrapper to make the a1 environment suitable for OpenAI gym."""
import gym

from locomotion_simulation.locomotion_custom.envs import env_builder
from locomotion_simulation.locomotion_custom.robots import a1
from locomotion_simulation.locomotion_custom.robots import robot_config


class A1GymEnvCustom(gym.Env):
    """A1 environment that supports the gym interface."""
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 action_limit=(0.75, 0.75, 0.75),
                 render=False,
                 config={},
                 on_rack=False):
        self._env = env_builder.build_regular_env(
            a1.A1,
            motor_control_mode=robot_config.MotorControlMode.POSITION,
            enable_rendering=render,
            action_limit=action_limit,
            on_rack=on_rack,
            config=config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def close(self):
        self._env.close()

    def render(self, mode):
        return self._env.render(mode)

    def __getattr__(self, attr):
        return getattr(self._env, attr)
