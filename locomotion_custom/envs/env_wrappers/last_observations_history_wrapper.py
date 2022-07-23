import gym
import numpy as np


class LastObservationsHistoryWrapper(gym.ObservationWrapper):
  """An env wrapper that keeps the last couple of observations."""
  def __init__(self, gym_env, n_observations: int = 4):
    super(LastObservationsHistoryWrapper, self).__init__(gym_env)
    self._observations_history = None
    self._n_observations = n_observations

    low = np.array([self.observation_space.low] * n_observations)
    high = np.array([self.observation_space.high] * n_observations)
    self.observation_space = gym.spaces.Box(low, high)

  def observation(self, new_observations):
    if self._observations_history is None:
        self._observations_history = np.zeros((self._n_observations, new_observations.shape[0]))

    self._observations_history = np.concatenate([self._observations_history, [new_observations]])[1:]

    return self._observations_history

  def reset(self, **kwargs):
      observation = self.env.reset(**kwargs)
      self._observations_history = None
      return self.observation(observation)

