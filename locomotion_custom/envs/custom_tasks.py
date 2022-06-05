# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import numpy as np

from locomotion_simulation.locomotion_custom.envs.locomotion_gym_env import LocomotionGymEnv

class BaseTask():
    """Default task."""

    def __init__(self):
        """Initializes the task."""
        self.current_base_pos = np.zeros(3)
        self.last_base_pos = np.zeros(3)

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env
        self.last_base_pos = env.robot.GetBasePosition()
        self.current_base_pos = self.last_base_pos

    def update(self, env):
        """Updates the internal state of the task."""
        self.last_base_pos = self.current_base_pos
        self.current_base_pos = env.robot.GetBasePosition()

    def done(self, env):
        """Checks if the episode is over.

        If the robot base becomes unstable (based on orientation), the episode
        terminates early.
        """
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        
        return rot_mat[-1] < 0.85

    def reward(self, env):
        """Get the reward without side effects."""
        del env
        return self.current_base_pos[0] - self.last_base_pos[0]


class EnergyTask(BaseTask):
    """Penalize energy consumption"""
    def __init__(self, target_speed, alpha_1, alpha_2, alpha_3):
        super().__init__()
        self.energy_consumption = 0
        self.base_velocity = np.zeros(3)
        self.rpy = (0, 0, 0)

        self.target_speed = target_speed
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alive = target_speed * alpha_3

    def reset(self, env):
        super().reset(env)
        self.energy_consumption = 0
        self.base_velocity = np.zeros(3)
        self.rpy = (0, 0, 0)

    def update(self, env):
        super().update(env)
        self.energy_consumption = env.robot.GetEnergyConsumptionPerControlStep()
        self.base_velocity = env.robot.GetBaseVelocity()
        self.rpy = env.robot.GetTrueBaseRollPitchYaw()

    def reward(self, env):
        """Get the reward without side effects."""
        del env

        forward = -self.alpha_2 * abs(self.target_speed - self.base_velocity[0])
        forward -= self.base_velocity[1] * self.base_velocity[1]
        forward -= self.rpy[2] * self.rpy[2]
        return forward + self.alpha_1 * self.energy_consumption + self.alive


class DirectionTask(BaseTask):
    """Returns reward depending on the direction"""
    def reward(self, env: LocomotionGymEnv):
        """
        Get the reward without side effects.
        """

        # get sensor data
        direction_sensor = env.sensor_by_name("Direction")
        dir = direction_sensor.direction
        
        # how far the robot has moved
        change = np.array(self.current_base_pos[:2]) - np.array(self.last_base_pos[:2])
        magnitude = np.linalg.norm(change)
        change = change / magnitude  # normalized move direction

        rot_quat = env.robot.GetTrueBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        forward = np.array([rot_mat[i] for i in [0, 3]])  # direction where the robot is looking at
    
        if env.rendering_enabled:
            debug_lines(self.current_base_pos, env, forward, None, dir)

        dir = dir / np.linalg.norm(dir)  # normalized target direction
        movement_dot = np.dot(dir, change)
        movement_reward = np.sign(movement_dot) * magnitude * movement_dot * movement_dot
        alignment_dot = np.dot(dir, forward)
        alignment_reward = np.sign(alignment_dot) * magnitude * alignment_dot * alignment_dot

        return movement_reward + alignment_reward


class DirectionSpeedTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.base_velocity = np.zeros(3)

    def reset(self, env):
        super().reset(env)
        self.base_velocity = env.robot.GetBaseVelocity()

    def update(self, env):
        super().update(env)
        self.base_velocity = env.robot.GetBaseVelocity()

    """Returns reward depending on the direction"""
    def reward(self, env: LocomotionGymEnv):
        """
        Get the reward without side effects.
        """

        # get sensor data
        direction_sensor = env.sensor_by_name("Direction")
        dir = direction_sensor.direction 
        speed = direction_sensor.speed
        velocity = np.zeros(3)
        
        # how far the robot has moved
        change = np.array(self.current_base_pos[:2]) - np.array(self.last_base_pos[:2])
        magnitude = np.linalg.norm(change)
        change = change / magnitude  # normalized move direction

        rot_quat = env.robot.GetTrueBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        forward = np.array([rot_mat[i] for i in [0, 3]])  # direction where the robot is looking at

        if speed is not None:
            velocity = np.array(env.robot.GetBaseVelocity()[:2])
            magnitude = speed - np.linalg.norm(dir - velocity)
            
        if env.rendering_enabled:
            self.debug_lines(self.current_base_pos, env, forward, velocity, dir)

        dir = dir / np.linalg.norm(dir)  # normalized target direction
        movement_dot = np.dot(dir, change)
        movement_reward = np.sign(movement_dot) * magnitude * movement_dot * movement_dot
        alignment_dot = np.dot(dir, forward)
        alignment_reward = np.sign(alignment_dot) * magnitude * alignment_dot * alignment_dot

        if speed is not None and magnitude < 0:
            movement_reward = -abs(movement_reward)
            alignment_reward = -abs(alignment_reward)

        return movement_reward + alignment_reward
    


def debug_lines(current_base_pos, env, forward, velocity = None, dir = None):
    env.pybullet_client.addUserDebugLine(
        current_base_pos,
        current_base_pos + np.append(dir, 0) * 2,
        lineColorRGB=[0, 0, 1],
        lineWidth=2.0,
        lifeTime=0.005)

    env.pybullet_client.addUserDebugLine(
        current_base_pos,
        current_base_pos + np.append(forward, 0) * 2,
        lineColorRGB=[1, 0, 0],
        lineWidth=2.0,
        lifeTime=0.005)

    if velocity is not None:
        env.pybullet_client.addUserDebugLine(
            current_base_pos,
            current_base_pos + np.append(velocity, 0) * 2,
            lineColorRGB=[0, 1, 0],
            lineWidth=2.0,
            lifeTime=0.005)
