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
import numpy as np
from locomotion_simulation.locomotion_custom.envs.locomotion_gym_env import LocomotionGymEnv

from locomotion_simulation.locomotion_custom.robots.a1 import INIT_MOTOR_ANGLES

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
  inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class MoveForwardTask(object):
  """move forward task."""

  def __init__(
      self,
      z_constrain=False,
      move_forward_coeff=1,
      other_direction_penalty=0,
      z_penalty=0,
      orientation_penalty=0,
      time_step_s=0.01,
      num_action_repeat=10,
      height_fall_coeff=0.3,
      alive_reward=0.1,
      fall_reward=0,
      target_vel=None,
      check_contact=False,
      target_vel_dir=(1, 0),
      subgoal_reward=None
      # init_orientation=None,
  ):
    """Initializes the task."""
    self._draw_ref_model_alpha = 1.
    # self.energy_weight = -0.01
    self.energy_weight = -0.005
    self.move_forward_coeff = move_forward_coeff
    self._ref_model = -1
    self._alive_reward = alive_reward
    self.fall_reward = fall_reward
    self._time_step = time_step_s
    self.num_action_repeat = num_action_repeat
    self.z_constrain = z_constrain
    self.other_direction_penalty = other_direction_penalty
    self.z_penalty = z_penalty
    self.init_orientation = np.array([0, 0, 0, 1])
    self.orientation_penalty = orientation_penalty
    self.height_fall_coeff = height_fall_coeff
    self.target_vel = target_vel
    self.check_contact = check_contact
    # return
    self.target_vel_dir = np.array(target_vel_dir)
    self.subgoal_reward = subgoal_reward

    # log data
    self._move_reward = 0 
    self._align_reward = 0
    self._energy_reward = 0

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos

    if self.subgoal_reward is not None:
      self.subgoal_trackers = np.ones(
        len(env._env_randomizers[-1].subgoal_ids),
        dtype=np.uint8
      )

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()

  def done(self, env):
    """Checks if the episode is over."""
    del env
    env = self._env
    pyb = env._pybullet_client

    root_pos_sim, _ = pyb.getBasePositionAndOrientation(
      env.robot.quadruped)

    rot_quat = env.robot.GetBaseOrientation()
    rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)

    rot_fall = rot_mat[-1] < 0.6
    height_fall = root_pos_sim[2] < self.height_fall_coeff
    if self.z_constrain:
      height_fall = root_pos_sim[2] < self.height_fall_coeff or root_pos_sim[2] > 0.8

    contact_done = False

    if self.check_contact:
      contacts = env._pybullet_client.getContactPoints(
        bodyA=env.robot.quadruped)
      for contact in contacts:
        if contact[2] is env._world_dict["ground"] or \
            ("terrain" in env._world_dict and
             contact[2] is env._world_dict["terrain"]):
          if contact[3] not in env.robot._foot_link_ids:
            contact_done = True
            break

      # contacts = env._pybullet_client.getContactPoints(bodyA=env.robot.quadruped)
      for contact in contacts:
        if contact[2] is not env._world_dict["ground"]:
          contact_done = True
          break
      speed = (np.array(self.current_base_pos) - np.array(self.last_base_pos)) / \
        (self._time_step * self.num_action_repeat)

      contact_done = contact_done and np.linalg.norm(speed) <= 0.05
    done = height_fall or rot_fall or contact_done
    return done

  def reward(self, env):
    """Get the reward without side effects."""
    del env

    env = self._env
    # energy_reward = np.abs(
    #     np.dot(env.robot.GetMotorTorques(),
    #            env.robot.GetMotorVelocities())) * self._time_step

    energy_reward = np.dot(
      env.robot.GetMotorTorques(),
      env.robot.GetMotorTorques()
    ) * self._time_step

    move_forward_reward = self._calc_reward_root_velocity()
    alive_reward = self._alive_reward
    orientation_reward = self._calc_reward_rotation()

    # provide logging data
    self._move_reward = move_forward_reward
    self._align_reward = orientation_reward
    self._energy_reward = energy_reward

    reward = move_forward_reward * self.move_forward_coeff + \
      energy_reward * self.energy_weight - \
      self.orientation_penalty * orientation_reward + \
      alive_reward
    # print("Rew:{:.4f} Move Rew:{:.4f}, Ori Rew:{:.4f}, Eng Rew:{:.4f}".format(
    #   reward, move_forward_reward, orientation_reward, energy_reward))
    # print(move_forward_reward)
    done = self.done(env)
    if done:
      reward += self.fall_reward

    if self.subgoal_reward is not None:
      # self.last_base_pos = env.robot.GetBasePosition()
      # print(self.current_base_pos)
      dis = env._env_randomizers[-1].subgoal_centers - \
        self.current_base_pos[:2]
      dis = np.linalg.norm(dis, axis=1)

      contacted_ones = np.where(
        (dis < env._env_randomizers[-1].radius) * self.subgoal_trackers
      )[0]

      for contacted_idx in contacted_ones:
        self.subgoal_trackers[contacted_idx] = 0
        reward += self.subgoal_reward

        env.pybullet_client.changeVisualShape(
          env._env_randomizers[-1].subgoal_ids[contacted_idx],
          -1,
          rgbaColor=(1, 0.2, 0.2, 0)
        )
        # env.pybullet_client.removeBody(contacted_idx)

    return reward

  def _get_pybullet_client(self):
    """Get bullet client from the environment"""
    return self._env._pybullet_client

  def _calc_reward_root_velocity(self):
    """Get the root velocity reward."""
    env = self._env
    robot = env.robot
    sim_model = robot.quadruped

    pyb = self._get_pybullet_client()

    root_vel_sim, _ = pyb.getBaseVelocity(sim_model)
    root_vel_sim = np.array(root_vel_sim)

    x_speed = (self.current_base_pos[0] - self.last_base_pos[0]
               ) / (self._time_step * self.num_action_repeat)
    y_speed = (self.current_base_pos[1] - self.last_base_pos[1]
               ) / (self._time_step * self.num_action_repeat)
    z_speed = (self.current_base_pos[2] - self.last_base_pos[2]
               ) / (self._time_step * self.num_action_repeat)

    xy_speed = np.array([x_speed, y_speed])

    along_speed = np.dot(xy_speed, self.target_vel_dir)
    per_speed = xy_speed - along_speed * self.target_vel_dir

    along_speed = np.clip(
      along_speed, a_min=None, a_max=self.target_vel
    )
    along_reward = self.target_vel ** 2 - (
      along_speed - self.target_vel
    ) ** 2

    forward_reward = along_reward - \
      self.other_direction_penalty * (np.linalg.norm(per_speed) ** 2) - \
      self.z_penalty * (z_speed ** 2)

    return forward_reward

  def _calc_reward_rotation(self):
    env = self._env
    pyb = self._get_pybullet_client()

    rot_quat = env.robot.GetBaseOrientation()

    if self.init_orientation is None:
      return 0
    # Norm of displacement vector
    rot_reward = np.sum(
      (self.init_orientation - np.array(rot_quat)) ** 2)  # * self.num_action_repeat
    return rot_reward

  @property
  def move_reward(self):
      return self._move_reward
  
  @property
  def align_reward(self):
      return self._align_reward
  
  @property
  def energy_reward(self):
      return self._energy_reward


class TwoStageTask(MoveForwardTask):
  def __init__(
    self,
    z_constrain=False,
    move_forward_coeff=1,
    other_direction_penalty=0, 
    z_penalty=0,
    orientation_penalty=0, 
    time_step_s=0.01, 
    num_action_repeat=10, 
    height_fall_coeff=0.3, 
    alive_reward=0.1, 
    fall_reward=0, 
    target_vel=None, 
    check_contact=False, 
    target_vel_dir=(1,0), 
    subgoal_reward=None):
    
    super().__init__(
        z_constrain, 
        move_forward_coeff, 
        other_direction_penalty, 
        z_penalty, 
        orientation_penalty, 
        time_step_s, 
        num_action_repeat, 
        height_fall_coeff, 
        alive_reward, 
        fall_reward, 
        target_vel, 
        check_contact, 
        target_vel_dir, 
        subgoal_reward)
    
    self.env: LocomotionGymEnv = None
    
    self.intial_leg_postions = INIT_MOTOR_ANGLES
    
    self.leg_position_threshold = 0.05
    self.orientation_threshold = 0.05
    
    self.leg_position_weight = -1 
    self.orientaion_weigth = -1
    
    self.walking_task = True

    self.walking_task_step_counter = 0
    self.standing_up_task_counter = 0    

    
  def reset(self, env):
    self.env = env
    return super().reset(env)

  def done(self, env):
    """
    function will be used to switch tasks
    """
    
    # update env
    self.env = env
    
    if self.walking_task and super().done(env=env):
      # switch to stand up task if walking task would have ended
      print("switched to standing up task")
      self.walking_task = False
    
    elif not self.walking_task and self._calc_standing_up_task():
      # check if standing up task has ended and switch back to walking task
      print("switched to walking task")
      self.walking_task = True
    
    return False

  def reward(self, env):
    """
    Depending if the task is to move foreward or standing up again and go
    into initial position the will be a specified reward
    
    """
    if self.walking_task:
      
      self.walking_task_step_counter += 1
      
      move_foreward_reward = super().reward(env)
      return move_foreward_reward
    else:
      
      self.standing_up_task_counter += 1
      
      reset_reward = self._calc_reset_reward()
      return reset_reward
    
  def _calc_standing_up_task(self):
    """
    getting back to walking task if the compined error is lower than a certain threshold
    """
    
    orientation_error = super()._calc_reward_rotation()
    leg_error = self._calc_reward_leg_position()
    
    if orientation_error <= self.orientation_threshold and leg_error <= self.leg_position_threshold:
      return True
    
    return False

  def _calc_reset_reward(self):
    """
    reward for going back into the robots initial position where it can start to start walking
    """
    orientation_reward = super()._calc_reward_rotation()

    leg_position_reward = self._calc_reward_leg_position()
    
    # both weights has to negative because we want to minimize error-rates -> turning this into a maximization
    return self.orientaion_weigth * orientation_reward + self.leg_position_weight * leg_position_reward

  def _calc_reward_leg_position(self):
    """
    caculation MSE between current leg postions and goal leg positions to recreate the initial state
    """

    # create local variable for env
    env = self._env

    # get current motor angles
    motor_angles = env.robot.GetTrueMotorAngles()

    # compute MSE loss between goal leg angles and curent leg angles
    leg_reward = np.mean(
        (self.intial_leg_postions - motor_angles)**2
    )

    return leg_reward
  
  @property
  def walking_percentage(self):
    """How long was the robot trying to learn the walking task

    Returns:
        float: percentage of walking task
    """
    return self.walking_task_step_counter / self.env.current_rollout_step
  
  @property
  def standing_up_percentage(self):
    """How long was the robot trying to learn the standing up task

    Returns:
        float: percentage of standing up task
    """
    return self.standing_up_task_counter / self.env.current_rollout_step
  