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

"""Simple sensors related to the environment."""
import numpy as np
import typing
from typing import Tuple
import glob
import json

from scipy.spatial.transform import Rotation
from sklearn.neighbors import VALID_METRICS_SPARSE
from callbacks import create_buckets

from locomotion_simulation.locomotion_custom.envs.sensors import sensor

_ARRAY = typing.Iterable[float]  # pylint:disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]  # pylint:disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any]  # pylint:disable=invalid-name


class LastActionSensor(sensor.BoxSpaceSensor):
    """A sensor that reports the last action taken."""

    def __init__(self,
                 num_actions: int,
                 lower_bound: _FLOAT_OR_ARRAY = -1.0,
                 upper_bound: _FLOAT_OR_ARRAY = 1.0,
                 name: typing.Text = "LastAction",
                 dtype: typing.Type[typing.Any] = np.float64,
                 common_data_path: typing.Text = "") -> None:
        """Constructs LastActionSensor.

        Args:
            num_actions: the number of actions to read
            lower_bound: the lower bound of the actions
            upper_bound: the upper bound of the actions
            name: the name of the sensor
            dtype: data type of sensor value
        """
        self._num_actions = num_actions
        self._env = None
        self._last_action = np.zeros(self._num_actions)
        self.counter = 0

        super(LastActionSensor, self).__init__(name=name,
                                               shape=(self._num_actions,),
                                               lower_bound=lower_bound,
                                               upper_bound=upper_bound,
                                               dtype=dtype,
                                               common_data_path=common_data_path)

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.

        Args:
            env: the environment who invokes this callback function.
        """
        self._env = env
        action_space = self._env.action_space.shape
        self._last_actions = np.zeros((self._num_actions, action_space))
        self.counter = 0
        
    def on_step(self, env):
        self._last_action[self.counter] = self._env.last_action
        self.counter += 1
        
    def _get_observation(self) -> _ARRAY:
        """Returns the last action of the environment."""
        return self._last_actions.flatten()


class CameraArraySensor(sensor.BoxSpaceSensor):
    """sensor that represents the front cameras"""

    def __init__(self,
                 lower_bound: _FLOAT_OR_ARRAY = 0,
                 upper_bound: _FLOAT_OR_ARRAY = 255,
                 name: typing.Text = "CameraArray",
                 resolution: Tuple = (100, 100),
                 dtype: typing.Type[typing.Any] = np.float64,
                 common_data_path: typing.Text = "") -> None:
        """Constructs camera array sensor.

        Args:
          num_actions: the number of actions to read
          lower_bound: the lower bound of the actions
          upper_bound: the upper bound of the actions
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._env = None

        # render shapes for the camera
        self.render_with, self.render_height = resolution

        # Warning is caused by typing from Box Space Sensor shape
        super(CameraArraySensor, self).__init__(name=name,
                                          shape=(self.render_height, self.render_with, 5),
                                          lower_bound=lower_bound,
                                          upper_bound=upper_bound,
                                          dtype=dtype,
                                          common_data_path=common_data_path)

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.

        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    @staticmethod
    def convert_gray2rgb(image_array):
        width, height = image_array.shape
        out = np.empty((width, height, 3), dtype=np.uint8)
        out[:, :, 0] = image_array
        out[:, :, 1] = image_array
        out[:, :, 2] = image_array
        return out

    def _get_observation(self) -> _ARRAY:
        """Returns 3 images from the front of the robot
        - rgb
        - depth
        - segmentation
        """
        base_pos = np.array(self._env._robot.GetBasePosition())

        quaternion_orientation = self._env._robot.GetTrueBaseOrientation()
        # calculate rotation matrix
        M = Rotation.from_quat(quaternion_orientation)
        M = M.as_matrix()

        camera_eye_position = (base_pos + M.T[2] * 0.2).tolist()  # TODO take measurement from real robot
        camera_target_position = camera_eye_position + M.T[0]  # is relative to cam pos
        camera_up_vector = M.T[2]  # is a global vector

        view_matrix = self._env._pybullet_client.computeViewMatrix(
            cameraEyePosition=camera_eye_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=camera_up_vector
        )
        proj_matrix = self._env._pybullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._env._render_width) / self._env._render_height,
            nearVal=0.1,
            farVal=100.0)

        (_, _, px, depth_img, seg_img) = self._env._pybullet_client.getCameraImage(
            width=self.render_with,
            height=self.render_height,
            renderer=self._env._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]

        depth_array = np.array(depth_img)
        seg_array = np.array(seg_img)

        # pack both arrays into one 5 dimensional image
        # dimension 0 to 2: rgb_array
        # dimension 3: depth_array
        # dimension 4: seg_array
        width, height = depth_array.shape
        out = np.empty((width, height, 5), dtype=np.uint8)
        out[:, :, :3] = rgb_array
        out[:, :, 3] = depth_array
        out[:, :, 4] = seg_array
        
        return out


class DirectionSensor(sensor.BoxSpaceSensor):
    """A sensor that reports the direction that the robot should move to."""

    def __init__(self,
                 speed: float = None,
                 distribution: typing.Text = "left_right",
                 mean: float = 0,
                 std: float = 0,
                 lower_sampling_limit: float = 0,
                 upper_sampling_limit: float = np.pi,
                 lower_bound: _FLOAT_OR_ARRAY = -1.0,
                 upper_bound: _FLOAT_OR_ARRAY = 1.0,
                 name: typing.Text = "Direction",
                 dtype: typing.Type[typing.Any] = np.float64,
                 common_data_path: typing.Text = "") -> None:
        """Constructs LastActionSensor.

        Args:
        lower_bound: the lower bound of the actions
        upper_bound: the upper bound of the actions
        name: the name of the sensor
        dtype: data type of sensor value
        """
        self._speed = speed
        self._distribution = distribution
        self._mean = mean
        self._std = std
        self._lower_sampling_limit = lower_sampling_limit
        self._upper_sampling_limit = upper_sampling_limit
        self._direction = np.zeros(2)
        self._angle = 0
        self._rel_angle = 0
        self._env = None
        self._buckets = np.ones(8) # are needed for adapted sampling, is a list with rewards against 360 degree
        
        super().__init__(name=name,
                        shape=(2,),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        dtype=dtype,
                        common_data_path=common_data_path)

    def __sample_angle(self):
        angle = 0
        # decide wich distribution we will use
        if self._distribution == "left_right":
            if np.random.rand() > 0.5:
                angle = np.random.normal(3 * np.pi / 2, np.pi / 4)
            else:
                angle = np.random.normal(np.pi / 2, np.pi / 4)
        elif self._distribution == "uniform":
            angle = np.random.uniform(self._lower_sampling_limit, self._upper_sampling_limit)
        elif self._distribution == "normal":
            angle = np.random.normal(self._mean, self._std)
        elif self._distribution == "adapted":
            # get minimal bucket to zero
            min_weight = min(self._buckets)
            buckets =  self._buckets + min_weight
            # normalize buckets
            buckets /= buckets.sum()
            # get reverse propability
            buckets = 1 - buckets
            # cut of all buckets that do not lay inside the limits
            # -> set the propability to zero
            theta = np.linspace(0, 2*np.pi, len(buckets) + 1)[:-1]
            logic1 = theta > self._upper_sampling_limit 
            logic2 = theta < self._lower_sampling_limit
            buckets[logic1 * logic2] = 0
            # normalize reversed propability
            bucket_sampling_weights = buckets / buckets.sum()
            bucket_idx = np.random.choice(list(range(len(self._buckets))), p=bucket_sampling_weights) 
            
            # get lower / upper limit for uniform sampling
            angels_per_bucket = 2 * np.pi / len(self._buckets)
            lower = bucket_idx * angels_per_bucket - angels_per_bucket  # is negaitve for idx = 0, take modulu 2pi after sampling
            higher = (bucket_idx + 1) * angels_per_bucket - angels_per_bucket
            
            # sample angel from bucket uniformly
            angle = np.random.uniform(lower, higher) % (2 * np.pi)
            
        return angle
    
    def __update_buckets(self):
        """
        1. load data_json 
        2. if there is no json -> buckets = np.ones(8)
        3. get angels and reward_acc
        4. create buckets
        """
        # get latest eval_data_file
        max_iter = 0
        steps = 0
        eval_data_file = ""
        for name in glob.glob(self.get_common_data_path() + "/eval_data_*"):
            steps = int(name.split("_")[-2])
            if steps > max_iter:
                max_iter = steps
                eval_data_file = name
                
        if max_iter == 0:
            return np.ones(8)
                
        # load json
        file = open(eval_data_file)
        data = json.load(file)
        
        return create_buckets(num_buckets = 36, 
                              angels = data["Direction"]["angle"], 
                              accs = data["reward_acc"],
                              lower_limit=self._lower_sampling_limit,
                              upper_limit=self._upper_sampling_limit)

    @staticmethod
    def __retrieve_2D_angle(vector):
        """returns the angle where the vector is pointing at 

        Args:
            vector (_type_): 2D Vector. np.linalg.norm(vector) = 1
        """
        
        if np.arcsin(vector[1]) >= 0:
            return np.arccos(vector[0])
        else:
            return 2 * np.pi - np.arccos(vector[0])
        
    @staticmethod
    def create_direction(angle):
        """returns a normed direction vector

        Args:
            angle (float): is a angel in the interval [0,2 * pi)

        Returns:
            np.array: normed direction vector
        """
        return np.array([np.cos(angle), np.sin(angle)])
    
    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.

        Args:
        env: the environment who invokes this callback function.
        """
        self._env = env
        
        self._buckets = self.__update_buckets()
        
        # get sampled angle
        self._angle = self.__sample_angle()
        self._rel_angle = self._angle
        
        self.on_step(env)
            
    def on_step(self, env):
        # update env
        self._env = env
        
        # get current true robot orientation
        # robot = self._env.get_attr("robot")[0]
        rot_quat = env.robot.GetTrueBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        forward = np.array([rot_mat[i] for i in [0, 3]]) 
        current_robot_angle = self.__retrieve_2D_angle(forward)
        
        self._rel_angle = self._angle - current_robot_angle
        
        # create direction vector
        self._direction = self.create_direction(self._rel_angle)
        
    def _get_observation(self) -> _ARRAY:
        """
        Returns where the robot should move to.
        """
        return self._direction
    
    @property
    def angle(self):
        return self._angle
    
    @property
    def rel_angle(self):
        return self._rel_angle
    
    @property
    def direction(self):
        return self._direction
    
    @property
    def target_direction(self):
        return self.create_direction(self._angle)
    
    @property
    def rel_direction(self):
        return self.create_direction(self._rel_angle)

    @property
    def buckets(self):
        return self._buckets
    
    @buckets.setter
    def buckets(self, value):
        print("setter is called with: ", len(value))
        self._buckets = value
        


class DirectionSensorOld(sensor.BoxSpaceSensor):
    """A sensor that reports the direction that the robot should move to."""

    def __init__(self,
                 speed: float = None,
                 distribution: typing.Text = "left_right",
                 mean: float = 0,
                 std: float = 0,
                 lower_bound: _FLOAT_OR_ARRAY = -1.0,
                 upper_bound: _FLOAT_OR_ARRAY = 1.0,
                 name: typing.Text = "Direction",
                 dtype: typing.Type[typing.Any] = np.float64,
                 common_data_path: typing.Text = "") -> None:
        """Constructs LastActionSensor.

        Args:
        lower_bound: the lower bound of the actions
        upper_bound: the upper bound of the actions
        name: the name of the sensor
        dtype: data type of sensor value
        """
        self.speed = speed
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.direction = np.zeros(2)
        self._angle = 0
        self._env = None
        self._buckets = np.ones(8) # are needed for adapted sampling, is a list with rewards against 360 degree
        
        super().__init__(name=name,
                        shape=(2,),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        dtype=dtype,
                        common_data_path=common_data_path)

    def __sample_angle(self):
        angle = 0
        # decide wich distribution we will use
        if self.distribution == "left_right":
            if np.random.rand() > 0.5:
                angle = np.random.normal(3 * np.pi / 2, np.pi / 4)
            else:
                angle = np.random.normal(np.pi / 2, np.pi / 4)
        elif self.distribution == "uniform":
            angle = np.random.uniform(0, 2 * np.pi)
        elif self.distribution == "normal":
            angle = np.random.normal(self.mean, self.std)
        elif self._distribution == "adapted":
            # get minimal bucket to zero
            min_weight = min(self._buckets)
            buckets =  self._buckets + min_weight
            # normalize buckets
            buckets /= buckets.sum()
            # get reverse propability
            buckets = 1 - buckets
            # cut of all buckets that do not lay inside the limits
            # -> set the propability to zero
            theta = np.linspace(0, 2*np.pi, len(buckets) + 1)[:-1]
            logic1 = theta > self._upper_sampling_limit 
            logic2 = theta < self._lower_sampling_limit
            buckets[logic1 * logic2] = 0
            # normalize reversed propability
            bucket_sampling_weights = buckets / buckets.sum()
            bucket_idx = np.random.choice(list(range(len(self._buckets))), p=bucket_sampling_weights) 
            
            # get lower / upper limit for uniform sampling
            angels_per_bucket = 2 * np.pi / len(self._buckets)
            lower = bucket_idx * angels_per_bucket - angels_per_bucket  # is negaitve for idx = 0, take modulu 2pi after sampling
            higher = (bucket_idx + 1) * angels_per_bucket - angels_per_bucket
            
            # sample angel from bucket uniformly
            angle = np.random.uniform(lower, higher) % (2 * np.pi)    
        return angle
    
    def __update_buckets(self):
        """
        1. load data_json 
        2. if there is no json -> buckets = np.ones(8)
        3. get angels and reward_acc
        4. create buckets
        """
        # get latest eval_data_file
        max_iter = 0
        steps = 0
        eval_data_file = ""
        for name in glob.glob(self.get_common_data_path() + "/eval_data_*"):
            steps = int(name.split("_")[-2])
            if steps > max_iter:
                max_iter = steps
                eval_data_file = name
                
        if max_iter == 0:
            return np.ones(8)
                
        # load json
        file = open(eval_data_file)
        data = json.load(file)
        
        return create_buckets(36, data["Direction"]["angle"], data["reward_acc"])

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.

        Args:
        env: the environment who invokes this callback function.
        """
        self._env = env
        
        self._buckets = self.__update_buckets()
        
        # get sampled angle
        self._angle = self.__sample_angle()

        # create direction vector
        self.direction = np.array([np.cos(self._angle), np.sin(self._angle)])

        # multiply the normed direction vector with the speed
        if self.speed is not None:
            self.direction *= self.speed    

    def _get_observation(self) -> _ARRAY:
        """
        Returns where the robot should move to.
        """
        return self.direction
    
    @property
    def angle(self):
        return self._angle
    
    @property
    def buckets(self):
        return self._buckets
    
    @buckets.setter
    def buckets(self, value):
        self._buckets = value
        
        
class SpeedSensor(sensor.BoxSpaceSensor):
    """A sensor that reports the speed that the robot should move to."""

    def __init__(self,
                 target_speed: float = None,
                 lower_bound: _FLOAT_OR_ARRAY = -100.0,
                 upper_bound: _FLOAT_OR_ARRAY = 100.0,
                 name: typing.Text = "Speed",
                 dtype: typing.Type[typing.Any] = np.float64,
                 common_data_path: typing.Text = "") -> None:
        """Constructs SpeedsSensor.

        Args:
        lower_bound: the lower bound of the actions
        upper_bound: the upper bound of the actions
        name: the name of the sensor
        dtype: data type of sensor value
        """
        self._target_speed = target_speed
        self._env = None
        self._current_speed = 0
        
        super().__init__(name=name,
                        shape=(1,),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        dtype=dtype,
                        common_data_path=common_data_path)
    
    def __get_speed_diff(self):
        return self._current_speed - self._target_speed
    
    def on_reset(self, env):
        self._env = env
        self._current_speed = 0
    
    def on_step(self, env):
        # calcualte current robot speed
        self._current_speed = np.linalg.norm(env.robot.GetBaseVelocity())
    
    def get_observation(self) -> np.ndarray:
        self.__get_speed_diff()

