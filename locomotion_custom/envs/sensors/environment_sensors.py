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

from scipy.spatial.transform import Rotation

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
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
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

        super(LastActionSensor, self).__init__(name=name,
                                               shape=(self._num_actions,),
                                               lower_bound=lower_bound,
                                               upper_bound=upper_bound,
                                               dtype=dtype)

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.

    Args:
      env: the environment who invokes this callback function.
    """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the last action of the environment."""
        return self._env.last_action


class CameraArray(sensor.BoxSpaceSensor):
    """sensor that represents the front cameras"""

    def __init__(self,
                 lower_bound: _FLOAT_OR_ARRAY = 0,
                 upper_bound: _FLOAT_OR_ARRAY = 255,
                 name: typing.Text = "CameraArray",
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
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
        self.render_height = 360  # TODO: into config, check for memory consumption on cluster
        self.render_with = 480  # TODO: into config

        # Warning is caused by typing from Box Space Sensor shape
        super(CameraArray, self).__init__(name=name,
                                          shape=(self.render_height, self.render_with, 5),
                                          lower_bound=lower_bound,
                                          upper_bound=upper_bound,
                                          dtype=dtype)

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
                 lower_bound: _FLOAT_OR_ARRAY = -1.0,
                 upper_bound: _FLOAT_OR_ARRAY = 1.0,
                 name: typing.Text = "Direction",
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
        """Constructs LastActionSensor.

        Args:
        lower_bound: the lower bound of the actions
        upper_bound: the upper bound of the actions
        name: the name of the sensor
        dtype: data type of sensor value
        """
        self.direction = np.zeros(2)
        self._env = None

        super().__init__(name=name,
                        shape=(2,),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        dtype=dtype)

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.

        Args:
        env: the environment who invokes this callback function.
        """
        self._env = env

        if np.random.rand() > 0.5:
            angle = np.random.normal(3 * np.pi / 2, np.pi / 4)
        else:
            angle = np.random.normal(np.pi / 2, np.pi / 4)

        self.direction = np.array([np.cos(angle), np.sin(angle)])

    def _get_observation(self) -> _ARRAY:
        """Returns where the robot should move to."""
        return self.direction