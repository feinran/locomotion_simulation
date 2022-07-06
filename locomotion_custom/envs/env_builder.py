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
"""Utilities for building environments."""
import sacred.utils
from helper import generate_names

from locomotion_simulation.locomotion_custom.envs import locomotion_gym_env
from locomotion_simulation.locomotion_custom.envs import locomotion_gym_config
from locomotion_simulation.locomotion_custom.envs import sensors
from locomotion_simulation.locomotion_custom.envs.env_wrappers import \
    observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from locomotion_simulation.locomotion_custom.envs.env_wrappers import \
    trajectory_generator_wrapper_env
from locomotion_simulation.locomotion_custom.envs.env_wrappers import simple_openloop
from locomotion_simulation.locomotion_custom.envs.sensors import robot_sensors
from locomotion_simulation.locomotion_custom.robots import a1
from locomotion_simulation.locomotion_custom.robots import laikago
from locomotion_simulation.locomotion_custom.robots import robot_config


from locomotion_simulation.locomotion_custom.envs.sensors import environment_sensors
from locomotion_simulation.locomotion_custom.envs import custom_tasks
from inspect import getmembers, isclass


def build_regular_env(robot_class,
                      motor_control_mode,
                      enable_rendering=False,
                      on_rack=False,
                      action_limit=(0.75, 0.75, 0.75),
                      wrap_trajectory_generator=True,
                      config={}):
    
    env_config = config["env"]
    _, results_dir, _= generate_names(config["results_dir"],
                                   config["checkpoint"],
                                   config["base_name"],
                                   config["algo_type"])
    
    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    sim_params.motor_control_mode = motor_control_mode
    sim_params.reset_time = 2
    sim_params.sim_time_step_s = env_config['sim_time_step_s']
    sim_params.num_action_repeat = env_config['action_repeat']
    sim_params.enable_action_interpolation = env_config['action_interpolation']
    sim_params.enable_action_filter = env_config['action_filter']
    sim_params.enable_clip_motor_commands = env_config['action_clip']
    sim_params.robot_on_rack = on_rack

    gym_config = locomotion_gym_config.LocomotionGymConfig(
        simulation_parameters=sim_params)

    # sensors = [
    #     robot_sensors.BaseDisplacementSensor(),
    #     robot_sensors.IMUSensor(),
    #     robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS),
    # ]
    sensors = []
    sensor_classes = dict(getmembers(robot_sensors, isclass))
    for sensor, parameters in env_config['robot_sensors'].items():
        parameters = parameters.copy()
        if parameters.pop('enabled'):
            sensor_class = sensor_classes[sensor]
            sensor = sensor_class(**parameters)
            sensors.append(sensor)

    env_sensors = []
    sensor_classes = dict(getmembers(environment_sensors, isclass))
    for sensor, parameters in env_config['env_sensors'].items():
        parameters = parameters.copy()
        # add common data storage
        parameters["common_data_path"] = results_dir
        if parameters.pop('enabled'):
            sensor_class = sensor_classes[sensor]
            sensor = sensor_class(**parameters)
            env_sensors.append(sensor)

    task = None
    if type(env_config['task']) == str:
        task = dict(getmembers(custom_tasks, isclass))[env_config['task']]()
    else:
        task_classes = dict(getmembers(custom_tasks, isclass))
        for task_name, parameters in env_config['task'].items():
            parameters = parameters.copy()
            if parameters.pop('enabled'):
                if task is not None:
                    raise Exception('More than one task is enabled!')

                task_class = task_classes[task_name]
                task = task_class(**parameters)

    if task is None:
        raise Exception('No task is enabled!')
                    

    env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                              robot_class=robot_class,
                                              env_sensors=env_sensors,
                                              robot_sensors=sensors,
                                              task=task,
                                              config=env_config)

    env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(
        env)
    if (motor_control_mode
        == robot_config.MotorControlMode.POSITION) and wrap_trajectory_generator:
        if robot_class == laikago.Laikago:
            env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
                env,
                trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
                    action_limit=action_limit))
        elif robot_class == a1.A1:
            env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
                env,
                trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
                    action_limit=action_limit))
    return env
