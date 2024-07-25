import cv2
import torch
import numpy as np
from collections import deque
from typing import Any, NamedTuple
from dm_env import StepType, specs

import sys
sys.path.append('/workspace/S/heguanhua2/robot_rl/robosuite_jimu')

import robosuite as suite
import robosuite.macros as macros
from robosuite.wrappers import Wrapper
from robosuite.controllers import load_controller_config

# IMAGE_CONVENTION default is "opengl", the img is "right side up"
macros.IMAGE_CONVENTION = "opencv"


# Modify the step() output, extend action
class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    time_limit_reached: Any

    def fisrt(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class RoboWrapper(Wrapper):
###############################################################
# 对原始robosuite仿真环境做包装
# 对step,action需要重复action_repeat次
# 对step/reset,得到环境状态需要stack frame_stack次
# 增加get_obs函数 扩充原来state的内容
###############################################################
    def __init__(self, env, action_repeat, frame_stack, device,\
            width=84, height=84):
        self._env = env
        self._action_repeat = action_repeat
        self._frame_stack = frame_stack
        self._device = device
        self._frames = deque([], maxlen=self._frame_stack)
        self._width = width
        self._height = height


    def _get_obs_imgs(self):
    #########################################
    # get env current observation camera imgs
    #########################################
        imgs = []
        obs = self._env.viewer._get_observations()\
                if self._env.viewer_get_obs\
                else self._env._get_observations()
        for cam in self._env.camera_names:
            img = obs[cam + "_image"]
            img = cv2.resize(img, (self._width, self._height))
            img = img.transpose((2, 0, 1)) # convert (h,w,3) to (3,h,w)
            imgs.append(img)
        pixels = np.concatenate(imgs, axis=0)
        return pixels


    def _get_stacked_pixels(self):
        assert len(self._frames) == self._frame_stack
        stacked_pixels = np.concatenate(list(self._frames), axis=0)
        return stacked_pixels


    def reset(self):
        obs = self._env.reset()
        pixels = self._get_obs_imgs()
        for _ in range(self._frame_stack):
            self._frames.append(pixels)
        stacked_pixels = self._get_stacked_pixels()
        return stacked_pixels


    def step(self, action):
        reward_sum = 0.0
        discount_prod = 1.0
        for _ in range(self._action_repeat):
            _, reward, done, info = self._env.step(action)
            reward_sum += reward
            if done:
                break
        pixels = self._get_obs_imgs()
        self._frames.append(pixels)
        stacked_pixels = self._get_stacked_pixels()
        return stacked_pixels, reward_sum, done, info


    def get_pixels_with_width_height(self, w, h):
        imgs = []
        obs = self._env.viewer._get_observations() \
               if self._env.viewer_get_obs \
               else self._env._get_observations()
        for cam in self._env.camera_names:
            img = obs[cam + "_image"]
            # img = img.transpose((2, 0, 1))
            imgs.append(img)
        pixels = np.concatenate(imgs, axis=0)
        return pixels


class RoboEnv:
###############################################
#
###############################################
    def __init__(self, env_name, action_repeat=2, frame_stack=3,
            device=None, seed=None, reward_rescale=None):
        #TODO: reward rescale
        assert reward_rescale is None
        reward_rescale_dict = {}
        self.reward_rescale = 1 \
                if reward_rescale is None\
                else reward_rescale_dict[env_name]

        # init the Env
        controller_config = load_controller_config(default_controller="OSC_POSE")
        env = suite.make(env_name=env_name,
                         robots='UR5e',
                         controller_configs=controller_config,
                         control_freq=160,
                         horizon=1000,
                         use_object_obs=True,
                         use_camera_obs=True,
                         camera_names="frontview",
                         camera_heights=256,
                         camera_widths=256,
                         reward_shaping=True)
        action_dim = env.action_dim
        print(f'action_dim={action_dim}')
        _ = env.reset()

        env = RoboWrapper(env,
                          action_repeat=action_repeat,
                          frame_stack=frame_stack,
                          device=device)
        self._env = env

        #TODO: set obs & action spec in BoundedArray
        num_channel = 3 * frame_stack
        self._obs_spec = specs.BoundedArray(shape=(num_channel, 84, 84),
                                            dtype='uint8',
                                            name='observation',
                                            minimum=0,
                                            maximum=255)
        self._action_spec = specs.BoundedArray(shape=(action_dim,),
                                               dtype='float32',
                                               name='action',
                                               minimum=-1.0,
                                               maximum=1.0)


    def reset(self):
        observation = self._env.reset()
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step = ExtendedTimeStep(observation=observation,
                                     step_type=StepType.FIRST,
                                     action=action,
                                     reward=0.0,
                                     discount=1.0,
                                     time_limit_reached=False)
        return time_step


    def step(self, action):
        observation, reward, done, env_info = self._env.step(action)
        discount = 1.0
        steptype = StepType.LAST if done else StepType.MID
        reward = reward * self.reward_rescale
        #TODO: robosuite env_info dont have "TimeLimit"
        time_limit_reached = env_info['TimeLimit.truncated'] if 'TimeLimit.truncated' in env_info else False

        time_step = ExtendedTimeStep(observation=observation,
                                     step_type=steptype,
                                     action=action,
                                     reward=reward,
                                     discount=discount,
                                     time_limit_reached=time_limit_reached)
        return time_step


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._obs_spec

    def set_env_state(self):
        pass

    def get_pixels_with_width_height(self, w, h):
        return self._env.get_pixels_with_width_height(w, h)


def make(name, frame_stack, action_repeat, seed, device=torch.device('cuda')):
    # env_options = dict(env_name=name,
    #                robots='Panda')
    # env = suite.make(**env_options)
    env = RoboEnv(env_name=name,
                  frame_stack=frame_stack,
                  action_repeat=action_repeat,
                  seed=seed,
                  device=device,)
    return env

