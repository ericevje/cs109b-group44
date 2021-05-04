from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataclasses import dataclass
from enum import IntEnum
import uuid 
import sys
import os

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

##############################
##### Fusion gym imports #####
##############################

# Add the client folder to sys.path
CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "fusion360gym", "client")
if CLIENT_DIR not in sys.path:
    sys.path.append(CLIENT_DIR)

from gym_env import GymEnv
from fusion_gym import fusion_env_wrapper

from model.ExtrudeAction import ExtrudeAction

    
@dataclass(frozen=True)
class State:
    step: int
    finished: bool


class FusionGymEnvironment(py_environment.PyEnvironment):
    """ FusionGymEnvironment implements the PyEnvironment interface from tensorflow """
        
    def __init__(self, gym_env: fusion_env_wrapper.FusionEnvWrapper, max_steps):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(5,), dtype=np.int64, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = State(0, False)
        self._episode_ended = False
                
        self._env = gym_env
        self._max_steps = max_steps
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = State(0, False)
        self._env.revert_to_target()
        return ts.restart(np.array([self._state.step], dtype=np.int32))

    def _step(self, action):
        self._state = State(self._state.step + 1, self._state.step >= self._max_steps)
        print(f"step: {self._state}")
        
        # Apply action
        try:
            decoded_action = ExtrudeAction.decode(action)
            print(decoded_action)
            graph, iou = self._env.extrude(
                decoded_action.start_face,
                decoded_action.end_face,
                decoded_action.operation.to_string()
            )
            print(graph)
        except ValueError:
            #print("Could not decode action")
            pass


        if self._state.finished:
            return ts.termination(np.array([self._state.step], dtype=np.int32), 0.0)
        else:
            return ts.transition(
              np.array([self._state.step], dtype=np.int32), reward=0.0, discount=1.0)




    