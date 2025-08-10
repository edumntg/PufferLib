'''A minimal template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.missile_launch import binding

class MissileLaunch(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=5, buf=None, seed=0):
        high = np.array([1000, 1000, 200, 200, np.pi, 50], dtype=np.float32)
        self.single_observation_space = gymnasium.spaces.Box(low=-high, high=high, shape=(6,), dtype=np.float32)

        low = np.array([-1.0, 0.0])
        high = np.array([1.0, 1.0])
        self.single_action_space = gymnasium.spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
                                       self.terminals, self.truncations, num_envs, seed, size=size)
        self.size = size

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)