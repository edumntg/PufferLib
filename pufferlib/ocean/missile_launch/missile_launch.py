'''A minimal template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.missile_launch import binding

class MissileLaunch(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, num_missiles=5, buf=None, seed=0):
        high = np.array([1000, 1000, 200, 200, np.pi, 50], dtype=np.float32)
        self.single_observation_space = gymnasium.spaces.Box(low=-high, high=high, shape=(6,), dtype=np.float32)

        # Actions: [thrust (0..1), gimbal (-1..1)]
        low = np.array([0.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs * num_missiles

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[i*num_missiles:(i+1)*num_missiles],
                self.actions[i*num_missiles:(i+1)*num_missiles],
                self.rewards[i*num_missiles:(i+1)*num_missiles],
                self.terminals[i*num_missiles:(i+1)*num_missiles],
                self.truncations[i*num_missiles:(i+1)*num_missiles],
                i,
                num_agents=num_missiles,
            ))

        self.c_envs = binding.vectorize(*c_envs)

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