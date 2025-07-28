'''A minimal template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.ball3d import binding

class Ball3D(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, buf=None, seed=0, num_balls = 2):
        self.single_observation_space = gymnasium.spaces.Box(low=-20.0, high=20.0,
            shape=(3,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(low = -1.0, high = 1.0, shape = (3,), dtype = np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs*num_balls

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[i*num_balls:(i+1)*num_balls],
                self.actions[i*num_balls:(i+1)*num_balls],
                self.rewards[i*num_balls:(i+1)*num_balls],
                self.terminals[i*num_balls:(i+1)*num_balls],
                self.truncations[i*num_balls:(i+1)*num_balls],
                i,
                num_agents=num_balls,
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
#
# if __name__ == '__main__':
#     N = 4096
#     env = Ball3D(num_envs=N)
#     env.reset()
#     steps = 0
#
#     CACHE = 1024
#     actions = np.random.randn(0, 5, (CACHE, N))
#
#     import time
#     start = time.time()
#     while time.time() - start < 10:
#         env.step(actions[steps % CACHE])
#         steps += 1
#
#     print('Squared SPS:', int(env.num_agents*steps / (time.time() - start)))
