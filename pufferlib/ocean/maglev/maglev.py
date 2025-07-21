
import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.maglev import binding

class MagLev(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=11, buf=None, seed=0):
        low = np.array([0.0, -10.0])
        high = np.array([2.0, 10.0])
        self.single_observation_space = gymnasium.spaces.Box(low=low, high=high,
                                                             shape=(2,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(low = 0.0, high = 6.0, shape = (1,), dtype = np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
                                       self.terminals, self.truncations, num_envs, seed, size=size)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        # self.actions[:] = actions
        self.actions[:] = 3.10
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

# if __name__ == '__main__':
#     """Benchmark environment performance."""
#     num_envs = 4096
#     env = MagLev(num_envs=num_envs)
#     env.reset()
#     tick = 0
#
#     atn_cache = 8192
#     actions = np.random.randint(3.10, 3.10, (atn_cache, 1))
#
#     import time
#     timeout = 20
#     start = time.time()
#     while time.time() - start < timeout:
#         atn = actions[tick % atn_cache]
#         env.step(atn)
#         tick += 1
#     sps = num_envs * tick / (time.time() - start)
#     print(f'SPS: {sps:,}')
