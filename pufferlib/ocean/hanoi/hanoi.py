import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.hanoi import binding

class Hanoi(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, num_disks = 3, num_pegs = 3, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=num_disks - 1,
                                                             shape=(num_pegs * num_disks,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([num_pegs, num_pegs])
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
                                       self.terminals, self.truncations, num_envs, seed, num_disks = num_disks, num_pegs = num_pegs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
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


def test_performance(timeout=10, atn_cache=1024):
    env = Hanoi(num_envs=1000)
    env.reset()
    tick = 0

    actions = [env.action_space.sample() for _ in range(atn_cache)]

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {env.num_agents * tick / (time.time() - start)}")

if __name__ == "__main__":
    test_performance()