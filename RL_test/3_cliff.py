from gridworld import GridWorldEnv
from gym import spaces

env = GridWorldEnv(n_width = 12,
                   n_height=4,
                   u_size=60,
                   default_reward = -1,
                   default_type = 0,
                   windy = False)

env.action_space = spaces.Discrete(4)

env.start = (0,0)
env.ends = [(11,0)]

for i in range(10):
    env.rewards.append((i+1,0,-100))
    env.ends.append((i+1,0))

env.types = [(5,1,1),(5,2,1)]

env.refresh_setting()

env.reset()

env.render()
input("press any key to continue...")
