import d3rlpy

# 대표적인 알고리즘들
from d3rlpy.algos import (
    DQN, DoubleDQN,
    DDPG, TD3, SAC,
    BC, BCQ, BEAR,
    CQL, AWAC
)

print("D3RLPY 0.x 알고리즘들:")
print([cls.__name__ for cls in [
    DQN, DoubleDQN, DDPG, TD3, SAC,
    BC, BCQ, BEAR, CQL, AWAC
]])
