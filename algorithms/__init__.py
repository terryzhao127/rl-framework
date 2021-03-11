from core.registry import Registry


ALGORITHM = Registry('ALGORITHM')


from algorithms.dqn import *
from algorithms.ppo import *
