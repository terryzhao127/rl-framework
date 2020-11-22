# RL Framework

A distributed reinforcement learning framework

## Getting Started

```shell script
export HOROVOD_GPU_OPERATIONS=NCCL

conda install -f env.yml

# Actor
python actor.py --env PongNoFrameskip-v4 --num_steps 1e7 --alg dqn --history_length 4

# Learner
python learner.py --env PongNoFrameskip-v4 --num_steps 1e7 --alg dqn
```
