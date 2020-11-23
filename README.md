# RL Framework

A distributed reinforcement learning framework

## Getting Started

```shell script
export HOROVOD_GPU_OPERATIONS=NCCL

conda env create -f env.yml

# Actor
python actor.py --env PongNoFrameskip-v4 --num_steps 1e7 --alg dqn --ip 127.0.0.1 --num_replicas 4

# Learner
python learner.py --env PongNoFrameskip-v4 --num_steps 1e7 --alg dqn
```
