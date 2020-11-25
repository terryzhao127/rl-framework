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


### DQN CartPole
```
python actor.py --env CartPole-v1 --alg dqn --model mlp --ip 127.0.0.1 --num_steps 20000 --exploration_fraction 0.5
python learner.py --env CartPole-v1 --alg dqn --model mlp --training_start 100 --epsilon_min 0.02 --buffer_size 5000 --update_freq 50 --num_steps 20000 --lr 0.0005 --exploration_fraction 0.5

```