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

## Tuned Examples

### DQN CartPole

```shell script
python actor.py --env CartPole-v1 --alg dqn --model qmlp --ip 127.0.0.1 --num_steps 5000 --exploration_fraction 0.5 --max_steps_per_update 1 --num_replicas 4

python learner.py --env CartPole-v1 --alg dqn --model qmlp --training_start 100 --pool_size 5000 --update_freq 50 --num_steps 20000 --lr 0.0005 --training_freq 1 --batch_size 32
```

### DQN Pong

```shell script
python actor.py --env PongNoFrameskip-v4 --alg dqn --model qcnn --ip 127.0.0.1 --num_steps 200000 --exploration_fraction 0.1 --max_steps_per_update 1 --num_replicas 5

python learner.py --env PongNoFrameskip-v4 --alg dqn --model qcnn --training_start 5000 --pool_size 5000 --update_freq 1000 --num_steps 1000000 --lr 0.00005 --training_freq 1 --batch_size 32
```

### PPO CartPole

```shell script
python actor.py --env CartPole-v1 --alg ppo --model acmlp --ip 127.0.0.1 --num_steps 200000 --max_steps_per_update 4000 --gamma 0.99 --lam 0.97

python learner.py --env CartPole-v1 --alg ppo --model acmlp --num_steps 200000 --pool_size 4000 --training_freq 1 --batch_size 4000
```

### PPO Pong

```shell script
python actor.py --env PongNoFrameskip-v4 --alg ppo --model accnn --ip 127.0.0.1 --num_steps 10000000 --max_steps_per_update 1000 --max_episode_length 0 --gamma 0.99 --lam 0.97

python learner.py --env PongNoFrameskip-v4 --alg ppo --model accnn --num_steps 10000000 --pool_size 1000 --training_freq 1 --epochs 10 --batch_size 1000
```