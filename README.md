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
python actor.py --env CartPole-v1 --alg dqn --model mlp --ip 127.0.0.1 --num_steps 20000 --exploration_fraction 0.5

python learner.py --env CartPole-v1 --alg dqn --model mlp --training_start 100 --buffer_size 5000 --update_freq 50 --num_steps 20000 --lr 0.0005
```

### DQN Pong

```shell script
python actor.py --env PongNoFrameskip-v4 --alg dqn --model cnn --ip 127.0.0.1 --num_steps 1000000 --exploration_fraction 0.1

python learner.py --env PongNoFrameskip-v4 --alg dqn --model cnn --training_start 5000 --buffer_size 5000 --update_freq 1000 --num_steps 1000000 --lr 0.00005
```

### PPO CartPole

```shell script
python actor.py --env CartPole-v1 --alg ppo --model acmlp --ip 127.0.0.1 --num_steps 200000 --max_steps_per_update 4000 --gamma 0.99 --lam 0.97

python --env CartPole-v1 --alg ppo --model acmlp --num_steps 200000 --pool_length 1 --training_freq 1 --batch_size 4000
```

### PPO Pong

```shell script
python actor.py --env PongNoFrameskip-v4 --alg ppo --model accnn --ip 127.0.0.1 --num_steps 3000000 --n_step 100

python learner.py --env PongNoFrameskip-v4 --alg ppo --model accnn --num_steps 3000000 --epochs 1
```