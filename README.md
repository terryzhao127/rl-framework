# RL Framework

A distributed reinforcement learning framework

## Getting Started

```shell script
export HOROVOD_GPU_OPERATIONS=NCCL

conda env create -f env.yml

# Run tuned examples saved by YAML files
python actor.py --config examples/ppo/cartpole_actor.yaml
python learner.py --config examples/ppo/cartpole_learner.yaml

# Run with CLI
python actor.py --env CartPole-v1 --alg dqn --model qmlp --ip localhost --num_steps 5000 --exploration_fraction 0.5 --max_steps_per_update 1 --num_replicas 4
python learner.py --env CartPole-v1 --alg dqn --model qmlp --training_start 100 --pool_size 5000 --update_freq 50 --num_steps 20000 --lr 0.0005 --training_freq 1 --batch_size 32
```
