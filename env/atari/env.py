from core.env import Env

from .atari_wrappers import make_env


class AtariEnv(Env):
    def __init__(self, gym_env_id, history_length=4, *args, **kwargs):
        super(AtariEnv, self).__init__(*args, **kwargs)
        self.env_wrapper = make_env(gym_env_id, history_length)

    def reset(self):
        return self.env_wrapper.reset()

    def step(self, action, *args, **kwargs):
        return self.env_wrapper.step(action)

    def get_action_space(self):
        return self.env_wrapper.action_space.n

    def get_observation_space(self):
        return self.env_wrapper.observation_space.shape

    def calc_reward(self, *args, **kwargs):
        raise NotImplemented

    def render(self) -> None:
        self.env_wrapper.render()
