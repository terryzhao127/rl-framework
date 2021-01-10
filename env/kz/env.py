from env.kz.shot_only_wrapper import FireOnlyEnv



class KZEnv(FireOnlyEnv):
    def __init__(self, mode='rl', autho_ip='192.168.111.201', **args):
        self.mode = mode
        self.autho_ip = autho_ip
        self.args = args
        super(FireOnlyEnv, self).__init__(self.mode, self.autho_ip, **self.args)

    def reset(self, blue_mode=None, red_mode=None):
        return super(FireOnlyEnv, self).reset(blue_mode=blue_mode, red_mode=red_mode)

    def step(self, action, *args, **kwargs):
        return super(FireOnlyEnv, self).step(action=action)

    def get_action_space(self):
        return super(FireOnlyEnv, self).get_action_space()

    def get_observation_space(self):
        return super(FireOnlyEnv, self).get_observation_space()

    def calc_reward(self, *args, **kwargs):
        raise NotImplemented

    def render(self) -> None:
        raise NotImplemented
