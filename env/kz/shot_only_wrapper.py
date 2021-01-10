import gym
import copy
import numpy as np
import signal
import os
import glob
import shutil

from airsim.sadri_env.action_def import PlaneActionModel
from env.kz.re_wrapper import ReWrappedEnv
from env.kz.state_action_processor import StateActionProcessor


class FireOnlyEnv(ReWrappedEnv):
  def __init__(
    self,
    mode='rl',
    autho_ip='192.168.111.201',
    **args
  ):
    self.fire_cumulated = 0
    self.delta = 0.5
    self.args = args
    self.args['fire_only'] = True
    self.mode = mode
    self.autho_ip = autho_ip
    self.state_action_processor = StateActionProcessor()
    self.obs_dim = self.state_action_processor.obs_dim
    super().__init__(
        mode=self.mode, auth_server_ip=autho_ip, **self.args)

    self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
    self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    signal.signal(signal.SIGALRM, handler=lambda signum, frame: exec('raise(RuntimeError("Timeout Error"))'))
    self.state = None

  def get_action_space():
    return self.action_space

  def get_observation_space():
    return self.observation_space
    
  def reset(self, blue_mode=None, red_mode=None):
    o = super().reset(blue_baseline_mode=blue_mode, red_baseline_mode=red_mode)
    self.state = o
    self.total_fires = 0
    return self.state_action_processor.process_state(o)

  def __del__(self):
    self.close()

  def step(self, action):
    plane_action = self._process_action(action)
    signal.alarm(5)
    try:
      o, r, d, info = super().step(plane_action)
    except Exception as e:
      print(e)
      signal.alarm(0)
      raise e
    signal.alarm(0)
    self.state = o
    o = self.state_action_processor.process_state(o)
    return o, r, d, info

  def restart(self):
    self.num_restart_times += 1
    super().__init__(mode=self.mode, auth_server_ip=self.autho_ip, **self.args)

  def _process_action(self, action):
    example_action = PlaneActionModel()
    plane_action = copy.deepcopy(example_action)

    fire = action[0] / 2. + 0.5 + 1e-5
    self.fire_cumulated = np.max([0, np.log(fire / 0.55)])
    if self.fire_cumulated > self.delta:
      fire = True
      self.fire_cumulated = 0
    else:
      fire = False
    plane_action.bLaunch = fire

    return plane_action

  def remove_data(self):
    home_path = os.getenv('HOME')
    plugin_exe_path = f'{home_path}/plugin_exe'

    data_path = f'{plugin_exe_path}/package/data'
    log_path = f'{plugin_exe_path}/package/log'
    data_folders = glob.glob(f'{data_path}/data*')
    for folder in data_folders:
      shutil.rmtree(folder)

    log_folders = glob.glob(f'{log_path}/Log*')
    for folder in log_folders:
      shutil.rmtree(folder)

    try:
      core_file = f'{plugin_exe_path}/package/core'
      os.remove(core_file)
    except:
      pass
