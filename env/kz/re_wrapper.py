from airsim.sadri_env.simulator_wrapper import AirSimEnv
from airsim.baseline.baseline_policy import BaselinePolicy
from airsim.utils import config

import numpy as np
import os
import glob
import shutil


class ReWrappedEnv(AirSimEnv):
  def __init__(
    self, mode='rl', blue_baseline_mode=0, red_baseline_mode=None,
    remove_data=False, fire_only=False, **kwargs
  ):
    valid_baseline = blue_baseline_mode is not None or red_baseline_mode is not None
    assert mode == 'selfplay' or valid_baseline
    super().__init__(
      mode=mode, blue_baseline_mode=blue_baseline_mode, red_baseline_mode=red_baseline_mode,
      **kwargs)
    self.del_baseline()

    self.remove_data = remove_data
    self.baseline_modes = [0, 1, 2, 3, 4]
    self.fire_only = fire_only
    if self.fire_only:
      print('*' * 50, 'Fire Only')


  def __del__(self):
    self.close()

  def del_baseline(self):
    try:
      self.red_baseline_policy.mcts.process.kill()
      del self.red_baseline_policy
    except:
      pass
    try:
      self.blue_baseline_policy.mcts.process.kill()
      del self.blue_baseline_policy
    except:
      pass

  def restart(self, **kwargs):
    self.del_baseline()
    super().restart(**kwargs)

  def reset(self, blue_baseline_mode=None, red_baseline_mode=None):
    self.del_baseline()
    if blue_baseline_mode is not None or red_baseline_mode is not None:
      self.blue_baseline_mode = blue_baseline_mode
      self.red_baseline_mode = red_baseline_mode
    state, _, _ = super().reset()
    self.del_baseline()
    if self.blue_baseline_mode is not None:
      if self.blue_baseline_mode == 'random':
        mode = np.random.choice(self.baseline_modes)
      else:
        mode = self.blue_baseline_mode
      self.blue_baseline_policy = BaselinePolicy(
        plane_id=self.blue_plane_config['id'], mode=mode)
      plane_id = self.blue_plane_config['id']
      self.agent_baseline_policy = BaselinePolicy(
        plane_id=self.red_plane_config['id'], mode=0)
      print(f'\nUsing bluebaseline: {mode}, planeID: {plane_id}\n')
    if self.red_baseline_mode is not None:
      if self.red_baseline_mode == 'random':
        mode = np.random.choice(self.baseline_modes)
      else:
        mode = self.red_baseline_mode
      self.red_baseline_policy = BaselinePolicy(
        plane_id=self.red_plane_config['id'], mode=mode)
      plane_id = self.red_plane_config['id']
      self.agent_baseline_policy = BaselinePolicy(
        plane_id=self.blue_plane_config['id'], mode=0)
      print(f'\nUsing redbaseline: {mode}, planeID: {plane_id}\n')
    self.agent_baseline_policy.mcts.process.kill()

    self.state = state
    return state

  def step(self, plane_action):
    if self.closed:
      raise RuntimeError('The simulator has been closed')

    self.num_steps += 1
    for _ in range(self.frame_stack_k):
      self.num_unit_steps += self.simulator_config['step_repeat_k']
      with self.step_time_stat:
        ret_state, ret_reward, ret_done, info = self._step(plane_action)
      if ret_done:
        break
      if self.mode == 'rl':
        plane_action.bLaunch = False
      else:
        plane_action[0].bLaunch = False
        plane_action[1].bLaunch = False
    return ret_state, ret_reward, ret_done, info

  def _step(self, plane_action):
    if self.fire_only:
      baseline_action = self.agent_baseline_policy.get_action(self.state)
      baseline_action.bLaunch = plane_action.bLaunch
    else:
      baseline_action = plane_action
    ret_state, ret_reward, ret_done = super()._step(baseline_action)
    if ret_done and self.remove_data:
      self.clean_data()
    if self.mode == 'rl':
      if self.blue_baseline_mode is not None:
        info = {
          'AgentScore': self.red_plane_state.fAggregateScore,
          'BaselineScore': self.blue_plane_state.fAggregateScore,
        }
      else:
        info = {
          'AgentScore': self.blue_plane_state.fAggregateScore,
          'BaselineScore': self.red_plane_state.fAggregateScore,
        }
    else:
      info = {
        'BlueScore': self.blue_plane_state.fAggregateScore,
        'RedScore': self.red_plane_state.fAggregateScore,
      }
    if ret_done and self.mode == 'rl':
      if self.blue_baseline_mode is not None:
        info['AgentEndReason'] =  config.over_info[self.red_plane_state.nEndReason]
        info['BaselineEndReason'] =  config.over_info[self.blue_plane_state.nEndReason]
      else:
        info['AgentEndReason'] =  config.over_info[self.blue_plane_state.nEndReason]
        info['BaselineEndReason'] =  config.over_info[self.red_plane_state.nEndReason]
    ret_reward = info['AgentScore'] / 2
    if ret_reward < -1:
        ret_reward = -10
    return ret_state, ret_reward, ret_done, info

  def clean_data(self):
    home_path = os.getenv("HOME")
    plugin_exe_path = f"{home_path}/plugin_exe"

    data_path = f"{plugin_exe_path}/package/data"
    log_path = f"{plugin_exe_path}/package/log"

    data_folders = glob.glob(f"{data_path}/data*")
    for folder in data_folders:
      shutil.rmtree(folder)

    log_folders = glob.glob(f"{log_path}/Log*")
    for folder in log_folders:
      shutil.rmtree(folder)

    try:
      core_file = f"{plugin_exe_path}/package/core"
      os.remove(core_file)
    except:
      pass

