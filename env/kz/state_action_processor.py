import numpy as np
import copy


class StateActionProcessor:
  def __init__(self):
    self.obs_dim = None
    self.dis_dim = None
    self.con_dim = None
    self.fixed_states = [
      'nAAMMissileNum',
      'dLongtitude_rad',
      'dLatitude_rad',
      'fAltitude_m',
      'fHeading_rad',
      'fPitch_rad',
      'fRoll_rad',
      'fVn_ms',
      'fVu_ms',
      'fVe_ms',
      'fAccN_ms2',
      'fAccU_ms2',
      'fAccE_ms2',
      'fRealAirspeed_kmh',
      'fNormalAcc_g',
    ]
    self.fixed_states_mean = np.array([
      0.5,  # missile0
      0.5,  # missile1
      0.5,  # missile2
      0.5,  # missile3
      0.5,  # missile4
      115 * np.pi / 180,  # longtitude
      30.8 * np.pi / 180,  # latitude
      6000,  # altitude
      0,  # sin(heading rad)
      0,  # cos(heading rad)
      0,  # pitch rad
      0,  # sin(roll rad)
      0,  # cos(roll rad)
      0,  # fv and facc
      0,
      0,
      0,
      0,
      0,
      1200,  # realairspeed
      4,  # fnormalacc_g
    ])
    self.fixed_states_std = np.array([
      0.5,  # missle0
      0.5,  # missle1
      0.5,  # missle2
      0.5,  # missle3
      0.5,  # missle4
      0.6 * np.pi / 180,  # longtitude
      1 * np.pi / 180,  # latutude
      4000,  # altitude
      1,  # sin cos
      1,
      np.pi / 2,  # pitch rad
      1,  # sin cos
      1,
      300,
      300,
      300,
      50,
      50,
      50,
      300,
      4,
    ])
    self.aaTarget_states = [
      'bValid',
      'fTargetDistanceDF_m',
      'fDisAlterRateDF_ms',
      'fTargetAzimDF_rad',
      'fTargetPitchDF_rad',
      'fTargetAzimVelGDF_rads',
      'fTargetPitchVelGDF_rads',
      'fVnDF_ms',
      'fVuDF_ms',
      'fVeDF_ms',
      'fAccNDF_ms2',
      'fAccUDF_ms2',
      'fAccEDF_ms2',
      'fEntranceAngleDF_rad',
      'fTargetAltDF_m',
      'fMachDF_M',
    ]
    self.aaTarget_states_mean = np.array([
      0.5,  # bValid
      50000,  # distance
      0,  # disalterRate
      0,  # sin Azim
      0,  # cos Azim
      0,  # pitch
      0,  # azim vel
      0,  # pitch vel
      0,  # fv and facc
      0,
      0,
      0,
      0,
      0,
      0,  # sin EntranceAngle
      0,  # cos EntranceAngle
      6000,  # altitude
      2,  # fmach
    ])
    self.aaTarget_states_std = np.array([
      0.5,  # bValid
      50000,  # distance
      700,  # alterRate
      1,  # sin azim
      1,  # cos azim
      np.pi / 2,  # pitch
      np.pi / 8,  # azim vel
      np.pi / 8,  # pitch vel
      300,
      300,
      300,
      50,
      50,
      50,
      1,  # sin EntranceAngle
      1,  # cos EntranceAngle
      4000,  # altitude
      2,  # fmach
    ])
    self.alarm_states = [
      'AlarmInfo_bValid',
      'fMisAzi',
    ]
    self.alarm_states_mean = np.array([
      0.5,
      0,  # sin
      0,  # cos
    ])
    self.alarm_states_std = np.array([
      0.5,
      1,
      1,
    ])

    self.actions = [
      'iCmdID',
      'fCmdSpd',
      'fCmdNy',
      'fCmdThrust',
      'fCmdPitchDeg',
      'fCmdRollDeg',
      'fCmdHeadingDeg',
    ]
    self.actions_mean = np.array([
      0,  #fcmdId
      1,  # fCmdSpd
      4,  # fCmdNy
      60,  # Thrust
      0,  # pitchDeg
      0,  # sin Roll
      0,  # cos Roll
      0,  # sin Heading
      0,  # cos Heading
    ])
    self.actions_std = np.array([
      1,  # fcmdId
      1,  # fcmdSpd
      4,  # fcmNy
      60,  # Thrust
      180,  # pitchDeg
      1,
      1,
      1,
      1,
    ])
    self.obs_dim = len(self.fixed_states_mean) + len(self.aaTarget_states_mean) + \
                   len(self.alarm_states_mean)
    self.dis_dim = 19
    self.con_dim = len(self.actions_mean) - 1

  def process_state(self, state):
    # class to numpy
    obs = np.array([])
    fixed_states = []
    for s in self.fixed_states:
      data = getattr(state, s)
      if s in ['fHeading_rad', 'fRoll_rad',]:
        fixed_states.append(np.sin(data))
        fixed_states.append(np.cos(data))
      elif s == 'nAAMMissileNum':
        missleindex = [0, 0, 0, 0, 0]
        missleindex[int(data)] = 1
        fixed_states += missleindex
      else:
        fixed_states.append(data)
    fixed_states = (np.array(fixed_states) - self.fixed_states_mean) / self.fixed_states_std
    obs = np.concatenate([obs, fixed_states])

    aaTarget_states = [0 for _ in range(len(self.aaTarget_states_mean))]
    for aaTarget_data in state.aaTargetData:
        idx = 0
        for s in self.aaTarget_states:
          data = getattr(aaTarget_data, s)
          if s in ['fTargetAzimDF_rad', 'fEntranceAngleDF_rad']:
            aaTarget_states[idx] = np.sin(data)
            idx += 1
            aaTarget_states[idx] = np.cos(data)
            idx += 1
          else:
            aaTarget_states[idx] = data
            idx += 1
        if aaTarget_data.bAETarget:
          break
    aaTarget_states = (np.array(aaTarget_states) - self.aaTarget_states_mean) / self.aaTarget_states_std
    obs = np.concatenate([obs, aaTarget_states])

    alarm_states = [0 for _ in range(len(self.alarm_states_mean))]
    for alarm_data in state.sAlarmInfo:
      if alarm_data.nPlatformType == 8:
        idx = 0
        for s in self.alarm_states:
          data = getattr(alarm_data, s)
          if s in ['fMisAzi']:
            alarm_states[idx] = np.sin(data)
            idx += 1
            alarm_states[idx] = np.cos(data)
            idx += 1
          else:
            alarm_states[idx] = data
            idx += 1
        break
    alarm_states = (np.array(alarm_states) - self.alarm_states_mean) / self.alarm_states_std
    obs = np.concatenate([obs, alarm_states])
    assert len(obs) == self.obs_dim
    return obs

  def process_action(self, action, raw_action, CmdID=True):
    # assign action to raw_action
    demo_action = copy.deepcopy(raw_action)
    if CmdID:
      demo_action.iCmdID = int(action[0]) + 1
      action = action[1:]
    action = np.clip(action, -1., 1.)
    demo_action.fcmdSpd = action[0] * self.actions_std[0] + self.actions_mean[0]
    demo_action.fcmNy = action[1] * self.actions_std[1] + self.actions_mean[1]
    demo_action.fCmdThrust = action[2] * self.actions_std[2] + self.actions_mean[2]
    demo_action.fCmdPitchDeg = action[3] * self.actions_std[3] + self.actions_mean[3]
    demo_action.fCmdRollDeg = np.arctan2(action[4], action[5]) / np.pi * 180
    demo_action.fCmdHeadingDeg = np.arctan2(action[6], action[7]) / np.pi * 180
    return demo_action

  def unprocess_action(self, raw_action):
    # class to numpy
    action = []
    # print(raw_action.__dict__)
    action.append(raw_action.iCmdID - 1)
    action.append(raw_action.fCmdSpd)
    action.append(raw_action.fCmdNy)
    action.append(raw_action.fCmdThrust)
    action.append(raw_action.fCmdPitchDeg)
    rollDeg = np.clip(raw_action.fCmdRollDeg, -180, 180)
    action.append(np.sin(rollDeg / 180 * np.pi))
    action.append(np.cos(rollDeg / 180 * np.pi))
    headingDeg = np.clip(raw_action.fCmdHeadingDeg, -180, 180)
    action.append(np.sin(headingDeg / 180 * np.pi))
    action.append(np.cos(headingDeg / 180 * np.pi))
    action = (np.array(action) - self.actions_mean) / self.actions_std
    return action
