from spinup.utils import rrc_utils
import functools
import gym
import numpy as np
import torch.nn as nn

from gym import wrappers
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
from rrc_simulation.gym_wrapper.envs import cube_env, custom_env


FRAMESKIP = 10
EPLEN = 50


def run_ppo(env_str='real_robot_challenge_phase_1-v3', pos_coef=1., ori_coef=1.,
            ac_norm_pen=.1, fingertip_coef=1., augment_rew=False,
            ori_thresh=np.pi/6, dist_thresh=0.06, ep_len=EPLEN, frameskip=FRAMESKIP,
            ac_wrappers=[], relative=(False, False), lim_penalty=0., **ppo_kwargs):
    scaled_ac = 'scaled' in ac_wrappers
    task_space = 'task' in ac_wrappers
    step_rew = 'step' in ac_wrappers
    sa_relative, ts_relative = relative
    rew_wrappers = [functools.partial(custom_env.CubeRewardWrapper,
                                           pos_coef=pos_coef, ori_coef=ori_coef,
                                           ac_norm_pen=ac_norm_pen, fingertip_coef=fingertip_coef,
                                           rew_fn='exp', augment_reward=augment_rew)]
    if step_rew:
        rew_wrappers.append(custom_env.StepRewardWrapper)
    rew_wrappers.append(functools.partial(custom_env.ReorientWrapper,
                                          goal_env=False, dist_thresh=dist_thresh,
                                          ori_thresh=ori_thresh))
    final_wrappers = []
    if scaled_ac:
        final_wrappers.append(functools.partial(custom_env.ScaledActionWrapper,
                              goal_env=False, relative=sa_relative,
                              lim_penalty=lim_penalty))

    final_wrappers += [functools.partial(wrappers.TimeLimit, max_episode_steps=ep_len),
                       rrc_utils.log_info_wrapper,
                       wrappers.ClipAction, wrappers.FlattenObservation]
    env_wrappers = []
    if task_space:
        env_wrappers.append(functools.partial(custom_env.TaskSpaceWrapper,
                            relative=ts_relative))
    env_wrappers += rew_wrappers + final_wrappers

    initializer = custom_env.ReorientInitializer(1, 0.09)
    env_fn = rrc_utils.make_env_fn(env_str, env_wrappers,
                                   initializer=initializer,
                                   action_type=cube_env.ActionType.POSITION,
                                   visualization=False, frameskip=frameskip)
    ppo_pytorch(env_fn=env_fn, info_kwargs=rrc_utils.reorient_info_kwargs, **ppo_kwargs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=6)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--env_name', type=str,
                        default='real_robot_challenge_phase_1-v3')
    parser.add_argument('--exp_name', type=str, default='ppo-reorient')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--datestamp', '--dt', action='store_true')

    args = parser.parse_args()

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('env_str', args.env_name, '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 250)
    eg.add('steps_per_epoch', 3000)
    eg.add('ac_kwargs:hidden_sizes', [(64,64)], 'hid')
    eg.add('ac_kwargs:activation', [nn.ReLU], 'ac-act')
    eg.add('frameskip', [1, FRAMESKIP], 'fs')
    eg.add('pos_coef', [0.5, 1., 2.], 'rew-pos')
    eg.add('ori_coef', [0.5, 1., 2.], 'rew-ori')
    eg.add('fingertip_coef', [.1], 'rew-ftip')
    eg.add('ac_norm_pen', [0.01, 0.1, .5, 10.], 'rew-pen')
    eg.add('relative', [(True, False)], 'rel')
    eg.add('ac_wrappers', [('scaled',), ('scaled', 'step')], 'acw')
    eg.run(run_ppo, num_cpu=args.cpu, data_dir=args.data_dir,
           datestamp=args.datestamp)
