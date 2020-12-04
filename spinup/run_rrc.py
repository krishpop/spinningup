import functools
import gym
import numpy as np
import torch.nn as nn

from gym import wrappers
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch, sac_pytorch, td3_pytorch
from rrc_iprl_package.envs import cube_env, env_wrappers, rrc_utils


FRAMESKIP = 15
EPLEN = 9 * 1000 // FRAMESKIP

rl_algs = {'sac': sac_pytorch, 'ppo': ppo_pytorch, 'td3': td3_pytorch}


def build_env_fn(pos_coef=1., ori_coef=.5, ori_thresh=np.pi, dist_thresh=0.06,
            ac_norm_pen=0, fingertip_coef=0, augment_rew=True,
            ep_len=EPLEN, frameskip=FRAMESKIP, rew_fn='exp',
            sample_radius=0.09, ac_wrappers=[], relative=(False, False, False),
            lim_pen=0.,):
    scaled_ac = 'scaled' in ac_wrappers
    task_space = 'task' in ac_wrappers
    step_rew = 'step' in ac_wrappers
    sa_relative, ts_relative, goal_relative = relative
    env_str = 'real_robot_challenge_phase_2-v2'
    action_type = cube_env.ActionType.POSITION
    rew_wrappers = [functools.partial(env_wrappers.CubeRewardWrapper,
                                           pos_coef=pos_coef, ori_coef=ori_coef,
                                           ac_norm_pen=ac_norm_pen, fingertip_coef=fingertip_coef,
                                           rew_fn=rew_fn, augment_reward=augment_rew)]
    if step_rew:
        rew_wrappers.append(env_wrappers.StepRewardWrapper)
    rew_wrappers.append(functools.partial(env_wrappers.ReorientWrapper,
                                          goal_env=False, dist_thresh=dist_thresh,
                                          ori_thresh=ori_thresh))
    final_wrappers = []
    if scaled_ac:
        final_wrappers.append(functools.partial(env_wrappers.ScaledActionWrapper,
                              goal_env=False, relative=sa_relative,
                              lim_penalty=lim_pen))
    if goal_relative:
        final_wrappers.append(env_wrappers.RelativeGoalWrapper)

    final_wrappers += [functools.partial(wrappers.TimeLimit, max_episode_steps=ep_len),
                       rrc_utils.p2_log_info_wrapper,
                       wrappers.ClipAction, wrappers.FlattenObservation]
    ewrappers = []
    if task_space:
        assert not scaled_ac, 'Can only use TaskSpaceWrapper OR ScaledActionWrapper'
        ewrappers.append(functools.partial(env_wrappers.TaskSpaceWrapper,
                                              relative=ts_relative))
        action_type = cube_env.ActionType.TORQUE
    ewrappers += rew_wrappers + final_wrappers

    initializer = env_wrappers.ReorientInitializer(1, sample_radius)
    env_fn = rrc_utils.make_env_fn(env_str, ewrappers,
                                   initializer=rrc_utils.p2_fixed_reorient, # initializer,
                                   action_type=action_type,
                                   frameskip=frameskip)
    return env_fn


def run_rl_alg(alg_name='ppo', pos_coef=1., ori_coef=.5, ori_thresh=np.pi, dist_thresh=0.06,
            ac_norm_pen=0, fingertip_coef=0, augment_rew=True,
            ep_len=EPLEN, frameskip=FRAMESKIP, rew_fn='exp',
            sample_radius=0.09, ac_wrappers=[], relative=(False, False, False),
            lim_pen=0., **alg_kwargs):
    env_fn = None # rrc_utils.p2_reorient_env_fn
    early_stop = rrc_utils.success_rate_early_stopping
    if env_fn is None:
        env_fn = build_env_fn(pos_coef, ori_coef, ori_thresh, dist_thresh,
            ac_norm_pen, fingertip_coef, augment_rew,
            ep_len, frameskip, rew_fn,
            sample_radius, ac_wrappers, relative,
            lim_pen)
    rl_alg = rl_algs.get(alg_name)
    assert rl_alg is not None, 'alg_name {} is not valid'.format(alg_name)
    rl_alg(env_fn=env_fn, info_kwargs=rrc_utils.p2_info_kwargs,
                early_stopping_fn=early_stop ,**alg_kwargs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='ppo')
    parser.add_argument('--cpu', type=int, default=6)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--exp_name', type=str, default='ppo-reorient')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--datestamp', '--dt', action='store_true')

    # experiment grid arguments
    parser.add_argument('--frameskip', '--f', type=int, nargs='*', default=[])
    parser.add_argument('--pos_coef', '--pc', type=float, nargs='*', default=[])
    parser.add_argument('--ori_coef', '--oc', type=float, nargs='*', default=[])
    parser.add_argument('--fingertip_coef', '--fc', type=float, nargs='*', default=[])
    parser.add_argument('--pos_thresh', '--pt', type=float, nargs='*', default=[])
    parser.add_argument('--ori_thresh', '--ot', type=float, nargs='*', default=[])
    parser.add_argument('--sample_rad', '--sr', type=float, nargs='*', default=[])
    parser.add_argument('--ac_norm_pen', type=float, nargs='*', default=[])
    parser.add_argument('--lim_penalty', type=float, nargs='*', default=[])
    parser.add_argument('--ep_len', type=float, nargs='*', default=[])

    # run PPO wrapper arguments
    parser.add_argument('--scaled_acwrapper', '--saw', action='store_true')
    parser.add_argument('--task_acwrapper', '--taw', action='store_true')
    parser.add_argument('--step_rewwrapper', '--srw', action='store_true')
    parser.add_argument('--relative_goalwrapper', '--rgw', action='store_true')
    parser.add_argument('--relative_taskwrapper', '--rtw', action='store_true')
    parser.add_argument('--relative_scaledwrapper', '--rsw', action='store_true')

    args = parser.parse_args()

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', args.epochs)
    eg.add('alg_name', args.alg)
    if args.alg == 'sac':
        if args.replay_size:
            eg.add('replay_size', args.replay_size, 'rbsize')
    if args.steps_per_epoch:
        eg.add('steps_per_epoch', args.steps_per_epoch)
    eg.add('ac_kwargs:hidden_sizes', [(64,64)], 'hid')
    eg.add('ac_kwargs:activation', [nn.ReLU], 'ac-act')
    if args.frameskip:
        eg.add('frameskip', args.frameskip, 'fs')
    if args.pos_coef:
        eg.add('pos_coef', args.pos_coef, 'rew-pos')
    if args.ori_coef:
        eg.add('ori_coef', args.ori_coef, 'rew-ori')
    if args.fingertip_coef:
        eg.add('fingertip_coef', args.fingertip_coef, 'rew-tip')
    if args.ac_norm_pen:
        eg.add('ac_norm_pen', args.ac_norm_pen, 'rew-pen')
    if args.lim_penalty:
        eg.add('lim_pen', [abs(lp) for lp in args.lim_penalty], 'lp')
    if args.pos_thresh:
        eg.add('pos_thresh', args.pos_thresh, 'pt')
    if args.ori_thresh:
        eg.add('ori_thresh', args.ori_thresh, 'ot')
    if args.sample_rad:
        eg.add('sample_radius', args.sample_rad, 'sr')
    if args.ep_len:
        eg.add('ep_len', args.ep_len, 'el')

    eg.add('ac_wrappers', [('scaled')], 'acw')
    # relative = [args.relative_scaledwrapper, args.relative_taskwrapper, args.relative_goalwrapper]
    # eg.add('relative', [relative], 'rel')
    eg.run(run_rl_alg, num_cpu=args.cpu, data_dir=args.data_dir,
           datestamp=args.datestamp)
