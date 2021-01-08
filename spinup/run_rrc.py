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


def run_rl_alg(alg_name='ppo', pos_coef=.1, ori_coef=.1, ori_thresh=np.pi/6, dist_thresh=0.09,
            ac_norm_pen=0.1, fingertip_coef=0.1, augment_rew=True,
            ep_len=EPLEN, frameskip=FRAMESKIP, rew_fn='exp',
            sample_radius=0.09, sa_relative=False, ts_relative=True,
            goal_relative=True, lim_pen=0.001, keep_goal=False, use_quat=False,
            cube_rew=False, step_rew=False, reorient_env=False, scaled_ac=False,
            task_space=False, **alg_kwargs):
    env_fn = None # rrc_utils.p2_reorient_env_fn
    early_stop = None # rrc_utils.success_rate_early_stopping
    if env_fn is None:
        env_fn = rrc_utils.build_env_fn(pos_coef=pos_coef, ori_coef=ori_coef,
                ori_thresh=ori_thresh, dist_thresh=dist_thresh,
                ac_norm_pen=ac_norm_pen, fingertip_coef=fingertip_coef,
                augment_rew=augment_rew, ep_len=ep_len, frameskip=frameskip,
                rew_fn=rew_fn, sample_radius=sample_radius,
                sa_relative=sa_relative, ts_relative=ts_relative,
                goal_relative=goal_relative, lim_pen=lim_pen, keep_goal=keep_goal,
                use_quat=use_quat, cube_rew=cube_rew, step_rew=step_rew,
                reorient_env=reorient_env, scaled_ac=scaled_ac,
                task_space=task_space)
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
    parser.add_argument('--load_path', type=str, default=None, help='path to pre-trained model')

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
    parser.add_argument('--relative_goalwrapper', '--rgw', nargs='*', type=bool, default=[])
    parser.add_argument('--relative_taskwrapper', '--rtw', nargs='*', type=bool, default=[])
    parser.add_argument('--relative_scaledwrapper', '--rsw', nargs='*', type=bool, default=[])
    parser.add_argument('--keep_goal', '--kg', nargs='*', type=bool, default=[])
    parser.add_argument('--use_quat', '--uq', nargs='*', type=bool, default=[])
    parser.add_argument('--cube_rew', action='store_true')
    parser.add_argument('--step_rew', action='store_true')

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
    if args.load_path:
        eg.add('load_path', args.load_path)
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
    if args.keep_goal:
        eg.add('keep_goal', args.keep_goal, 'kg')
    if args.use_quat:
        eg.add('use_quat', args.use_quat, 'uq')
    if args.relative_scaledwrapper:
        eg.add('sa_relative', args.relative_scaledwrapper, 'rsw')
    if args.relative_taskwrapper:
        eg.add('ts_relative', args.relative_taskwrapper, 'rtw')
    if args.relative_goalwrapper:
        eg.add('goal_relative', args.relative_goalwrapper, 'rgw')
    if args.cube_rew:
        eg.add('cube_rew', [args.cube_rew])
    if args.step_rew:
        eg.add('step_rew', [args.step_rew])

    # relative = [args.relative_scaledwrapper, args.relative_taskwrapper, args.relative_goalwrapper]
    # eg.add('relative', [relative], 'rel')
    eg.run(run_rl_alg, num_cpu=args.cpu, data_dir=args.data_dir,
           datestamp=args.datestamp)
