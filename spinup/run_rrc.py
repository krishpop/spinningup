import functools
import gym
import numpy as np
import torch.nn as nn

from gym import wrappers
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch, sac_pytorch, td3_pytorch
from rrc_iprl_package.envs import cube_env, env_wrappers, rrc_utils

FRAMESKIP = 15

rl_algs = {'sac': sac_pytorch, 'ppo': ppo_pytorch, 'td3': td3_pytorch}


def run_rl_alg(difficulty=1, alg_name='ppo', pos_coef=.1, ori_coef=.1, ori_thresh=np.pi/8,
               dist_thresh=.09, ac_norm_pen=.01, fingertip_coef=.1,
               augment_rew=True, ep_len=None, frameskip=FRAMESKIP,
               rew_fn='exp', action_type='torque', sample_radius=0.09,
               residual=False, sa_relative=True,
               ts_relative=False, goal_relative=False, lim_pen=0.001,
               keep_goal=False, use_quat=False, cube_rew=False, step_rew=False,
               reorient_env=False, scaled_ac=False, task_space=False,
               res_torque=True,
               **alg_kwargs):
    env_fn = None # rrc_utils.p2_reorient_env_fn
    # early_stop = None # rrc_utils.success_rate_early_stopping
    ep_len = ep_len or 9 * 1000 // frameskip  # 15 seconds of interaction
    if env_fn is None:
        env_fn = rrc_utils.build_env_fn(difficulty=difficulty,
                pos_coef=pos_coef, ori_coef=ori_coef,
                ori_thresh=ori_thresh, dist_thresh=dist_thresh,
                ac_norm_pen=ac_norm_pen, fingertip_coef=fingertip_coef,
                augment_rew=augment_rew, ep_len=ep_len, frameskip=frameskip,
                rew_fn=rew_fn, action_type=action_type, sample_radius=sample_radius,
                residual=residual, sa_relative=sa_relative, ts_relative=ts_relative,
                goal_relative=goal_relative, lim_pen=lim_pen, keep_goal=keep_goal,
                use_quat=use_quat, cube_rew=cube_rew, step_rew=step_rew,
                reorient_env=reorient_env, scaled_ac=scaled_ac,
                task_space=task_space, res_torque=res_torque)
    assert alg_name in rl_algs, \
           'alg_name {} is not in {}'.format(alg_name, list(rl_algs.keys()))
    rl_alg = rl_algs.get(alg_name)
    rl_alg(env_fn=env_fn, info_kwargs=rrc_utils.p2_info_kwargs,
           **alg_kwargs)

def process(arg):
    if isinstance(arg, str):
        return eval(arg)
    return arg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='ppo')
    parser.add_argument('--cpu', type=int, default=6)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--exp_name', type=str, default='ppo-rrc')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--datestamp', '--dt', action='store_true')
    parser.add_argument('--load_path', type=str, default=None, help='path to pre-trained model')
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--difficulty', type=int, nargs='*', default=[1])
    parser.add_argument('--action_type', type=str, default='pos')

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
    parser.add_argument('--keep_goal', '--kg', nargs='*', type=bool, default=[])
    parser.add_argument('--use_quat', '--uq', nargs='*', type=bool, default=[])

    # run PPO wrapper arguments
    parser.add_argument('--residual', '--res', action='store_true')
    parser.add_argument('--res_force', '--rf',  action='store_true')
    parser.add_argument('--scaled_acwrapper', '--saw', action='store_true')
    parser.add_argument('--task_acwrapper', '--taw', action='store_true')
    parser.add_argument('--step_rewwrapper', '--srw', action='store_true')
    parser.add_argument('--relative_goalwrapper', '--rgw', nargs='*', type=bool, default=[])
    parser.add_argument('--relative_taskwrapper', '--rtw', nargs='*', type=bool, default=[])
    parser.add_argument('--relative_scaledwrapper', '--rsw', nargs='*', type=bool, default=[])
    parser.add_argument('--cube_rew', action='store_true')
    parser.add_argument('--step_rew', action='store_true')
    parser.add_argument('--rew_fn', type=str)
    parser.add_argument('--act_fn', nargs='*', type=str, default=[nn.ReLU])

    args = parser.parse_args()

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', args.epochs)
    eg.add('alg_name', args.alg)
    eg.add('difficulty', args.difficulty, 'lvl')
    eg.add('action_type', args.action_type, 'atype')
    if args.alg == 'sac':
        if args.replay_size:
            eg.add('replay_size', args.replay_size, 'rbsize')
    if args.steps_per_epoch:
        eg.add('steps_per_epoch', args.steps_per_epoch)
    if args.load_path:
        eg.add('load_path', args.load_path)
    eg.add('ac_kwargs:hidden_sizes', [(64,64)], 'hid')
    if args.act_fn:
        eg.add('ac_kwargs:activation', [process(af) for af in args.act_fn], 'ac-act')
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
    if args.residual:
        eg.add('residual', args.residual, 'res')
    if args.res_force:
        eg.add('res_torque', False, 'rtor')
    if args.relative_scaledwrapper:
        eg.add('sa_relative', args.relative_scaledwrapper, 'rsw')
    if args.relative_taskwrapper:
        eg.add('ts_relative', [False, True], 'rtw')
    if args.relative_goalwrapper:
        eg.add('goal_relative', args.relative_goalwrapper, 'rgw')
    if args.cube_rew:
        eg.add('cube_rew', [args.cube_rew])
    if args.step_rew:
        eg.add('step_rew', [args.step_rew])
    if args.rew_fn:
        eg.add('rew_fn', args.rew_fn)

    if args.scaled_acwrapper:
        eg.add('scaled_ac', args.scaled_acwrapper)
    if args.task_acwrapper:
        eg.add('task_space', args.task_acwrapper)
    if args.step_rewwrapper:
        eg.add('step_rew', args.step_rewwrapper)
    if args.alg == 'ppo':
        eg.run(run_rl_alg, num_cpu=args.cpu, data_dir=args.data_dir,
               datestamp=args.datestamp)
    else:
        eg.run(run_rl_alg, data_dir=args.data_dir,
               datestamp=args.datestamp)

