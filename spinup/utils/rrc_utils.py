import gym
import numpy as np
from gym import wrappers
import functools
from rrc_simulation.gym_wrapper.envs import cube_env, custom_env
from rrc_simulation.tasks import move_cube
from gym.envs.registration import register


registered_envs = [spec.id for spec in gym.envs.registry.all()]

FRAMESKIP = 10
EPLEN = move_cube.episode_length // FRAMESKIP
EPLEN_SHORT = 100  # 500 total timesteps

if "real_robot_challenge_phase_1-v2" not in registered_envs:
    register(
        id="real_robot_challenge_phase_1-v2",
        entry_point=custom_env.PushCubeEnv
        )
if "real_robot_challenge_phase_1-v3" not in registered_envs:
    register(
        id="real_robot_challenge_phase_1-v3",
        entry_point=custom_env.PushReorientCubeEnv
        )
if "real_robot_challenge_phase_1-v4" not in registered_envs:
    register(
        id="real_robot_challenge_phase_1-v4",
        entry_point=custom_env.SparseCubeEnv
        )


total_steps = 5e6
step_rates = np.linspace(0, 0.6, 10)

def success_rate_early_stopping(steps, success_rate):
    return step_rates[min(9, int(steps/total_steps * 10))] > success_rate


def make_env_fn(env_str, wrapper_params=[], **make_kwargs):
    """Returns env_fn to pass to spinningup alg"""

    def env_fn(visualization=False):
        env = gym.make(env_str, visualization=visualization, **make_kwargs)
        for w in wrapper_params:
            if isinstance(w, dict):
                env = w['cls'](env, *w.get('args', []), **w.get('kwargs', {}))
            else:
                env = w(env)
        return env
    return env_fn


push_random_initializer = cube_env.RandomInitializer(difficulty=1)

fixed_reorient_initializer = cube_env.RandomGoalReorientInitializer(difficulty=1)

push_curr_initializer = custom_env.CurriculumInitializer(initial_dist=0.,
                                                         num_levels=5)
push_fixed_initializer = custom_env.CurriculumInitializer(initial_dist=0.,
                                                          num_levels=2)
reorient_initializer = reorient_curr_initializer = custom_env.CurriculumInitializer(
        initial_dist=0.06, num_levels=3, difficulty=4,
        fixed_goal=custom_env.RandomOrientationInitializer.goal)
recenter_initializer = custom_env.ReorientInitializer(1, 0.09)

push_initializer = push_fixed_initializer

lift_initializer = cube_env.RandomInitializer(difficulty=2)
ori_initializer = cube_env.RandomInitializer(difficulty=3)
# Val in info string calls logger.log_tabular() with_min_and_max to False
push_info_kwargs = {'is_success': 'SuccessRateVal', 'final_dist': 'FinalDist',
    'final_score': 'FinalScore', 'init_sample_radius': 'InitSampleDistVal',
    'goal_sample_radius': 'GoalSampleDistVal'}
reorient_info_kwargs = {'is_success': 'SuccessRateVal',
        'is_success_ori': 'OriSuccessRateVal',
        'final_dist': 'FinalDist', 'final_ori_dist': 'FinalOriDist',
        'final_ori_scaled': 'FinalOriScaledDist',
        'final_score': 'FinalScore'}

info_keys = ['is_success', 'is_success_ori', 'final_ori_dist', 'final_dist',
             'final_score']
curr_info_keys = info_keys + ['goal_sample_radius', 'init_sample_radius']
reorient_info_keys = ['is_success', 'is_success_ori', 'final_dist', 'final_score',
                      'final_ori_dist', 'final_ori_scaled']
action_type = cube_env.ActionType.POSITION

log_info_wrapper = functools.partial(custom_env.LogInfoWrapper,
                                     info_keys=info_keys)
reorient_log_info_wrapper = functools.partial(custom_env.LogInfoWrapper,
                                              info_keys=reorient_info_keys)

final_wrappers = [functools.partial(custom_env.ScaledActionWrapper, goal_env=True,
                                    relative=True),
                  functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN),
                  log_info_wrapper,
                  wrappers.ClipAction, wrappers.FlattenObservation]
final_wrappers = final_wrappers_short = [
       functools.partial(custom_env.ScaledActionWrapper, goal_env=False, relative=True),
       functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN_SHORT),
       log_info_wrapper,
       wrappers.FlattenObservation]

final_wrappers_reorient = [
        functools.partial(custom_env.ScaledActionWrapper,
            goal_env=False, relative=True),
        functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN_SHORT),
        reorient_log_info_wrapper,
        wrappers.FlattenObservation]

final_wrappers_reorient_abs = [
        functools.partial(custom_env.ScaledActionWrapper,
            goal_env=False, relative=False),
        functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN_SHORT),
        reorient_log_info_wrapper,
        wrappers.FlattenObservation]


final_wrappers_vds = [functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN),
        custom_env.FlattenGoalWrapper]

abs_task_wrapper = functools.partial(custom_env.TaskSpaceWrapper, relative=False)
rel_task_wrapper = functools.partial(custom_env.TaskSpaceWrapper, relative=True)
rew_wrappers_step = [functools.partial(custom_env.CubeRewardWrapper,
                                       pos_coef=1., ori_coef=1.,
                                       ac_norm_pen=0.2, fingertip_coef=1.,
                                       rew_fn='exp', augment_reward=True),
                     custom_env.StepRewardWrapper,
                     functools.partial(custom_env.ReorientWrapper,
                                       goal_env=False, dist_thresh=0.075,
                                       ori_thresh=np.pi/6)]
rew_wrappers = [functools.partial(custom_env.CubeRewardWrapper,
                                  pos_coef=.1, ori_coef=.1,
                                  ac_norm_pen=.1, fingertip_coef=.1,
                                  rew_fn='exp', augment_reward=True),
                functools.partial(custom_env.ReorientWrapper,
                                  goal_env=False, dist_thresh=0.075,
                                  ori_thresh=np.pi)]

recenter_rew_wrappers = [functools.partial(custom_env.CubeRewardWrapper, pos_coef=1., ori_coef=.5,
                                       ac_norm_pen=0., augment_reward=True, rew_fn='exp'),
                     functools.partial(custom_env.ReorientWrapper, goal_env=False, dist_thresh=0.05,
                         ori_thresh=np.pi)]

goal_filter_wrapper = [functools.partial(wrappers.FilterObservation,
                                        filter_keys=['desired_goal',
                                            'observation'])]
rrc_ppo_wrappers = goal_filter_wrapper + final_wrappers
rrc_vds_wrappers = goal_filter_wrapper + final_wrappers_vds

push_wrappers = [functools.partial(custom_env.CubeRewardWrapper, pos_coef=1.,
                          ac_norm_pen=0.2, rew_fn='exp')]
push_wrappers = push_wrappers + final_wrappers

recenter_wrappers_rel = recenter_rew_wrappers + final_wrappers_reorient
recenter_wrappers_abs = recenter_rew_wrappers + final_wrappers_reorient_abs

abs_task_wrappers =  [abs_task_wrapper] + rew_wrappers + final_wrappers_reorient[1:] + [wrappers.ClipAction]
rel_task_wrappers =  [rel_task_wrapper] + rew_wrappers + final_wrappers_reorient[1:] + [wrappers.ClipAction]

abs_task_step_wrappers =  [abs_task_wrapper] + rew_wrappers_step + final_wrappers_reorient[1:] + [wrappers.ClipAction]
rel_task_step_wrappers =  [rel_task_wrapper] + rew_wrappers_step + final_wrappers_reorient[1:] + [wrappers.ClipAction]

rrc_env_str = 'rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1'
rrc_ppo_env_fn = make_env_fn(rrc_env_str, rrc_ppo_wrappers,
                             initializer=push_initializer,
                             action_type=action_type,
                             visualization=False,
                             frameskip=FRAMESKIP)

rrc_vds_env_fn = make_env_fn(rrc_env_str, rrc_vds_wrappers,
                             initializer=push_initializer,
                             action_type=action_type,
                             visualization=False,
                             frameskip=FRAMESKIP)


push_env_str = 'real_robot_challenge_phase_1-v2'
push_ppo_env_fn = make_env_fn(push_env_str, push_wrappers,
                              initializer=push_initializer,
                              action_type=action_type,
                              visualization=False,
                              frameskip=FRAMESKIP)

reorient_env_str = 'real_robot_challenge_phase_1-v3'
abs_task_env_fn = make_env_fn(reorient_env_str, abs_task_wrappers,
                              initializer=reorient_initializer,
                              action_type=cube_env.ActionType.TORQUE,
                              visualization=False,
                              frameskip=5)
rel_task_env_fn = make_env_fn(reorient_env_str, rel_task_wrappers,
                              initializer=reorient_initializer,
                              action_type=cube_env.ActionType.TORQUE,
                              visualization=False,
                              frameskip=5)


abs_task_step_env_fn = make_env_fn(reorient_env_str, abs_task_step_wrappers,
                              initializer=reorient_initializer,
                              action_type=cube_env.ActionType.TORQUE,
                              visualization=False,
                              frameskip=5)
rel_task_step_env_fn = make_env_fn(reorient_env_str, rel_task_step_wrappers,
                              initializer=reorient_initializer,
                              action_type=cube_env.ActionType.TORQUE,
                              visualization=False,
                              frameskip=5)

recenter_rel_ppo_env_fn = make_env_fn(reorient_env_str, recenter_wrappers_rel,
                              initializer=recenter_initializer,
                              action_type=action_type,
                              visualization=False,
                              frameskip=FRAMESKIP)

recenter_ppo_env_fn = make_env_fn(reorient_env_str, recenter_wrappers_abs,
                              initializer=recenter_initializer,
                              action_type=action_type,
                              visualization=False,
                              frameskip=FRAMESKIP)

reorient_center_env_fn = make_env_fn(reorient_env_str, recenter_wrappers_rel,
                                     initializer=fixed_reorient_initializer,
                                     action_type=action_type,
                                     visualization=False)


eval_keys = ['is_success', 'is_success_ori', 'final_ori_dist', 'final_dist',
             'final_score']
