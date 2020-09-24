import gym
from gym import wrappers
from rrc_simulation.gym_wrapper.envs import cube_env, custom_env
from rrc_simulation.tasks import move_cube
from gym.envs.registration import register


registered_envs = [spec.id for spec in gym.envs.registry.all()]

FRAMESKIP = 10
EPLEN = move_cube.episode_length // FRAMESKIP

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


def make_env_fn(env_str, wrapper_params=[], **make_kwargs):
    """Returns env_fn to pass to spinningup alg"""

    def env_fn():
        env = gym.make(env_str, **make_kwargs)
        for w in wrapper_params:
            if isinstance(w, dict):
                env = w['cls'](env, *w.get('args', []), **w.get('kwargs', {}))
            else:
                env = w(env)
        return env
    return env_fn


if cube_env:
    push_random_initializer = cube_env.RandomInitializer(difficulty=1)
    push_curr_initializer = custom_env.CurriculumInitializer(initial_dist=0.,
                                                             num_levels=5)
    push_fixed_initializer = custom_env.CurriculumInitializer(initial_dist=0.,
                                                              num_levels=2)
    reorient_initializer = custom_env.CurriculumInitializer(
            initial_dist=0.05, num_levels=3, difficulty=4,
            fixed_goal=custom_env.RandomOrientationInitializer.goal)
    reorient_initializer = custom_env.ReorientInitializer(1, 0.1)

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
                 'final_score', 'goal_sample_radius', 'init_sample_radius']
    reorient_info_keys = ['is_success', 'is_success_ori', 'final_dist', 'final_score',
                          'final_ori_dist', 'final_ori_scaled']

    rrc_ppo_wrappers = [
            {'cls': wrappers.FilterObservation,
             'kwargs': dict(filter_keys=['desired_goal',
                                         'observation'])},
            wrappers.FlattenObservation,
            wrappers.ClipAction,
            {'cls': wrappers.TimeLimit,
             'kwargs': dict(max_episode_steps=EPLEN)},
            ]
    rrc_vds_wrappers = [
            {'cls': wrappers.TimeLimit,
             'kwargs': dict(max_episode_steps=EPLEN)},
            custom_env.FlattenGoalWrapper,
            ]

    push_wrappers = [
            {'cls': custom_env.LogInfoWrapper,
             'kwargs': dict(info_keys=info_keys)},
            {'cls': custom_env.CubeRewardWrapper,
             'kwargs': dict(pos_coef=1., ac_norm_pen=0.2, rew_fn='exp')}
        ]
    push_wrappers = rrc_ppo_wrappers[1:] + push_wrappers
    reorient_wrappers = [
            {'cls': custom_env.LogInfoWrapper,
             'kwargs': dict(info_keys=reorient_info_keys)},
            {'cls': custom_env.CubeRewardWrapper,
             'kwargs': dict(pos_coef=.2, ori_coef=.5,
                            ac_norm_pen=0.2, rew_fn='exp')},
        ]
    reorient_wrappers = rrc_ppo_wrappers[1:] + reorient_wrappers

    action_type = cube_env.ActionType.POSITION

    rrc_env_str = 'rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1'
    rrc_ppo_env_fn = make_env_fn(rrc_env_str, rrc_ppo_wrappers,
                                 initializer=push_initializer,
                                 action_type=action_type,
                                 visualization=False,
                                 frameskip=FRAMESKIP)
    test_ppo_env_fn = make_env_fn(rrc_env_str, rrc_ppo_wrappers,
                                  initializer=push_initializer,
                                  action_type=action_type,
                                  visualization=True,
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

    test_push_ppo_env_fn = make_env_fn(push_env_str, push_wrappers,
                                       initializer=push_initializer,
                                       action_type=action_type,
                                       visualization=True,
                                       frameskip=FRAMESKIP)

    task_space_wrappers = [wrappers.ClipAction,
                           {'cls': custom_env.TaskSpaceWrapper,
                            'kwargs': dict(relative=False)},
                           {'cls': custom_env.ReorientWrapper,
                            'kwargs': dict(goal_env=False, dist_thresh=0.08)},
                           dict(cls=wrappers.TimeLimit, kwargs=dict(max_episode_steps=EPLEN)),
                           dict(cls=custom_env.LogInfoWrapper, kwargs=dict(info_keys=info_keys[:-2])),
                           dict(cls=custom_env.CubeRewardWrapper, kwargs=dict(
                               pos_coef=.8, ori_coef=1., ac_norm_pen=.2, rew_fn='exp')),
                           wrappers.FlattenObservation]

    reorient_env_str = 'real_robot_challenge_phase_1-v3'
    task_space_env_fn = make_env_fn(push_env_str, task_space_wrappers,
                                  initializer=reorient_initializer,
                                  action_type=cube_env.ActionType.TORQUE,
                                  visualization=False,
                                  frameskip=FRAMESKIP)


    reorient_ppo_env_fn = make_env_fn(reorient_env_str, reorient_wrappers,
                                  initializer=reorient_initializer,
                                  action_type=action_type,
                                  visualization=False,
                                  frameskip=FRAMESKIP)

    test_reorient_ppo_env_fn = make_env_fn(reorient_env_str, push_wrappers,
                                       initializer=reorient_initializer,
                                       action_type=action_type,
                                       visualization=True,
                                       frameskip=FRAMESKIP)
