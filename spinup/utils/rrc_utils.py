import gym
from gym import wrappers


FRAMESKIP = 3

try:
    from rrc_simulation.gym_wrapper.envs import cube_env, custom_env
    from rrc_simulation.tasks import move_cube
    from gym.envs.registration import register

    registered_envs = [spec.id for spec in gym.envs.registry.all()]
    EPLEN= move_cube.episode_length // FRAMESKIP
    if "real_robot_challenge_phase_1-v2" not in registered_envs:
        register(
            id="real_robot_challenge_phase_1-v2",
            entry_point=custom_env.PushCubeEnv
            )
except ImportError:
    move_cube = cube_env = custom_env = None




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
    rrc_env_str = 'rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1'
    push_initializer = cube_env.RandomInitializer(difficulty=1)
    lift_initializer = cube_env.RandomInitializer(difficulty=2)
    ori_initializer = cube_env.RandomInitializer(difficulty=3) 
    rrc_ppo_wrappers = [
            {'cls': wrappers.TimeLimit, 
             'kwargs': dict(max_episode_steps=EPLEN)},
            {'cls': wrappers.FilterObservation, 
             'kwargs': dict(filter_keys=['desired_goal', 
                                         'observation'])},
            wrappers.FlattenObservation, 
            {'cls': wrappers.RescaleAction, 
             'args': [-1, 1]},
            wrappers.ClipAction
            ]
    push_wrappers = rrc_ppo_wrappers[:1] + rrc_ppo_wrappers[2:]
    action_type = cube_env.ActionType.POSITION
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

