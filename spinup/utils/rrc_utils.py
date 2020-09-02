import gym
from gym import wrappers


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


rrc_env_str = 'rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1'
rrc_ppo_wrappers = [{'cls': wrappers.FilterObservation, 
                     'kwargs': dict(filter_keys=['desired_goal', 'observation'])},
                    wrappers.FlattenObservation, 
                    {'cls': wrappers.RescaleAction, 
                     'args': [-1, 1]},
                    wrappers.ClipAction]
rrc_ppo_env_fn = make_env_fn(rrc_env_str, rrc_ppo_wrappers)

push_env_str = 'rrc_simulation.gym_wrapper:real_robot_challenge_phase_push-v1'
push_ppo_env_fn = make_env_fn(push_env_str, rrc_ppo_wrappers[1:])

test_push_ppo_env_fn = make_env_fn(push_env_str, rrc_ppo_wrappers[1:], visualization=True)