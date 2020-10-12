import unittest
import numpy as np

from spinup.utils import rrc_utils


def get_pos_from_obs(observation):
    return observation[-27:-18], observation[-18:-9]


def main():
    env = rrc_utils.recenter_rel_ppo_env_fn()
    obs = env.reset()
    q_pos, x_pos = get_pos_from_obs(obs)
    obs, r, d, i = env.step(env.action_space.high)
    q_pos_next, x_pos_next = get_pos_from_obs(obs)
    obs, r, d, i = env.step(env.action_space.low)
    q_pos_low, x_pos_low = get_pos_from_obs(obs)
    print('delta q_pos:', np.linalg.norm(q_pos_next - q_pos))
    print('delta x_pos:', np.linalg.norm(x_pos_next - x_pos))
    print('delta x_pos min action:', np.linalg.norm(x_pos_next - x_pos_low))

    obs = [get_pos_from_obs(env.reset())[1]]
    for _ in range(30):
        o,r,d,i = env.step(env.action_space.sample())
        obs.append(get_pos_from_obs(o)[1])
    obs = np.asarray(obs)
    print('mean delta x_pos:', np.abs(obs[:-1] - obs[1:]).mean(axis=0))


if __name__ == '__main__':
    main()
