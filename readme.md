**Status:** Maintenance (expect bug fixes and minor updates)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```


(added by @krishpop) RRC Additions
---------------------------------

Additional functionality added for [Real Robot Challenge](https://real-robot-challenge.com/):

- spinup/utils/rrc_utils.py contains utility functions for creating RRC envs with appropriate wrappers

Example Usage:

```
$ python -m spinup.run ppo --env_fn rrc_utils.rrc_ppo_env_fn --exp_name ppo_rrc_t1 --cpu auto
$ python -m spinup.run ppo --exp_name ppo_push ----env_fn rrc_utils.push_ppo_env_fn \
    --steps_per_epoch 22500 --epochs 50 --cpu 10 --info_kwargs rrc_utils.push_info_kwargs
```

To visualize a trained policy:
```
$ tar -xzvf ./data/scaled_actions.tar.gz
$ python -m spinup.run test_policy ./data/scaled_actions/scaled_actions_s0/ \
    --env_fn rrc_utils.test_reorient_ppo_env_fn -n-r -n 1
```
