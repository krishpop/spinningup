import json
import numpy as np
import torch
from rrc_iprl_package.control.custom_pinocchio_utils import CustomPinocchioUtils
from spinup.algos.pytorch.ppo.ppo import PPOBuffer
from spinup.utils.logx import EpochLogger


FRAMESKIP = 15
EP_LEN = 6 * 1000 // 15

def transitions_to_replay_buffer(npz_path, env, ac, ep_len=EP_LEN):
    action_log = np.load(npz_path, allow_pickle=True)['action_log']
    transitions = [x for x in action_log if isinstance(x['action'], np.ndarray)]
    obs_dim, act_dim = env.observation_space.shape, env.action_space.shape
    env.reset()
    cpu = CustomPinocchioUtils(env.platform.simfinger.finger_urdf_path,
                               env.platform.simfinger.tip_link_names)

    local_steps_per_epoch = int(steps_per_epoch / num_procs())

    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    t = transitions[0]
    done = False
    for next_t in transitions[1:]:
        if done:
            done = False
            t = next_t
            continue
        o, a, r = t['observation'], t['action'], t['reward']
        o = process_observation(o, env, cpu)
        with torch.no_grad():
            pi = ac.pi._distribution(o)
            logp_a = ac.pi._log_prob_from_distribution(pi, a)
            v, logp = ac.v(o)
        a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
        buf.store(o, a, r, v.numpy(), logp.numpy())
        if next_t['t'] - t['t'] > FRAMESKIP + 1:
            _, v, _ = ac.step(torch.as_tensor(next_t['observation'], dtype=torch.float32))
            buf.finish_path(v)
            done = True
        t = next_t

    return buf

def process_rl_observation(obs, env, cpu, obs_names=None):
    if obs_names is None:
        obs_names = list(env.unwrapped.observation_space.spaces.keys())
    obs_list = []
    for on in obs_names:
        if on == 'robot_position':
            val = obs['observation']['position']
        elif on == 'robot_velocity':
            val = obs['observation']['velocity']
        elif on == 'robot_tip_positions':
            val = cpu.forward_kinematics(obs['observation']['position'])
        elif on == 'object_position':
            val = obs['achieved_goal']['position']
        elif on == 'object_orientation':
            val = obs['achieved_goal']['orientation']
        elif on == 'goal_position':
            val = obs['desired_goal']['position']
        elif on == 'goal_orientation':
            val = obs['desired_goal']['orientation']
        elif on in ['action', 'last_action']:
            if isinstance(obs['action'], dict):
                val = obs['action']['torque'] if action
        else:
            # in goal_env case, just concat observation, achived, and desired together
            obs_list.append(obs[on])
        obs_list.append(np.asarray(val, dtype='float64').flatten())
    return np.concatenate(obs_list)


def train_offline(ac, buf, env, save_path=None, steps_per_epoch=4000,
        gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), info_kwargs=dict()):
    epochs = 1

    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(save_path, 'vars'+itr+'.pkl'))
        env = state['env']
    except Exception as e:
        print("Failed to load env! Got error: {}".format(str(e)))
        env = None

    config = json.load(open(osp.join(save_path, config), 'r'))
    logger_kwargs['exp_name'] = config['logger_kwargs']['exp_name']

    max_ep_len = config['max_ep_len']

    logger_kwargs.update({"output_dir": save_path})
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    save_path = osp.join(save_path, 'pyt_save')
    assert osp.exists(save_path), 'provide a valid save/load path, {} does not exist'.format(save_path)
    p = re.compile('\d+')
    saves = [int(x.split('.')[0][5:]) for x in os.listdir(save_path) if len(x)>8 and 'model' in x]
    if 'fine_model.pt' in list(os.listdir(save_path)):
        load_path = save_path = osp.join(save_path, 'ft_model.pt')
    else:
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(save_path) if len(x)>8 and 'model' in x]
        load_path = osp.join(save_path, 'model{}.pt'.format(max(saves)))
        save_path = osp.join(save_path, 'ft_model.pt')

    ac = torch.load(load_path)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    # o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, epoch)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        early_stop = False
        for k, v in info_kwargs.items():
            if v in logger.epoch_dict:
                with_min_and_max = 'Val' not in v
                logger.log_tabular(v, average_only=True,
                        with_min_and_max=with_min_and_max)
            if k == 'is_success' and early_stopping_fn:
                early_stop = early_stopping_fn(
                        steps=epoch * local_steps_per_epoch * num_procs(),
                        success_rate=logger.log_current_row[v])

        logger.dump_tabular()


