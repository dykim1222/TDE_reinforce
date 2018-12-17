import copy
import glob
import os
import time
import pdb
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot
from mymodels import TemporalDifferenceModule, CollectSamples, Logger


args = get_args()
assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

args.num_processes=8
num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

try:
    os.makedirs(args.log_dir, exist_ok=True)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)









def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, True)

    frame_skip = 4          # frame skip
    if args.tb_dir[-1] != '/':
        args.tb_dir = args.tb_dir + '/'
    logger = Logger(args.tb_dir)
    logger.write_settings(args)
    if args.use_tdm:

        # beta scheduler
        if args.beta_schedule == 'const':
            beta_func = lambda x : float(args.beta_int)
        elif args.beta_schedule == 'sqrt':
            beta_func = lambda x : 1./np.sqrt(x+1)
        elif args.beta_schedule == 'log':
            beta_func = lambda x : 1./np.log(x+2)
        elif args.beta_schedule == 'linear':
            beta_func = lambda x : 1./(x+1)

        # bonus function variations
        if args.bonus_func == 'linear':
            bonus_func = lambda x : x+1
        elif args.bonus_func == 'square':
            bonus_func = lambda x : (x+1)**2
        elif args.bonus_func == 'sqrt':
            bonus_func = lambda x : (x+1)**(1/2)
        elif args.bonus_func == 'log':
            bonus_func = lambda x : np.log(x+1)


        # temporal difference module
        tdm = TemporalDifferenceModule(inputSize= 2*int(envs.observation_space.shape[0]),
                                      outputSize=args.time_intervals,
                                      num_fc_layers=int(args.num_layers),
                                      depth_fc_layers=int(args.fc_width),
                                      lr=float(args.opt_lr),
                                      buffer_max_length = args.buffer_max_length,
                                      buffer_RL_ratio=args.buffer_RL_ratio,
                                      frame_skip=frame_skip,
                                      tdm_epoch=args.tdm_epoch,
                                      tdm_batchsize=args.tdm_batchsize,
                                      logger=logger,
                                      bonus_func=bonus_func).to(device)

        #collect random trajectories
        sample_collector = CollectSamples(envs, args.num_processes, initial=True)
        tdm.buffer_rand = sample_collector.collect_trajectories(args.num_rollouts, args.steps_per_rollout)

        # initial training
        tdm.update()

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)
    start = time.time()
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        # acting
        for step in range(args.num_steps):

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])


            # Obser reward and next obs
            # envs.render()

            obs_old = obs.clone()
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            #compute intrinsic bonus
            if args.use_tdm:
                tdm.symm_eval = True if step == args.num_steps-1 else False
                reward_int = tdm.compute_bonus(obs_old, obs)
                reward += beta_func(step + j*args.num_steps) * reward_int.cpu()

                if (j % args.log_interval == 0) and (step == args.num_steps-1):
                    logger.add_reward_intrinsic(reward_int, (j+1)*args.num_steps*args.num_processes)

            #saving to buffer.
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        # saving to buffer and periodic updating parameters
        if (args.use_tdm):
            tdm.buffer_RL_temp.append((rollouts.obs, rollouts.masks))
            if (j%args.num_steps==0 and j>0):
                tdm.update()

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)


        rollouts.after_update()



        # save for every interval-th episode or for the last epoch
        # no
        # save every 1-million steps
        if (((j+1)*args.num_steps*args.num_processes)%1e6 == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            if j == num_updates - 1:
                save_here = os.path.join(save_path, args.env_name + "_step_{}M.pt".format((j+1)*args.num_steps*args.num_processes//1e6))
            else:
                save_here = os.path.join(save_path, args.env_name + "_final.pt")
            torch.save(save_model, save_here) # saved policy.



        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        # printing outputs
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j,
                       total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))
            logger.add_reward(episode_rewards, (j+1)*args.num_steps*args.num_processes)

        #
        # if j % args.tb_interval == 0:
        #     # mean/std or median/1stqt?
        #     logger.add_tdm_loss(loss, self.epoch_count*i)




        # evaluation process
        # if (args.eval_interval is not None
        #         and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     eval_envs = make_vec_envs(
        #         args.env_name, args.seed + args.num_processes, args.num_processes,
        #         args.gamma, eval_log_dir, args.add_timestep, device, True)
        #
        #     vec_norm = get_vec_normalize(eval_envs)
        #     if vec_norm is not None:
        #         vec_norm.eval()
        #         vec_norm.ob_rms = get_vec_normalize(envs).ob_rms
        #
        #     eval_episode_rewards = []
        #
        #     obs = eval_envs.reset()
        #     eval_recurrent_hidden_states = torch.zeros(args.num_processes,
        #                     actor_critic.recurrent_hidden_state_size, device=device)
        #     eval_masks = torch.zeros(args.num_processes, 1, device=device)
        #
        #     while len(eval_episode_rewards) < 10:
        #         with torch.no_grad():
        #             _, action, _, eval_recurrent_hidden_states = actor_critic.act(
        #                 obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)
        #
        #         # Obser reward and next obs
        #         # envs.render()
        #         obs, reward, done, infos = eval_envs.step(action)
        #
        #         eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
        #                                         for done_ in done])
        #         for info in infos:
        #             if 'episode' in info.keys():
        #                 eval_episode_rewards.append(info['episode']['r'])
        #
        #     eval_envs.close()
        #
        #     print(" Evaluation using {} episodes: mean reward {:.5f}\n".
        #         format(len(eval_episode_rewards),
        #                np.mean(eval_episode_rewards)))


        # # plotting
        # if args.vis and j % args.vis_interval == 0:
        #     try:
        #         # Sometimes monitor doesn't properly flush the outputs
        #         win = visdom_plot(viz, win, args.log_dir, args.env_name,
        #                           args.algo, args.num_env_steps)
        #     except IOError:
        #         pass
    #if done save:::::::::::
    logger.save()


if __name__ == "__main__":
    main()
