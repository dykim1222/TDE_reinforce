import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import pdb
import time
import copy
from collections import deque
from baselines.bench import Monitor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tensorflow as tf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class layer(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(layer, self).__init__()
        self.fc = nn.Linear(inputSize, outputSize)
        init.xavier_normal_(self.fc.weight)

    def forward(self, input):
        return F.relu(self.fc(input))

class layer_last(nn.Module):  # no ReLU in the last layer
    def __init__(self, inputSize, outputSize):
        super(layer_last, self).__init__()
        self.fc = nn.Linear(inputSize, outputSize)
        init.xavier_normal_(self.fc.weight)

    def forward(self, input):
        return self.fc(input)


class TemporalDifferenceModule(nn.Module):
    def __init__(self, inputSize, outputSize, num_fc_layers, depth_fc_layers,
                lr, buffer_max_length, buffer_RL_ratio, frame_skip,
                tdm_epoch, tdm_batchsize, logger, bonus_func):
        super(TemporalDifferenceModule, self).__init__()
        self.time_intervals = outputSize
        self.num_fc_layers = num_fc_layers
        self.layer0 = layer(inputSize, depth_fc_layers).to(device)
        for i in range(num_fc_layers-1):
            module = layer(depth_fc_layers, depth_fc_layers).to(device)
            self.add_module('layer{}'.format(i+1), module)
        self.layer_last = layer_last(depth_fc_layers, outputSize).to(device)
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.buffer_rand = deque(maxlen=buffer_max_length)
        self.buffer_max_length = buffer_max_length
        self.buffer_RL_temp = deque(maxlen=buffer_max_length)
        self.buffer_RL = deque(maxlen=buffer_max_length)
        self.buffer_RL_ratio = buffer_RL_ratio
        self.tdm_epoch = tdm_epoch
        self.tdm_batchsize = tdm_batchsize
        self.buffer_rand_clean = False
        self.frame_skip=frame_skip
        self.logger= logger
        self.epoch_count = 0
        self.bonus_func = bonus_func
        self.symm_eval = False
        self.symm_eval_count = 0


    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.num_fc_layers-1):
            cur = self._modules['layer{}'.format(i+1)](cur)
        out = self.layer_last(cur)
        return out

    def compute_bonus(self, obs_old, obs):
        # compute intrinsic reward
        input = torch.cat([obs_old, obs],dim=1)
        input = self.forward(input)
        input = torch.argmax(input, dim=1)

        if self.symm_eval: # to check symmetricity
            symm = torch.cat([obs, obs_old], dim=1)
            symm = self.forward(symm)
            symm = torch.argmax(symm, dim=1)
            delta = torch.abs(symm-input)
            val = (delta/(torch.abs(input)+1) + delta/(torch.abs(symm)+1))/2
            val = torch.mean(val.float())
            self.logger.add_symm_eval(val, self.symm_eval_count)
            self.symm_eval_count += 1

        input = self.bonus_func(input)
        return input.to(device)

    def label_maker(self, number):
        label = torch.zeros(self.time_intervals)
        if number <= 4:
            label[0] = 1
        elif number <= 12:
            label[1] = 1
        elif number <= 24:
            label[2] = 1
        elif number <= 36:
            label[3] = 1
        elif number <= 54:
            label[4] = 1
        else:
            label[5] = 1
        return label

    def clean(self, raw):
        append_here = deque(maxlen=10000)
        num_processes = raw[0][0].shape[1]
        for i in range(len(raw)):
            for j in range(num_processes):
                traj = raw[i][0][:,j] # shape: [epi_length, state_dim]
                mask = raw[i][1][:,j]

                if mask.sum().item() < raw[i][0].shape[0]: # if there is a break
                    break_location = np.where(mask.cpu()==0)[0]
                    break_location = np.insert(break_location,0,0)
                    break_location = np.insert(break_location,len(break_location),len(break_location))
                    for k in range(len(break_location)-1):
                        traj_curr = traj[break_location[k]:break_location[k+1],:]
                        if traj_curr.shape[0] != 0:
                            append_here.append(traj_curr)
        return append_here

    def frame_skipping(self, cleaned_buffer):
        # frame skip
        skipped_buffer = deque(maxlen=10000)
        for i in range(len(cleaned_buffer)):
            skipped_traj = cleaned_buffer[i][0].unsqueeze(0)
            for j in range(cleaned_buffer[i].shape[0]):
                if j%self.frame_skip == 0 and j>0:
                    skipped_traj = torch.cat((skipped_traj, cleaned_buffer[i][j].unsqueeze(0)))
            skipped_buffer.append(skipped_traj)
        return skipped_buffer

    def preprocess_trajectories(self):
        # clean raw traj data with masks
        if self.buffer_rand_clean:
            # clean buffer_RL_temp and put it in the buffer_RL
            self.buffer_RL = self.buffer_RL + self.frame_skipping(self.clean(self.buffer_RL_temp))
            self.buffer_RL_temp = deque(maxlen=10000)
        else:
            # after frame skip overwrite to bufer_rand
            self.buffer_rand = self.frame_skipping(self.clean(self.buffer_rand))
            self.buffer_rand_clean = True

    def standardize_data(self, data):
        mean = data.mean(0)
        std = data.std(0)
        data = (data-mean)/std
        data[data != data] = 0
        return data

    def sample_data(self): # check this again
        state_dim = self.buffer_rand[0].shape[1]  #check this again
        batch = torch.zeros(self.tdm_batchsize, 2*state_dim).to(device)
        labels = torch.zeros(self.tdm_batchsize, self.time_intervals).to(device)
        for i in range(self.tdm_batchsize):
            eps = 0.0 if len(self.buffer_RL)==0 else torch.rand(1)
            if eps <= 1-self.buffer_RL_ratio:
                traj = self.buffer_rand[torch.randint(len(self.buffer_rand),(1,))]
            else:
                traj = self.buffer_RL[torch.randint(len(self.buffer_RL),(1,))]

            two_states_idx = torch.randint(len(traj),(2,))
            two_states_idx = torch.sort(two_states_idx)[0]
            sample = torch.cat((traj[two_states_idx[0]],traj[two_states_idx[1]]))
            label = self.label_maker((two_states_idx[1]-two_states_idx[0])*4)
            batch[i] = sample
            labels[i] = label

        batch = self.standardize_data(batch)
        idx = torch.randperm(batch.shape[0])
        batch = batch[idx]
        labels = labels[idx]
        return batch.to(device), labels.to(device)

    def update(self):

        self.preprocess_trajectories()
        for i in range(self.tdm_epoch):
            batch, labels = self.sample_data()
            pred = self.forward(batch)
            loss = self.criterion(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 40.0)
            self.optimizer.step()
            self.logger.add_tdm_loss(loss, self.epoch_count*self.tdm_epoch + i)
        self.epoch_count += 1
        print('Optimization done.')


class CollectSamples(object):

    def __init__(self, env, num_processes, policy=None, initial=False):
        self.env = env
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.num_processes = num_processes
        self.action_shape = (num_processes, env.action_space.shape[0])
        if policy is not None:
            self.policy = policy
        self.initial = initial

    def random_policy(self):
        return torch.Tensor(npr.uniform(self.low, self.high, self.action_shape)).to(device)

    def collect_trajectories(self, num_rollouts, steps_per_rollout):
        buffer = deque(maxlen=10000)

        for rollout_number in range(num_rollouts):
            traj = (self.env.reset()).unsqueeze(0)
            mask= torch.ones(1, self.num_processes, 1).to(device)
            for step in range(steps_per_rollout):
                # self.env.render()
                obs, _, done, _ = self.env.step(self.random_policy())
                obs = obs.unsqueeze(0)
                traj = torch.cat((traj,obs))
                indic = (torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])).unsqueeze(0)
                mask = torch.cat((mask.to(device),indic.to(device)))
                if (done.sum()==self.num_processes) or  (step == steps_per_rollout-1):
                    if rollout_number%5==0:
                        print("Trajectory {:2}/{:2} collected.".format(rollout_number,num_rollouts))
                    buffer.append((traj,mask))
                    break
        print('Initial Roll outs are buffered successfully.')
        return buffer





dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def errorfill(x, y, yerr, label, color=None, marker = 'o', alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    base_line, = ax.plot(x, y, color=color,marker = 'o',label = label)
    if color is None:
        color = base_line.get_color()
    ax.fill_between(x, ymax, ymin, facecolor=color, alpha=alpha_fill)



class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = tf.summary.FileWriter(log_dir)
        self.loss_tdm = []
        self.mean = []
        self.median = []
        self.std = []
        self.qtl = []
        self.rw_int_mean = []
        self.rw_int_std = []
        self.symm_eval=[]

    def write_settings(self, args):
        with open(self.log_dir+"experiment_settings.txt", "w") as f:
            for arg in vars(args):
                param = str(arg)
                value = getattr(args, arg)
                f.write('Param: {:<25} Value: {:<10}\n'.format(param, value))
                # self.args[param] = value

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def add_tdm_loss(self, loss, iter_count):
        self.loss_tdm.append(loss.item())
        self.scalar_summary('loss', loss.item(), iter_count+1)

    def add_reward(self, reward, iter_count):
        self.mean.append(np.mean(reward))
        self.median.append(np.median(reward))
        self.std.append(np.std(reward))
        self.qtl.append(np.array([np.percentile(reward, 25),np.percentile(reward, 75)]))

        self.scalar_summary('tdm_mean', self.mean[-1], iter_count+1)
        self.scalar_summary('tdm_median', self.median[-1], iter_count+1)
        self.scalar_summary('tdm_std', self.std[-1], iter_count+1)
        # self.scalar_summary('tdm_qtl', np.mean(self.qtl[-1]), iter_count+1)


    def add_reward_intrinsic(self, reward_int, iter_count):
        self.rw_int_mean.append(reward_int.squeeze().mean().item())
        self.rw_int_std.append(reward_int.squeeze().std().item())
        self.scalar_summary('intrinsic_reward_mean', self.rw_int_mean[-1], iter_count+1)
        self.scalar_summary('intrinsic_reward_std', self.rw_int_std[-1], iter_count+1)

    def add_symm_eval(self, val, iter_count):
        self.symm_eval.append(val.item())
        self.scalar_summary('symmetricity', val.item(), iter_count+1)

    def save(self):
        filename = 'results'
        path = os.path.join(self.log_dir, filename)
        np.savez(path,
                 loss = np.array(self.loss_tdm),
                 mean = np.array(self.mean),
                 median = np.array(self.median),
                 std = np.array(self.std),
                 qtl = np.array(self.qtl),
                 rw_int_mean = np.array(self.rw_int_mean),
                 rw_int_std = np.array(self.rw_int_std),
                 symm_eval = np.array(self.symm_eval)
                 )
        print('Results Saved.')


if __name__ == "__main__":
    func = lambda t: t**2
    nn = TemporalDifferenceModule(10,5,2,100,1e-4, func) #working
    x = torch.rand(1,5)
    y = torch.rand(1,5)
    nn.compute_bonus(x,y)
