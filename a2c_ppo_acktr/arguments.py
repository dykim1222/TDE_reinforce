import argparse

import torch

# args.num_env_steps) // args.num_steps // args.num_processes
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    # parser.add_argument('--eval-interval', type=int, default=None,
    #                     help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-env-steps', type=int, default=1e6,
                        help='number of environment steps to train (default: 1e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    '''
    ###################################################################
    My Params
    ###################################################################
    '''
    parser.add_argument('--use_tdm', default=False,
                        help='use Temporal Difference Module')


    parser.add_argument('--tb_dir', default='./saved/',
                        help='logging directory path for tensorboard')


    # parser.add_argument('--tb_interval', default=100,
    #                     help='logging interval for tensorboard')
    parser.add_argument('--num_layers', default=1,
                        help='num layers for tdm')
    parser.add_argument('--fc_width', default=200,
                        help='width of hid layer of tdm')
    parser.add_argument('--opt_lr', default=1e-4,
                        help='optimizer lr for tdm')
    # parser.add_argument('--optimizer', default='adam',
    #                     help='which optimizer to use')    # rmsprop?
    parser.add_argument('--beta_int', default=1/16,
                        help='int reward param initial value')
    parser.add_argument('--time_intervals', default=6,
                        help='how many time interval predictions')
    parser.add_argument('--buffer_RL_ratio', default=0.9,
                        help='ratio fo bufferRL in the trin batch')
    parser.add_argument('--tdm_epoch', default=30,
                        help='tdm train epoch numb')
    parser.add_argument('--tdm_batchsize', default=512,
                        help='tdm training batch size')
    parser.add_argument('--buffer_max_length', default=1000,
                        help='buffer_RL max length (not too big so that we get better trajectories.)')
    parser.add_argument('--num_rollouts', default=40,
                        help='num_rollout for buffer_rand')
    parser.add_argument('--steps_per_rollout', default=200,
                        help='steps for buffer_rand')
    parser.add_argument('--beta_schedule', default='const',
                        help='beta scheduler')
    parser.add_argument('--bonus_func', default='linear',
                        help='how to give bonus')

    '''
    ###################################################################
    My Params
    ###################################################################
    '''

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
