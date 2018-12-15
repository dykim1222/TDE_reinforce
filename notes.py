'''

ENVS:
    Classic Control:
        MountainCarContinuous-v0:       S=R2,  A=R1
        Pendulum-v0:                    S=R3,  A=R1  (no Done)

    Box2D:
        LunarLanderContinuous-v2:       S=R8,  A=R2
        BipedalWalker-v2 :              S=R24, A=R4


python main.py --env-name "LunarLanderContinuous-v2" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True
--tb_dir 1




1. how many envs?
2. param search over beta = 10^-k, k = -4, …, 1

3. different rate for ratio?
4. different schedule for beta?
5. label maker?
6. different bonus function?
7. time interval?


@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Exp schedule:
12/14
1. param ablations:

'--tb_dir'

'--beta_int': -3,-2,-1,0
'--num_layers': 1,2
'--fc_width': 200, 300
'--opt_lr': 1e-3, 1e-4
'''
'''

import numpy as np
i=1
beta_int = 10.0**np.arange(-3,1)
num_layers = [1,2]
fc_width = [200,300]
opt_lr = [1e-3,1e-4]

for beta in beta_int:
    for nl in num_layers:
        for fc in fc_width:
            for lr in opt_lr:
                print("python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir {} --beta_int {} --num_layers {} --fc_width {} --opt_lr {}".format(i, beta, nl, fc, lr))
                i += 1
'''




srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 1 --beta_int 0.001 --num_layers 1 --fc_width 200 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 2 --beta_int 0.001 --num_layers 1 --fc_width 200 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 3 --beta_int 0.001 --num_layers 1 --fc_width 300 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 4 --beta_int 0.001 --num_layers 1 --fc_width 300 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 5 --beta_int 0.001 --num_layers 2 --fc_width 200 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 6 --beta_int 0.001 --num_layers 2 --fc_width 200 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 7 --beta_int 0.001 --num_layers 2 --fc_width 300 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 8 --beta_int 0.001 --num_layers 2 --fc_width 300 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 9 --beta_int 0.01 --num_layers 1 --fc_width 200 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 10 --beta_int 0.01 --num_layers 1 --fc_width 200 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 11 --beta_int 0.01 --num_layers 1 --fc_width 300 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 12 --beta_int 0.01 --num_layers 1 --fc_width 300 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 13 --beta_int 0.01 --num_layers 2 --fc_width 200 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 14 --beta_int 0.01 --num_layers 2 --fc_width 200 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 15 --beta_int 0.01 --num_layers 2 --fc_width 300 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 16 --beta_int 0.01 --num_layers 2 --fc_width 300 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 17 --beta_int 0.1 --num_layers 1 --fc_width 200 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 18 --beta_int 0.1 --num_layers 1 --fc_width 200 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 19 --beta_int 0.1 --num_layers 1 --fc_width 300 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 20 --beta_int 0.1 --num_layers 1 --fc_width 300 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 21 --beta_int 0.1 --num_layers 2 --fc_width 200 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 22 --beta_int 0.1 --num_layers 2 --fc_width 200 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 23 --beta_int 0.1 --num_layers 2 --fc_width 300 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 24 --beta_int 0.1 --num_layers 2 --fc_width 300 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 25 --beta_int 1.0 --num_layers 1 --fc_width 200 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 26 --beta_int 1.0 --num_layers 1 --fc_width 200 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 27 --beta_int 1.0 --num_layers 1 --fc_width 300 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 28 --beta_int 1.0 --num_layers 1 --fc_width 300 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 29 --beta_int 1.0 --num_layers 2 --fc_width 200 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 30 --beta_int 1.0 --num_layers 2 --fc_width 200 --opt_lr 0.0001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 31 --beta_int 1.0 --num_layers 2 --fc_width 300 --opt_lr 0.001
srun python main.py --env-name 'LunarLanderContinuous-v2' --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --use_tdm True --tb_dir 32 --beta_int 1.0 --num_layers 2 --fc_width 300 --opt_lr 0.0001
