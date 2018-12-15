'''

ENVS:
    Classic Control:
        MountainCarContinuous-v0:       S=R2,  A=R1
        Pendulum-v0:                    S=R3,  A=R1  (no Done)

    Box2D:
        LunarLanderContinuous-v2:       S=R8,  A=R2
        BipedalWalker-v2 :              S=R24, A=R4


python main.py --env-name "LunarLanderContinuous-v2" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01




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
'--opt_lr': 2  1e-3, 1e-4
'''
