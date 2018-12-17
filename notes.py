'''

ENVS:
    Classic Control:
        MountainCarContinuous-v0:       S=R2,  A=R1
        Pendulum-v0:                    S=R3,  A=R1

    Box2D:
        LunarLanderContinuous-v2:       S=R8,  A=R2 (Base Env)
        BipedalWalker-v2 :              S=R24, A=R4



4. different schedule for beta?
6. different bonus function?
5. label maker/time_intervals?
3. different rate for ratio?

2. performance comparison over beta = 10^-k, k = -4, …, 1
1. how many envs?


@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Exp schedule:
12/14
1. param ablations:

    '--tb_dir'

    '--beta_int': -3,-2,-1,0
    '--num_layers': 1,2
    '--fc_width': 200, 300
    '--opt_lr': 1e-3, 1e-4

    Good: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    Bad: 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32

    beta = 1e-4, 1e-3

    15, 16 slowest: beta 1e-2, nl2,fc300, optlr 1e-3,1e-4
    13, 14 less yet slow: beta 1e-2, nl 2,fc 200, optlr 1e-3,1e-4
    11     less yet slow: beta 1e-2, nl 1, fc 300, opt lr 1e-3
    fc 200 better?

    12 less fast : beta 1e-2, nl 1, fc 300, opt lr 1e-4
    opt lr 1e-4 better?



    best yet 1 beta 1e-3, nl 1, fc 200, opt lr 1e-3
    very similar 2 beta 1e-3, nl 1, fc 200, opt lr 1e-4

    better than 1 3, 4

    after one-by-one comparison...
    first 8 are the best ones

    nl 2: 5,6,7,8 better mean
    fc 300: 3,4,7,8 stabler? idk..

    best: 8 = beta 1e-3, nl 2, fc 300, lr 1e-4
        7 = lr 1e-3 stable but slower : 7 is better than 4 ==> nl 2 better
        6 = fc 200 faster but unstable
        4 = nl 1 stable but slower


    RESULT:::::
    params to fix:
    nl 2
    fc 300
    lr 1e-4

    params to try later:
    beta 1e-4, 1e-3
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Exp schedule:
12/15
2. beta decaying schedule and bonus func:
    with fixed :
        nl 2
        fc 300
        lr 1e-4
        tb_dir
    try:
        beta_func = const, linear, log, sqrt
        bonus_func = linear, log, sqrt, square
        try: beta = 10**np.arange(-3,3)

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Exp Schedule:
12/15
3. Harder Env: BipedalWalker-v2
param search same as #2
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Exp Schedule:
12/17
4. Not really trying new but trying to check whether we are really learning the metric: i.e. to check symmetricity
To do so: I will compute k-many more graphs: for each time prediction interval compute 1/2 * (|f(s,s') - f(s',s)|/|f(s,s')| + |f(s,s') - f(s',s)|/|f(s',s)|)
'''
#
# # make 0 first
# import numpy as np
# betas = 10.0**np.arange(-3,3)
# bfs = ['const', 'sqrt', 'log', 'linear']
# bnfs = ['linear', 'square', 'log', 'sqrt']
#
# for beta in betas:
#     for bf in bfs:
#         for bnf in bnfs:
#             print("srun python main.py --env-name 'LunarLanderContinuous-v2' --use_tdm True --beta_int {} --num_layers 2 --fc_width 300 --opt_lr 1e-4 --beta_schedule {} --bonus_func {}".format(beta, bf, bnf))
