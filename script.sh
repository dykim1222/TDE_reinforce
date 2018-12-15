#!/bin/bash

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

# assume i am at ~/code/tmd/12_15_func


count=1
for beta in 0.001 0.01 0.1 1.0 10.0; do
  for bf in "const" "sqrt" "log" "linear"; do
    for bnf in "square" "sqrt" "log" "linear"; do
      mkdir $count
      cp sb.s $count
      cd $count
      git clone https://github.com/dykim1222/mlproject.git
      cp sb.s mlproject
      cd mlproject
      echo "srun python main.py --env-name 'LunarLanderContinuous-v2' --use_tdm True --beta_int ${beta} --num_layers 2 --fc_width 300 --opt_lr 1e-4 --beta_schedule ${bf} --bonus_func ${bnf}" >> sb.s
      cd ~/code/tmd/12_15_func
      count = $((count+1))
    done;
  done;
done;
