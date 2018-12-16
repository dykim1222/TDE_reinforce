#!/bin/bash

count=1
for beta in 0.001 0.01 0.1 1.0 10.0 100.0; do
  for bf in "const" "sqrt" "log" "linear"; do
    for bnf in "square" "sqrt" "log" "linear"; do
      mkdir $count
      cp sb.s $count
      cd $count
      git clone https://github.com/dykim1222/mlproject.git
      cp sb.s mlproject
      cd mlproject
      echo "srun python main.py --env-name 'BipedalWalkerHardcore-v2' --num-env-steps 1e7 --use_tdm True --beta_int ${beta} --num_layers 2 --fc_width 300 --opt_lr 1e-4 --beta_schedule ${bf} --bonus_func ${bnf}" >> sb.s
      sbatch sb.s
      cd /home/kimdy/code/tmd/12_16_bipedalhardcore
      count=$((count+1))
done; done; done;

# changed last cd?
# run 0 no_tdm?
