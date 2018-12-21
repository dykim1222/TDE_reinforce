# Curiosity-Driven Reinforcement Learning Through Temporal Distance

## Foundations of Machine Learning 2018 Project by Dae Young Kim

### PPO implementation code imported from [PyTorch PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr).

### My modifications/writings are only in: 
  - `README.md`
  - `main.py`
  - `mymodels.py`
  - `plottheresults.py`
  - `./a2c_ppo_acktr/arguments.py`
  
  
### Requirements:
  - PyTorch
  - TensorFlow
  - [Open AI Gym](https://github.com/openai/gym)
  - pybullet
  

### To run:
```
python main.py --env-name 'BipedalWalkerHardcore-v2' --use_tdm True --beta_int 10.0 --num_layers 2 --fc_width 300 --opt_lr 1e-4 --beta_schedule linear --bonus_func log
```

### Plots:
![rewardbipedalworkerhardcore](https://github.com/dykim1222/mlproject/blob/master/plots/rewardbp.png)

