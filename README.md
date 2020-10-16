# What is this
This repo contains multiple, self-contained implementations of AlphaZero. 
They are based on this repo https://github.com/tmoer/alphazero_singleplayer
as well as the following paper [A0C: Alpha Zero in Continuous Action Space](https://arxiv.org/pdf/1805.09613.pdf)

## master branch

## tf branch
Base implementation.  

## TDOS
- [ ] Add  model saving and loading using the config for instantiation.  
- [ ] Add mixed precision training.  
- [ ] Add pandas exporter for wandb/hydra data.  


## Possible Enhancements to try
- [ ] Implement stochastic weight averaging.  
- [ ] Test a prioritized replay buffer.  
- [ ] Warm starting.  
- [ ] RAD style data augmentations.  
- [ ] Playout cap randomization.  
- [ ] Implement e-greedy and test it vs entropy regularization.  
- [ ] Test BatchNorm.  
- [ ] Implement improved Q-loss.
- [ ] Different optimizers.
