# What is this
This repo contains multiple, self-contained implementations of AlphaZero. 
They are based on this repo https://github.com/tmoer/alphazero_singleplayer
as well as the following paper [A0C: Alpha Zero in Continuous Action Space](https://arxiv.org/pdf/1805.09613.pdf)

## master branch

## tf branch
Base implementation.  

## TDOS
- [ ] Add  model saving and loading using the config for instantiation.  
- [ ] Add new policy classes.  
- [ ] Fix WandB logging.  
- [ ] Make an installable package.  
- [ ] Add typing.  
- [ ] Fix Hydra instantiation.   
- [ ] Docs. 
- [ ] Delete tf branch.  


## Possible Enhancements to try
- [ ] Implement stochastic weight averaging.  
- [ ] Test a prioritized replay buffer.  
- [ ] Warm starting.  
- [ ] RAD style data augmentations.  
- [ ] Playout cap randomization.  
- [ ] Test L2 reg (BatchNorm makes no sense for multimodal data).  
- [ ] Subtract state dependent baseline in loss (mean/median of counts).
