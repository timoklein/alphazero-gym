# What is this
This repo contains multiple, self-contained implementations of AlphaZero. 
They are based on this repo https://github.com/tmoer/alphazero_singleplayer
as well as the following paper [A0C: Alpha Zero in Continuous Action Space](https://arxiv.org/pdf/1805.09613.pdf)

## master branch

## tf branch
Base implementation.  

## TODOS
- [ ] Implement stochastic weight averaging.
- [ ] Add Gaussian Mixture option to continuous policy network.
- [ ] Add  model saving and loading using the config for instantiation.


## Possible Enhancements to try
- [x] Automatic entropy tuning a la SAC
- [x] Implement different V_target options
- [ ] Test a prioritized replay buffer.
- [ ] Warm starting.
- [ ] RAD style data augmentations.
- [ ] Playout cap randomization.
