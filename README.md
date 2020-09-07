# What is this
This repo contains multiple, self-contained implementations of AlphaZero. 
They are based on this repo https://github.com/tmoer/alphazero_singleplayer
as well as the following paper [A0C: Alpha Zero in Continuous Action Space](https://arxiv.org/pdf/1805.09613.pdf)

## master branch

## tf branch
Base implementation.  

## TODOS
- [x] Verify that this works with tensorflow (in a virtualenv).
- [x] Port models to PyTorch and test.
- [x] Implement tensorboard visualization.
- [x] Implement Agent abstraction.
- [x] Create branch for current base version.
- [x] Create proper policy methods for MCTS.
- [x] restructure project.
- [x] Implement model saving and loading.
- [x] Implement more detailed logging.
- [x] Decouple node and mcts.
- [x] Implement Simulation policy.
- [x] Implement MCTS progressive widening.
- [x] Implement A0C.
- [x] Use Pendulum to test (might need some custom wrappers).
- [x] Factor out loss objects and use torch Categorical in Discrete Net
- [x] Refactor discrete version to always return actions as well
- [x] Refactor buffer into more efficient version
- [x] Remove the argparse crap and use Hydra instead
- [ ] Enable GPU training.


## Possible Enhancements to try
- [x] Automatic entropy tuning a la SAC
- [x] Implement different V_target options
- [ ] Test a prioritized replay buffer.
- [ ] Warm starting.
- [ ] RAD style data augmentations.
- [ ] Playout cap randomization.
