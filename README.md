# What is this
This repo contains multiple, self-contained implementations of AlphaZero. 

* **tf**   
    Base implementation from https://github.com/tmoer/alphazero_singleplayer.  
* **alphazero_discrete**  
    PyTorch port with some code restructuring.  
* **master**  
    Implementation of [A0C: Alpha Zero in Continuous Action Space](https://arxiv.org/pdf/1805.09613.pdf).


## TODOS
- [x] Verify that this works with tensorflow (in a virtualenv).
- [x] Port models to PyTorch and test.
- [x] Implement tensorboard visualization.
- [x] Implement Agent abstraction.
- [x] Create branch for current base version.
- [x] Create proper policy methods for MCTS.
- [ ] Implement MCTS progressive widening.
- [ ] Implement A0C.
- [ ] Enable GPU training.
- [ ] Use Pendulum to test (might need some custom wrappers).


## Possible Enhancements to try
- [ ] Maybe parallelize data generation.
- [ ] Test a prioritized replay buffer.
- [ ] Try normalizing flow policy.
- [ ] Warm starting.
- [ ] Playout cap randomization.
- [ ] RAD style data augmentations.
