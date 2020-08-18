# What is this

Reimplementation of the paper [A0C: Alpha Zero in Continuous Action Space](https://arxiv.org/pdf/1805.09613.pdf) in PyTorch.

The code of this implementation is based on the single player AlphaZero repo of the author https://tmoer.github.io/AlphaZero/.

## TODOS
- [x] Verify that this works with tensorflow (in a virtualenv).
- [x] Port models to PyTorch and test.
- [x] Implement tensorboard visualization.
- [x] Implement Agent abstraction.
- [ ] Implement MCTS progressive widening.
- [ ] Implement A0C.
- [ ] Enable GPU training.
- [ ] Use Pendulum to test (might need some custom wrappers).
- [ ] Try normalizing flow policy.
