# Discretized A0C

This is a *VERY HACKY* implementation of the ordinal discretization scheme
 mentioned in
 [Discretizing Continuous Action Space for On-Policy Optimization](https://arxiv.org/pdf/1901.10500.pdf).

Notes:

- The code works and runs. Performance should be slightly improved over continuous control.  
- There are several hardcoded values in the MCTS and Policy for the number of bins.  
- I won't integrate this into the main branch unless there is a specific reason for me to work with this codebase again.  
