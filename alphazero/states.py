import numpy as np
import torch

class ActionDiscrete:
    """ Action object """

    def __init__(self, index, parent_node, Q_init=0.0):
        self.index = index
        self.parent_node = parent_node
        self.W = 0.0
        self.n = 0
        self.Q = Q_init

    def add_child_node(self, state, r, terminal, model):
        self.child_node = Node(
            state, r, terminal, self, self.parent_node.num_actions, model
        )
        return self.child_node

    def update(self, R):
        self.n += 1
        self.W += R
        self.Q = self.W / self.n

class Node:
    """ Node object """

    def __init__(self, state, r, terminal, parent_action, num_actions, model=None):
        """ Initialize a new node """
        self.state = state  # game state
        self.r = r  # reward upon arriving in this node
        self.terminal = terminal  # whether the domain terminated in this node
        self.parent_action = parent_action
        self.n = 0
        
        self.evaluate(model)
        # Child actions
        self.num_actions = num_actions
        self.child_actions = [
            ActionDiscrete(a, parent_node=self, Q_init=self.V) for a in range(num_actions)
        ]
        state = torch.from_numpy(state[None,]).float()
        self.priors = model.predict_pi(state).flatten()

    def evaluate_nn(self, model):
        """ Bootstrap the state value """
        state = torch.from_numpy(self.state[None,]).float()
        self.V = (
            np.squeeze(model.predict_V(state)) if not self.terminal else np.array(0.0)
        )

    def evaluate_simulation(self, mcts_env):
        """Performs uniform random rollouts.
        """
        self.V = 0
        terminal = False
        while not terminal:
            action = mcts_env.action_space.sample()
            _, reward, terminal, _ = mcts_env.step(action)
            self.V += reward

    def update_visit_counts(self):
        """ update count on backward pass """
        self.n += 1