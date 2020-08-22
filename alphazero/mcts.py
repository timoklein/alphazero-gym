import numpy as np
import copy

from .states import Node
from .helpers import copy_atari_state, restore_atari_state, stable_normalizer, argmax

class MCTS:
    """ MCTS object """

    def __init__(
        self, model, num_actions, is_atari, c_uct, gamma, root_state, root=None
    ):
        self.root_node = root
        self.root_state = root_state
        self.model = model
        self.num_actions = num_actions
        self.c_uct = c_uct
        self.gamma = gamma

        self.is_atari = is_atari

    def initialize_search(self):
        if self.root_node is None:
            self.root_node = Node(
                self.root_state,
                r=0.0,
                terminal=False,
                parent_action=None,
                num_actions=self.num_actions,
                model=self.model,
            )  # initialize new root
        else:
            self.root_node.parent_action = None  # continue from current root
        if self.root_node.terminal:
            raise ValueError("Can't do tree search from a terminal node")

        if self.is_atari:
            snapshot = copy_atari_state(
                Env
            )  # for Atari: snapshot the root at the beginning

    def search(self, n_traces, Env, mcts_env):
        """ Perform the MCTS search from the root """

        self.initialize_search()

        for i in range(n_traces):
            # reset to root for new trace
            node = self.root_node 

            if not self.is_atari:
                # copy original Env to rollout from
                # TODO: Can we speed this up?
                mcts_env = copy.deepcopy(Env) 
            else:
                restore_atari_state(mcts_env, snapshot)

            while not node.terminal:
                action = self.selectionUCT(node, self.c_uct)

                # take step
                new_state, reward, terminal, _ = mcts_env.step(action.index)

                if hasattr(action, "child_node"):
                    # selection
                    node = action.child_node
                    continue
                else:
                    # expansion
                    node = action.add_child_node(new_state, reward, terminal, self.model)
                    break

            self.backprop(node, self.gamma)

    # TODO: Slowest method... How can we speed this up?
    @staticmethod
    def selectionUCT(node, c_uct):
        """ Select one of the child actions based on UCT rule """
        UCT = np.array(
            [
                child_action.Q
                + prior * c_uct * (np.sqrt(node.n + 1) / (child_action.n + 1))
                for child_action, prior in zip(node.child_actions, node.priors)
            ]
        )
        max_a = argmax(UCT)
        return node.child_actions[max_a]

    @staticmethod
    def backprop(node, gamma):
        R = node.V
        while node.parent_action is not None:  # loop back-up until root is reached
            R = node.r + gamma * R
            action = node.parent_action
            action.update(R)
            node = action.parent_node
            node.update_visit_counts()

    def return_results(self, temperature):
        """ Process the output at the root node """
        counts = np.array(
            [child_action.n for child_action in self.root_node.child_actions]
        )
        Q = np.array([child_action.Q for child_action in self.root_node.child_actions])
        pi_target = stable_normalizer(counts, temperature)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root_node.state, pi_target, V_target

    def forward(self, action, state):
        """ Move the root forward """
        if not hasattr(self.root_node.child_actions[action], "child_node"):
            self.root_node = None
            self.root_state = state
        elif (
            np.linalg.norm(
                self.root_node.child_actions[action].child_node.state - state
            )
            > 0.01
        ):
            print(
                "Warning: this domain seems stochastic. Not re-using the subtree for next search. "
                + "To deal with stochastic environments, implement progressive widening."
            )
            self.root_node = None
            self.root_state = state
        else:
            self.root_node = self.root_node.child_actions[action].child_node