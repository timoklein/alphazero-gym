#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha Zero
@author: Thomas Moerland, Delft University of Technology
"""
import copy
import torch
import numpy as np


from .helpers import copy_atari_state, restore_atari_state, stable_normalizer, argmax


##### MCTS functions #####
class Action:
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


# TODO: Put these methods into the MCTS for clarity and modularity
class Node:
    """ Node object """

    def __init__(self, state, r, terminal, parent_action, num_actions, model):
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
            Action(a, parent_node=self, Q_init=self.V) for a in range(num_actions)
        ]
        state = torch.from_numpy(state[None,]).float()
        self.priors = model.predict_pi(state).flatten()

    def evaluate(self, model):
        """ Bootstrap the state value """
        state = torch.from_numpy(self.state[None,]).float()
        self.V = (
            np.squeeze(model.predict_V(state)) if not self.terminal else np.array(0.0)
        )

    def update_visit_counts(self):
        """ update count on backward pass """
        self.n += 1


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
            node = self.root_node  # reset to root for new trace

            if not self.is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env, snapshot)

            while not node.terminal:
                action = self.selectionUCT(node, self.c_uct)

                # take step
                new_state, reward, terminal, _ = mcts_env.step(action.index)

                if hasattr(action, "child_node"):
                    node = self.selection(action)
                    continue
                else:
                    node = self.expansion(action, new_state, reward, terminal)  # expand
                    break

            self.backprop(node, self.gamma)

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
        winner = argmax(UCT)
        return node.child_actions[winner]

    def selectionUniform(node):
        pass

    @staticmethod
    def selection(action):
        return action.child_node

    def expansion(self, action, state, reward, terminal):
        node = action.add_child_node(state, reward, terminal, self.model)
        return node

    @staticmethod
    def simulation(node, n_rollouts):
        """AlphaZero doesn't simulate but uses the value estimate from the model instead.
        """
        pass

    @staticmethod
    def backprop(node, gamma):
        R = node.V
        while node.parent_action is not None:  # loop back-up until root is reached
            R = node.r + gamma * R
            action = node.parent_action
            action.update(R)
            node = action.parent_node
            node.update_visit_counts()

    def return_results(self, temp):
        """ Process the output at the root node """
        counts = np.array(
            [child_action.n for child_action in self.root_node.child_actions]
        )
        Q = np.array([child_action.Q for child_action in self.root_node.child_actions])
        pi_target = stable_normalizer(counts, temp)
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

