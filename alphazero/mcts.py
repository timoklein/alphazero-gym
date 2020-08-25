#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha Zero
@author: Thomas Moerland, Delft University of Technology
"""
import copy
from typing import Tuple
import gym
import torch
import numpy as np
from abc import ABC, abstractmethod

from .states import ActionDiscrete, NodeDiscrete
from .helpers import copy_atari_state, restore_atari_state, stable_normalizer, argmax


class MCTS(ABC):
    @abstractmethod
    def selection(self):
        raise NotImplementedError

    @abstractmethod
    def expansion(self):
        raise NotImplementedError

    @abstractmethod
    def evaluation(self):
        raise NotImplementedError

    @abstractmethod
    def backprop(self):
        raise NotImplementedError

    @abstractmethod
    def search(self):
        raise NotImplementedError


class NNMCTSDiscrete(MCTS):
    """ MCTS object """

    def __init__(
        self,
        model: torch.nn.Module,
        num_actions: int,
        is_atari: bool,
        c_uct: float,
        gamma: float,
        root_state: np.array,
        root=None,
    ):
        self.root_node = root
        self.root_state = root_state
        self.model = model
        self.num_actions = num_actions
        self.c_uct = c_uct
        self.gamma = gamma

        self.is_atari = is_atari

    def initialize_search(self) -> None:
        if self.root_node is None:
            self.root_node = NodeDiscrete(
                self.root_state,
                r=0.0,
                terminal=False,
                parent_action=None,
                num_actions=self.num_actions,
            )
            self.evaluation(self.root_node)
        else:
            self.root_node.parent_action = None  # continue from current root
        if self.root_node.terminal:
            raise ValueError("Can't do tree search from a terminal node")

        if self.is_atari:
            snapshot = copy_atari_state(
                Env
            )  # for Atari: snapshot the root at the beginning

    def evaluation(self, node: NodeDiscrete) -> None:
        state = torch.from_numpy(node.state[None,]).float()
        node.V = (
            np.squeeze(self.model.predict_V(state))
            if not node.terminal
            else np.array(0.0)
        )
        node.child_actions = [
            ActionDiscrete(a, parent_node=node, Q_init=node.V)
            for a in range(node.num_actions)
        ]
        node.priors = self.model.predict_pi(state).flatten()

    def search(self, n_traces: int, Env: gym.Env, mcts_env: gym.Env):
        """ Perform the MCTS search from the root """

        self.initialize_search()

        for i in range(n_traces):
            node = self.root_node  # reset to root for new trace

            if not self.is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env, snapshot)

            while not node.terminal:
                action = self.selectionUCT(self.c_uct, node)

                # take step
                new_state, reward, terminal, _ = mcts_env.step(action.index)

                if getattr(action, "child_node"):
                    node = self.selection(action)
                    continue
                else:
                    node = self.expansion(action, new_state, reward, terminal)  # expand
                    break

            self.backprop(node, self.gamma)

    @staticmethod
    def selectionUCT(c_uct, node: NodeDiscrete) -> ActionDiscrete:
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

    @staticmethod
    def selection(action: ActionDiscrete) -> NodeDiscrete:
        return action.child_node

    def expansion(
        self, action: ActionDiscrete, state: np.array, reward: float, terminal: bool
    ) -> NodeDiscrete:
        node = action.add_child_node(state, reward, terminal)
        self.evaluation(node)
        return node

    @staticmethod
    def backprop(node: NodeDiscrete, gamma: float):
        R = node.V
        while node.parent_action is not None:  # loop back-up until root is reached
            R = node.r + gamma * R
            action = node.parent_action
            action.update(R)
            node = action.parent_node
            node.update_visit_counts()

    def return_results(self, temperature: float) -> Tuple[np.array, np.array, np.array]:
        """ Process the output at the root node """
        counts = np.array(
            [child_action.n for child_action in self.root_node.child_actions]
        )
        Q = np.array([child_action.Q for child_action in self.root_node.child_actions])
        pi_target = stable_normalizer(counts, temperature)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root_node.state, pi_target, V_target

    def forward(self, action: int, state: np.array) -> None:
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
