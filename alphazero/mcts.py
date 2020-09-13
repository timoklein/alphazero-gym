import copy
from typing import Tuple
import gym
import torch
import numpy as np
from abc import ABC, abstractmethod

from .states import (
    Action,
    Node,
    ActionContinuous,
    ActionDiscrete,
    NodeContinuous,
    NodeDiscrete,
)
from .helpers import copy_atari_state, restore_atari_state, argmax

# scales the step reward between -1 and 0
PENDULUM_R_SCALE = 16.2736044


class MCTS(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        n_rollouts: int,
        c_uct: float,
        gamma: float,
        device: str,
        root_state: np.array,
        root=None,
    ) -> None:

        self.device = torch.device(device)
        self.root_node = root
        self.root_state = root_state
        self.model = model
        self.n_rollouts = n_rollouts
        self.c_uct = c_uct
        self.gamma = gamma

    @abstractmethod
    def selectionUCT(self):
        ...

    @abstractmethod
    def search(self):
        ...

    def get_softz_value_target(self, counts: np.array) -> np.array:
        Q = np.array([child_action.Q for child_action in self.root_node.child_actions])
        return np.sum((counts / np.sum(counts)) * Q)

    def get_a0c_value_target(self) -> np.array:
        Q = np.array([child_action.Q for child_action in self.root_node.child_actions])
        return Q.max()

    def get_greedy_value_target(self) -> np.array:
        node = self.root_node

        while node.terminal and node.has_children:
            counts = np.array([child_action.n for child_action in node.child_actions])
            child = node.child_actions[argmax(counts)].child_node
            if not child:
                break
            else:
                node = child

        Q = np.array([child_action.Q for child_action in node.child_actions])
        return Q.max()

    @staticmethod
    def selection(action: Action) -> Node:
        return action.child_node

    @staticmethod
    def expansion(
        action: Action, state: np.array, reward: float, terminal: bool
    ) -> Node:
        node = action.add_child_node(state, reward, terminal)
        return node

    @staticmethod
    def simulation(mcts_env: gym.Env) -> np.array:
        V = 0
        terminal = False
        while not terminal:
            action = mcts_env.action_space.sample()
            _, reward, terminal, _ = mcts_env.step(action)
            V += reward

        return V.unsqueeze(dim=0)

    @staticmethod
    def backprop(node: Node, gamma: float):
        R = node.V
        # loop back-up until root is reached
        while node.parent_action is not None:
            R = node.r + gamma * R
            action = node.parent_action
            action.update(R)
            node = action.parent_node
            node.update_visit_counts()


class MCTSDiscrete(MCTS):
    """ MCTS object """

    def __init__(
        self,
        model: torch.nn.Module,
        num_actions: int,
        is_atari: bool,
        n_rollouts: int,
        c_uct: float,
        gamma: float,
        device: str,
        root_state: np.array,
        root=None,
    ):

        super().__init__(
            model=model,
            n_rollouts=n_rollouts,
            c_uct=c_uct,
            gamma=gamma,
            device=device,
            root_state=root_state,
            root=root,
        )

        self.num_actions = num_actions
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
        else:
            # continue from current root
            self.root_node.parent_action = None
        if self.root_node.terminal:
            raise ValueError("Can't do tree search from a terminal node")

        # for Atari: snapshot the root at the beginning
        if self.is_atari:
            snapshot = copy_atari_state(Env)

    def evaluation(self, node: NodeDiscrete, V: float = None) -> None:
        state = torch.from_numpy(node.state[None,]).float().to(self.device)

        # only use the neural network to estimate the value if we have none
        if not V:
            node.V = (
                np.squeeze(self.model.predict_V(state))
                if not node.terminal
                else np.array(0.0)
            )
        else:
            node.V = V

        node.child_actions = [
            ActionDiscrete(a, parent_node=node, Q_init=node.V)
            for a in range(node.num_actions)
        ]
        node.priors = self.model.predict_pi(state).flatten()

    def search(self, Env: gym.Env, mcts_env: gym.Env, simulation: bool):
        """ Perform the MCTS search from the root """

        self.initialize_search()
        if simulation:
            mcts_env = copy.deepcopy(Env)
            V = self.simulation(mcts_env)
            self.evaluation(self.root_node, V)
        else:
            self.evaluation(self.root_node)

        for i in range(self.n_rollouts):
            # reset to root for new trace
            node = self.root_node

            if not self.is_atari:
                # copy original Env to rollout from
                mcts_env = copy.deepcopy(Env)
            else:
                restore_atari_state(mcts_env, snapshot)

            while not node.terminal:
                action = self.selectionUCT(self.c_uct, node)

                # take step
                new_state, reward, terminal, _ = mcts_env.step(action.action)
                if getattr(action, "child_node"):
                    # selection
                    node = self.selection(action)
                    continue
                else:
                    # expansion
                    node = self.expansion(action, new_state, reward, terminal)

                    # Evaluate node -> Perform simulation if not already in a terminal state
                    if not terminal and simulation:
                        V = self.simulation(mcts_env)
                        self.evaluation(node, V)
                    else:
                        self.evaluation(node)
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

    def return_results(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """ Process the output at the root node """
        actions = np.array(
            [child_action.action for child_action in self.root_node.child_actions]
        )
        counts = np.array(
            [child_action.n for child_action in self.root_node.child_actions]
        )
        V_target = self.get_softz_value_target(counts)
        return self.root_node.state, actions.squeeze(), counts, V_target

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


class MCTSContinuous(MCTS):
    """ MCTS object """

    def __init__(
        self,
        model: torch.nn.Module,
        n_rollouts: int,
        c_uct: float,
        c_pw: float,
        kappa: float,
        gamma: float,
        device: str,
        root_state: np.array,
        root=None,
    ):
        super().__init__(
            model=model,
            n_rollouts=n_rollouts,
            c_uct=c_uct,
            gamma=gamma,
            device=device,
            root_state=root_state,
            root=root,
        )

        self.c_pw = c_pw
        self.kappa = kappa

    def initialize_search(self) -> None:
        if self.root_node is None:
            self.root_node = NodeContinuous(
                self.root_state, r=0.0, terminal=False, parent_action=None,
            )
        else:
            # continue from current root
            self.root_node.parent_action = None
        if self.root_node.terminal:
            raise ValueError("Can't do tree search from a terminal node")

    def add_value_estimate(self, node: NodeContinuous, mcts_env: gym.Env = None):
        if mcts_env:
            node.V = self.simulation(mcts_env)
        else:
            state = torch.from_numpy(node.state[None,]).float().to(self.device)
            node.V = (
                np.squeeze(self.model.predict_V(state))
                if not node.terminal
                else np.array(0.0)
            )

    def add_pw_action(self, node: NodeContinuous) -> None:
        state = torch.from_numpy(node.state[None,]).float().to(self.device)
        action = self.model.sample_action(state)
        new_child = ActionContinuous(action, parent_node=node, Q_init=node.V)
        node.child_actions.append(new_child)

    def search(self, Env: gym.Env, mcts_env: gym.Env, simulation: bool):
        """ Perform the MCTS search from the root """

        self.initialize_search()
        if simulation:
            mcts_env = copy.deepcopy(Env)
            self.add_value_estimate(self.root_node, mcts_env)
            self.add_pw_action(self.root_node)
        else:
            self.add_value_estimate(self.root_node)
            self.add_pw_action(self.root_node)

        for i in range(self.n_rollouts):
            # reset to root for new trace
            node = self.root_node

            # copy original Env to rollout from
            mcts_env = copy.deepcopy(Env)

            while not node.terminal:
                action = self.selectionUCT(node)

                # take step
                new_state, reward, terminal, _ = mcts_env.step(action.action)
                reward /= PENDULUM_R_SCALE
                if getattr(action, "child_node"):
                    # selection
                    node = self.selection(action)
                    continue
                else:
                    # expansion
                    node = self.expansion(
                        action, np.squeeze(new_state), reward, terminal
                    )

                    if not terminal and simulation:
                        self.add_value_estimate(node, mcts_env)
                    else:
                        self.add_value_estimate(node)
                    break

            self.backprop(node, self.gamma)

    def selectionUCT(self, node: NodeContinuous) -> ActionContinuous:
        """ Select one of the child actions based on UCT rule """
        if node.check_pw(self.c_pw, self.kappa):
            self.add_pw_action(node)
            return node.child_actions[-1]
        else:
            UCT = np.array(
                [
                    child_action.Q
                    + self.c_uct * (np.sqrt(node.n + 1) / (child_action.n + 1))
                    for child_action in node.child_actions
                ]
            )
            winner = argmax(UCT)
            return node.child_actions[winner]

    def return_results(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """ Process the output at the root node """
        actions = np.array(
            [child_action.action for child_action in self.root_node.child_actions]
        )
        counts = np.array(
            [child_action.n for child_action in self.root_node.child_actions]
        )
        V_target = self.get_a0c_value_target()
        return self.root_node.state, actions.squeeze(), counts, V_target

