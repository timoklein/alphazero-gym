import copy
import random
from typing import Tuple
import gym
import torch
import numpy as np
from abc import ABC, abstractmethod

from alphazero.search.states import (
    Action,
    Node,
    ActionContinuous,
    ActionDiscrete,
    NodeContinuous,
    NodeDiscrete,
)
from alphazero.helpers import argmax

# scales the step reward between -1 and 0
PENDULUM_R_SCALE = 16.2736044


class MCTS(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        n_rollouts: int,
        c_uct: float,
        gamma: float,
        epsilon: float,
        device: str,
        V_target_policy: str,
        root_state: np.ndarray,
        root=None,
    ) -> None:

        self.device = torch.device(device)
        self.root_node = root
        self.root_state = root_state
        self.model = model
        self.n_rollouts = n_rollouts
        self.c_uct = c_uct
        self.gamma = gamma
        self.epsilon = epsilon
        self.V_target_policy = V_target_policy

    @abstractmethod
    def selectionUCT(self, node: Node) -> Action:
        ...

    @abstractmethod
    def search(self, Env: gym.Env, mcts_env: gym.Env, simulation: bool) -> None:
        ...

    @staticmethod
    def get_on_policy_value_target(Q: np.ndarray, counts: np.ndarray) -> np.ndarray:
        return np.sum((counts / np.sum(counts)) * Q)

    @staticmethod
    def get_off_policy_value_target(Q) -> np.ndarray:
        return Q.max()

    def get_greedy_value_target(self, final_selection: str) -> np.ndarray:
        node = self.root_node

        while node.terminal and node.has_children:
            if final_selection == "max_value":
                Q = np.ndarray(
                    [child_action.Q for child_action in self.root_node.child_actions]
                )
                child = node.child_actions[Q.argmax()].child_node
            else:
                counts = np.ndarray(
                    [child_action.n for child_action in node.child_actions]
                )
                child = node.child_actions[counts.argmax()].child_node

            if not child:
                break
            else:
                node = child

        Q = np.ndarray([child_action.Q for child_action in node.child_actions])
        return Q.max()

    # TODO: Factor out
    def epsilon_greedy(self, node: Node, UCT: np.ndarray) -> Action:
        if random.random() < self.epsilon:
            # return a random child if the epsilon greedy conditions are met
            return node.child_actions[random.randint(0, len(node.child_actions) - 1)]
        else:
            winner = argmax(UCT)
            return node.child_actions[winner]

    @staticmethod
    def selection(action: Action) -> Node:
        return action.child_node

    @staticmethod
    def expansion(
        action: Action, state: np.ndarray, reward: float, terminal: bool
    ) -> Node:
        node = action.add_child_node(state, reward, terminal)
        return node

    @staticmethod
    def simulation(mcts_env: gym.Env) -> np.ndarray:
        V = np.zeros(1)
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

    def return_results(
        self, final_selection: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Process the output at the root node """
        actions = np.ndarray(
            [child_action.action for child_action in self.root_node.child_actions]
        )
        counts = np.ndarray(
            [child_action.n for child_action in self.root_node.child_actions]
        )

        Q = np.ndarray(
            [child_action.Q for child_action in self.root_node.child_actions]
        )

        if self.V_target_policy == "greedy":
            V_target = self.get_greedy_value_target(final_selection)
        elif self.V_target_policy == "on_policy":
            V_target = self.get_on_policy_value_target(Q, counts)
        else:
            V_target = self.get_off_policy_value_target(Q)

        return self.root_node.state, actions.squeeze(), counts, Q, V_target


class MCTSDiscrete(MCTS):
    """ MCTS object """

    def __init__(
        self,
        model: torch.nn.Module,
        num_actions: int,
        n_rollouts: int,
        c_uct: float,
        gamma: float,
        epsilon: float,
        V_target_policy: str,
        device: str,
        root_state: np.ndarray,
        root=None,
    ):

        super().__init__(
            model=model,
            n_rollouts=n_rollouts,
            c_uct=c_uct,
            gamma=gamma,
            epsilon=epsilon,
            V_target_policy=V_target_policy,
            device=device,
            root_state=root_state,
            root=root,
        )

        self.num_actions = num_actions

    def initialize_search(self, env: gym.Env) -> None:
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

    def evaluation(self, node: NodeDiscrete, V: float = None) -> None:
        state = (
            torch.from_numpy(
                node.state[
                    None,
                ]
            )
            .float()
            .to(self.device)
        )

        # only use the neural network to estimate the value if we have none
        if not V:
            node.V = (
                np.squeeze(self.model.predict_V(state))
                if not node.terminal
                else np.ndarray(0.0)
            )
        else:
            node.V = V

        node.child_actions = [
            ActionDiscrete(a, parent_node=node, Q_init=node.V)
            for a in range(node.num_actions)
        ]
        node.priors = self.model.predict_pi(state).flatten()

    def search(self, Env: gym.Env, mcts_env: gym.Env, simulation: bool) -> None:
        """ Perform the MCTS search from the root """

        snapshot = self.initialize_search(env=Env)
        if simulation:
            mcts_env = copy.deepcopy(Env)
            V = self.simulation(mcts_env)
            self.evaluation(self.root_node, V)
        else:
            self.evaluation(self.root_node)
        for i in range(self.n_rollouts):
            # reset to root for new trace
            node = self.root_node

            # copy original Env to rollout from
            mcts_env = copy.deepcopy(Env)

            while not node.terminal:
                action = self.selectionUCT(node)

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

    def selectionUCT(self, node: NodeDiscrete) -> ActionDiscrete:
        """ Select one of the child actions based on UCT rule """
        UCT = np.ndarray(
            [
                child_action.Q
                + prior * self.c_uct * (np.sqrt(node.n + 1) / (child_action.n + 1))
                for child_action, prior in zip(node.child_actions, node.priors)
            ]
        )
        if self.epsilon == 0:
            # do standard UCT action selection if epsilon=0
            winner = argmax(UCT)
            return node.child_actions[winner]
        else:
            return self.epsilon_greedy(node=node, UCT=UCT)

    def forward(self, action: int, state: np.ndarray) -> None:
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
        epsilon: float,
        V_target_policy: str,
        device: str,
        root_state: np.ndarray,
        root=None,
    ):
        super().__init__(
            model=model,
            n_rollouts=n_rollouts,
            c_uct=c_uct,
            gamma=gamma,
            epsilon=epsilon,
            V_target_policy=V_target_policy,
            device=device,
            root_state=root_state,
            root=root,
        )

        self.c_pw = c_pw
        self.kappa = kappa

    def initialize_search(self) -> None:
        if self.root_node is None:
            self.root_node = NodeContinuous(
                self.root_state,
                r=0.0,
                terminal=False,
                parent_action=None,
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
            state = (
                torch.from_numpy(
                    node.state[
                        None,
                    ]
                )
                .float()
                .to(self.device)
            )
            node.V = (
                np.squeeze(self.model.predict_V(state))
                if not node.terminal
                else np.ndarray(0.0)
            )

    def add_pw_action(self, node: NodeContinuous) -> None:
        state = (
            torch.from_numpy(
                node.state[
                    None,
                ]
            )
            .float()
            .to(self.device)
        )
        action = self.model.sample_action(state)
        new_child = ActionContinuous(action, parent_node=node, Q_init=node.V)
        node.child_actions.append(new_child)

    def search(self, Env: gym.Env, mcts_env: gym.Env, simulation: bool) -> None:
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

    def selectionUCT(self, node: NodeContinuous) -> ActionContinuous:  # type: ignore[return-value, override]
        """ Select one of the child actions based on UCT rule """
        # no epsilon greedy if we add a node with progressive widening
        if node.check_pw(self.c_pw, self.kappa):
            self.add_pw_action(node)
            return node.child_actions[-1]
        else:
            UCT = np.ndarray(
                [
                    child_action.Q
                    + self.c_uct * (np.sqrt(node.n + 1) / (child_action.n + 1))
                    for child_action in node.child_actions
                ]
            )
            if self.epsilon == 0:
                # do standard UCT action selection if epsilon=0
                winner = argmax(UCT)
                return node.child_actions[winner]
            else:
                return self.epsilon_greedy(node=node, UCT=UCT)
