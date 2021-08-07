import copy
import random
from typing import Any, Optional, Tuple, Union
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
    """Base MCTS class.

    The base MCTS class implements functionality that is common for either discrete or
    continuous action spaces. This is specifically:
        - The calculation of the value target.
        - Epsilon-greedy action selection.
        - Selection phase of the MCTS.
        - Expansion phase of the MCTS.
        - Backup phase of the MCTS.
        - Return of the final search results.

    The search itself, adding of the value targets and the selection phase must be implemented
    differently for discrete and continuous action spaces due to progressive widening.
    """

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
    ) -> None:
        """Constructor.

        Parameters
        ----------
        model: torch.nn.Module
            Point to the network model used to guide the search.
        n_rollouts: int
            Number of search traces per execution of the MCTS search.
        c_uct: float
            UCT exploration term constant.
        gamma: float
            Discount factor for the backup phase of the MCTS.
        epsilon: float
            Epsilon constant for epsilon greedy action selection.
        device: str
            Device for the search execution. Can be either "cpu" or "gpu".
        V_target_policy: str
            Method for calculating the value targets. Can be either "on_policy", "off_policy" or "greedy".
        root_state: np.ndarray
            Environment state associated with the root node.
        """
        self.device = torch.device(device)
        self.root_node = None
        self.root_state = root_state
        self.model = model
        self.n_rollouts = n_rollouts
        self.c_uct = c_uct
        self.gamma = gamma
        self.epsilon = epsilon
        self.V_target_policy = V_target_policy

    @abstractmethod
    def selectionUCT(self, node: Node) -> Action:
        """Interface for the selection method. Must be implemented differently for both action spaces"""
        ...

    @abstractmethod
    def search(self, Env: gym.Env) -> None:
        """Interface method for executing the search."""
        ...

    @staticmethod
    def get_on_policy_value_target(Q: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Calculate the on-policy value target.

        The on-policy value target is the sum of the root action counts weighted by
        the visitation count distribution. The visitation count distribution is obtained
        by dividing the visitation counts with the total sum of counts at the root.

        Parameters
        ----------
        Q: np.ndarray
            Q values of the root node actions
        counts: np.ndarray
            Visitation counts of the root node actions.

        Returns
        -------
        np.ndarray
            Scalar numpy array containing the value target.
        """
        return np.sum((counts / np.sum(counts)) * Q)

    @staticmethod
    def get_off_policy_value_target(Q: np.ndarray) -> Any:
        """Calculate the off-policy value target.

        The off policy value target is the maximum action value at the root node.
        This is the value target proposed in the A0C paper:
        https://arxiv.org/pdf/1805.09613.pdf.

        Parameters
        ----------
        Q: np.ndarray
            Action values at the root node.

        Returns
        -------
        Any
            Scalar numpy array containing the value target.
        """
        return Q.max()

    def get_greedy_value_target(self, final_selection: str) -> Any:
        """Calculate the greedy value target.

        The greedy value target descends down the tree selecting the action with the
        highest action value/highest visitation count until a leaf node is reached.
        This action's Q-value is then returned as value target.
        More information about this value target can be found here:
        https://ala2020.vub.ac.be/papers/ALA2020_paper_18.pdf.

        Parameters
        ----------
        final_selection: str
            Final selection policy used in the search. Can be "max_value" or "max_visit".

        Returns
        -------
        Any
            Scalar value target.
        """
        assert self.root_node is not None
        node = self.root_node

        while node.terminal and node.has_children:
            if final_selection == "max_value":
                Q = np.array(
                    [child_action.Q for child_action in self.root_node.child_actions]
                )
                child = node.child_actions[Q.argmax()].child_node
            else:
                counts = np.array(
                    [child_action.n for child_action in node.child_actions]
                )
                child = node.child_actions[counts.argmax()].child_node

            if not child:
                break
            else:
                node = child

        Q = np.array([child_action.Q for child_action in node.child_actions])
        return Q.max()

    def epsilon_greedy(self, node: Node, UCT: np.ndarray) -> Action:
        """Implementation of epsilon greedy action selection.

        Parameters
        ----------
        node: Node
            Node from which a child action should be selected.
        UCT: np.ndarray
            UCT values of the child actions in the passed node.

        Returns
        -------
        Action
            Action selected by this algorithm.
        """
        if random.random() < self.epsilon:
            # return a random child if the epsilon greedy conditions are met
            return node.child_actions[random.randint(0, len(node.child_actions) - 1)]
        else:
            winner = argmax(UCT)
            return node.child_actions[winner]

    @staticmethod
    def selection(action: Action) -> Optional[Node]:
        """MCTS selection phase. Select the child node of the chosen action.

        When this method returns None, the search proceeds with the expansion stage.

        Parameters
        ----------
        action: Action
            Action whose child node should be selected.

        Returns
        -------
        Optional[Node]
            Child node of the action.
        """
        return action.child_node

    @staticmethod
    def expansion(
        action: Action, state: np.ndarray, reward: float, terminal: bool
    ) -> Node:
        """Expand the tree by adding a new child node given an action.

        Parameters
        ----------
        action: Action
            Action that leads to the newly created node.
        state: np.ndarray
            Environment state of the new node.
        reward: float
            Reward obtained from the environment transition that leads to the new node.
        terminal: bool
            Flag indicating whether the new state is terminal or not.

        Returns
        -------
        Node
            Node wrapping the passed environment state.
        """
        node = action.add_child_node(state, reward, terminal)
        return node

    @staticmethod
    def backprop(node: Node, gamma: float) -> None:
        """Implementation of the MCTS backup phase.

        Starting from the passed node, the backup loop goes back up the tree until the
        root node is reached. For each node along its path it updates:
            - The cumulative discounted reward of the node.
            - The visitation count node.
        For each action along the path:
            - Update of the cumulative discounted reward.
            - Update of the visitation count.
            - Update of the action value (cumulative reward/ visitation count)

        Parameters
        ----------
        node: Node
            Leaf node of the algorithm that has been evaluated by the neural network.
        gamma: float
            Discount factor.
        """
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
        """Returns the root node statistics once an MCTS stage has been completed.

        Parameters
        ----------
        final_selection: str
            Policy according to which the final action should be chosen. Needed for the
            calculation of the greedy value targets.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the:
                - Root node state.
                - Actions selected at the root node.
                - Root node visitation counts.
                - Root node action values.
                - Root node value targets.
        """
        assert self.root_node is not None
        actions = np.array(
            [child_action.action for child_action in self.root_node.child_actions]
        )
        counts = np.array(
            [child_action.n for child_action in self.root_node.child_actions]
        )

        Q = np.array([child_action.Q for child_action in self.root_node.child_actions])

        if self.V_target_policy == "greedy":
            V_target = self.get_greedy_value_target(final_selection)
        elif self.V_target_policy == "on_policy":
            V_target = self.get_on_policy_value_target(Q, counts)
        else:
            V_target = self.get_off_policy_value_target(Q)

        return self.root_node.state, actions.squeeze(), counts, Q, V_target


class MCTSDiscrete(MCTS):
    """Implementation of the MCTS algorithm for discrete action spaces.

    Assumes that the number of actions in each state is finite and does not change over time.
    """

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
    ):
        """Constructor.

        Parameters
        ----------
        model: torch.nn.Module
            Point to the network model used to guide the search.
        num_actions: int
            Number of available actions in each environment state.
        n_rollouts: int
            Number of search traces per execution of the MCTS search.
        c_uct: float
            UCT exploration term constant.
        gamma: float
            Discount factor for the backup phase of the MCTS.
        epsilon: float
            Epsilon constant for epsilon greedy action selection.
        V_target_policy: str
            Method for calculating the value targets. Can be either "on_policy", "off_policy" or "greedy".
        device: str
            Device for the search execution. Can be either "cpu" or "gpu".
        root_state: np.ndarray
            Environment state associated with the root node.
        """
        super().__init__(
            model=model,
            n_rollouts=n_rollouts,
            c_uct=c_uct,
            gamma=gamma,
            epsilon=epsilon,
            V_target_policy=V_target_policy,
            device=device,
            root_state=root_state,
        )

        self.num_actions = num_actions

    def initialize_search(self) -> None:
        """Initialize the search at the root node. This includes concretely:
            - Construction of a Node object from the environment state passed to this class.
            - Check whether the root node is terminal or not.

        In case the tree is reused, no new root is constructed.
        """
        if self.root_node is None:
            self.root_node = NodeDiscrete(  # type: ignore[assignment]
                self.root_state,
                r=0.0,
                terminal=False,
                parent_action=None,
                num_actions=self.num_actions,
            )
        else:
            # continue from current root
            self.root_node.parent_action = None
        if self.root_node.terminal:  # type: ignore[attr-defined]
            raise ValueError("Can't do tree search from a terminal node")

    def evaluation(self, node: NodeDiscrete) -> None:
        """Use the neural network to evaluate a node.

        Evaluation of a node consists of adding a value estimate and prior probabilities
        for all available actions at the node.

        Parameters
        ----------
        node: NodeDiscrete
            Node to be evaluated.
        """
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
            (self.model.predict_V(state)).item()  # type:ignore[operator]
            if not node.terminal
            else 0.0
        )

        node.child_actions = [
            ActionDiscrete(a, parent_node=node, Q_init=node.V)
            for a in range(node.num_actions)
        ]
        node.priors = self.model.predict_pi(state).flatten()  # type:ignore[operator]

    def search(self, Env: gym.Env) -> None:
        """Execute the MCTS search.

        The MCTS algorithm relies on knowing the environment's transition dynamics to
        perform rollouts. In the case of OpenAI gym this means that the environment has to
        be passed. It is then copied in this method before each search trace. A total
        of n_rollouts searches are executed.

        Parameters
        ----------
        Env: gym.Env
            OpenAI gym environment
        """

        self.initialize_search()

        assert self.root_node is not None

        # add network estimates to the root node
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
                if hasattr(action, "child_node"):
                    # selection
                    node = self.selection(action)
                    continue
                else:
                    # expansion
                    node = self.expansion(action, new_state, reward, terminal)

                    # Evaluate node -> Add distribution and value estimate
                    self.evaluation(node)
                    break

            self.backprop(node, self.gamma)

    def selectionUCT(self, node: NodeDiscrete) -> Action:  # type: ignore[override]
        """UCT selection method for discrete action spaces.

        Calculates the UCT value for all actions of a node. If enabled, epsilon-greedy
        action selection is performed.

        Parameters
        ----------
        node: NodeDiscrete
            Node for which a child action should be selected.

        Returns
        -------
        Action
            Best action according to UCT or epsilon-greedy.
        """
        assert node.priors is not None
        UCT = np.array(
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
        """Moves the root node forward.

        This method implements tree reuse. The action selected in the environment leads to
        a state which is the new root node of the tree. Through this the search tree is
        preserved.

        Parameters
        ----------
        action: int
            Action selected in the environment.
        state: np.ndarray
            State obtained by selecting the passed action.
        """
        assert self.root_node is not None
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
    """Continuous MCTS implementation.

    This class uses progressive widening to deal with continuous action spaces.
    More information about progressive widening can be found here:
    https://hal.archives-ouvertes.fr/hal-00542673v2/document
    """

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
    ):
        """Constructor.

        Parameters
        ----------
        model: torch.nn.Module
            Point to the network model used to guide the search.
        n_rollouts: int
            Number of search traces per execution of the MCTS search.
        c_uct: float
            UCT exploration term constant.
        c_pw: float
            Progressive widening factor.
        kappa: float
            Progressive widening exponent.
        gamma: float
            Discount factor for the backup phase of the MCTS.
        epsilon: float
            Epsilon constant for epsilon greedy action selection.
        V_target_policy: str
            Method for calculating the value targets. Can be either "on_policy", "off_policy" or "greedy".
        device: str
            Device for the search execution. Can be either "cpu" or "gpu".
        root_state: np.ndarray
            Environment state associated with the root node.
        """
        super().__init__(
            model=model,
            n_rollouts=n_rollouts,
            c_uct=c_uct,
            gamma=gamma,
            epsilon=epsilon,
            V_target_policy=V_target_policy,
            device=device,
            root_state=root_state,
        )

        self.c_pw = c_pw
        self.kappa = kappa

    def initialize_search(self) -> None:
        """Initialize the search at the root node. This includes concretely:
            - Construction of a Node object from the environment state passed to this class.
            - Check whether the root node is terminal or not.

        Note that tree reuse is not possible for continuous domains.
        """
        self.root_node = NodeContinuous(  # type: ignore[assignment]
            self.root_state, r=0.0, terminal=False, parent_action=None
        )
        if self.root_node.terminal:  # type: ignore[attr-defined]
            raise ValueError("Can't do tree search from a terminal node")

    def add_value_estimate(self, node: NodeContinuous) -> None:
        """Adds a neural network value estimate to the passed node.

        Parameters
        ----------
        node: NodeContinuous
            Node which should be evaluated.
        """
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
            np.squeeze(self.model.predict_V(state))  # type: ignore[operator]
            if not node.terminal
            else np.array(0.0)
        )

    def add_pw_action(self, node: NodeContinuous) -> None:
        """Adds a new action to the passed node.

        This method uses the network the evaluate the node and produce a distribution
        over the action space. From this distribution a new action is sampled and appended
        to the node.
        Note 1: It is inefficient to evaluate the network each time a new action should be sampled.
            It is faster to add the distribution to the node and sample from it multiple times.
            This works as the generated distribution is deterministic given a state. The version
            implemented in this class is slower. Since the used neural networks are small it does
            not matter too much.
        Note 2: This method does not check whether the criterion for progressive widening is met.

        Parameters
        ----------
        node: NodeContinuous
            Node to which a new action should be added.
        """
        state = (
            torch.from_numpy(
                node.state[
                    None,
                ]
            )
            .float()
            .to(self.device)
        )
        action = self.model.sample_action(state)  # type: ignore[operator]
        new_child = ActionContinuous(action, parent_node=node, Q_init=node.V)
        node.child_actions.append(new_child)

    def search(self, Env: gym.Env) -> None:
        """Execute the MCTS search.

        The MCTS algorithm relies on knowing the environment's transition dynamics to
        perform rollouts. In the case of OpenAI gym this means that the environment has to
        be passed. It is then copied in this method before each search trace. A total
        of n_rollouts searches are executed.

        Parameters
        ----------
        Env: gym.Env
            OpenAI gym environment
        """

        self.initialize_search()
        assert self.root_node is not None
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

                if hasattr(action, "child_node"):
                    # selection
                    node = self.selection(action)
                    continue
                else:
                    # expansion
                    node = self.expansion(
                        action, np.squeeze(new_state), reward, terminal
                    )

                    self.add_value_estimate(node)
                    break

            self.backprop(node, self.gamma)

    def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
        """UCT selection method for continuous action spaces.

        The method first checks for progressive widening. If widening is performed, no
        UCT is needed and the newly generated action is returned.
        Otherwise we proceed with regular UCT selection and use epsilon-greedy if enabled

        Parameters
        ----------
        node: NodeContinuous
            Node for which an action should be selected.

        Returns
        -------
        Action
            Action that is either:
                - Newly generated through progressive widening.
                - The action with the highest UCT value.
                - A random action from epsilon-greedy if enabled.
        """
        # no epsilon greedy if we add a node with progressive widening
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
            if self.epsilon == 0:
                # do standard UCT action selection if epsilon=0
                winner = argmax(UCT)
                return node.child_actions[winner]
            else:
                return self.epsilon_greedy(node=node, UCT=UCT)
