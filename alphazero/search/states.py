from __future__ import annotations
from typing import List, Optional, Union, cast
import numpy as np
from math import ceil
from abc import ABC, abstractmethod


class Node(ABC):
    """Base class for an MCTS node.

    Attributes
    ----------
    state: np.ndarray
        Environment state this node is associated with.
    r: float
        Reward obtained from transitioning to this state.
    terminal: bool
        Flag indicating whether the state is terminal or not.
    parent_action: Optional[Action]
        Pointer to the parent action of this node.
    n: int
        Node visitation count.
    V: float
        Value function estimate of this node.
    child_actions: List[Action]
        List of child actions.
    """

    state: np.ndarray
    r: float
    terminal: bool
    parent_action: Optional[Action]
    n: int
    V: float
    child_actions: List[Action]

    _slots__ = (
        "state",
        "r",
        "terminal",
        "parent_action",
        "n",
        "V",
        "child_actions",
    )

    @property
    @abstractmethod
    def has_children(self) -> bool:
        """Interface for checking whether the node has child actions available or not."""
        ...

    @abstractmethod
    def update_visit_counts(self) -> None:
        """Interface for incrementing the visitation counts in the backup phase."""
        ...


class Action(ABC):
    """Base class for an MCTS action.

    Attributes
    -------
    action: Union[np.ndarray, int]
        Environment action this class wraps.
    parent_node: Node
        MCTS node this action is associated with.
    W: float
        Total sum of backpropagated rewards for this action.
    n: int
        Visitation count of this action.
    Q: float
        Action-value estimate.
    child_node: Optional[Node]
        Node obtained when executing this action.
    """

    action: Union[np.ndarray, int]
    parent_node: Node
    W: float
    n: int
    Q: float
    child_node: Optional[Node]

    __slots__ = ("action", "parent_node", "W", "n", "Q", "child_node")

    @abstractmethod
    def add_child_node(self, state: np.ndarray, r: float, terminal: bool) -> Node:
        """Interface for adding a child node."""
        ...

    @property
    def has_child(self) -> bool:
        """Check whether the action leads to a child node."""
        return cast(bool, self.child_node)

    def update(self, R: float) -> None:
        """Updates this action during the MCTS backup phase.

        The following steps are performed during the update:
            - Incrementation of the visitation count of this instance.
            - Adding the accumulated discounted reward of this trace to the action.
            - Update of the action-value with the new cumulative reward.

        Parameters
        ----------
        R: float
            Accumulated reward of the current search trace.
        """
        self.n += 1
        self.W += R
        self.Q = self.W / self.n


class NodeDiscrete(Node):
    """Node object for the discrete MCTS.

    Attributes
    ----------
    state: np.ndarray
        Environment state this node is associated with.
    r: float
        Reward obtained from transitioning to this state.
    terminal: bool
        Flag indicating whether the state is terminal or not.
    parent_action: Optional[ActionDiscrete]
        Pointer to the parent action of this node.
    n: int
        Node visitation count.
    V: float
        Value function estimate of this node.
    child_actions: List[ActionDiscrete]
        List of child actions.
    priors: Optional[np.ndarray]
        Prior probabilities for each action assigned by the neural network.
    num_actions: int
        Number of actions available at this node.
    """

    state: np.ndarray
    r: float
    terminal: bool
    parent_action: Optional[ActionDiscrete]
    n: int
    V: float
    child_actions: List[ActionDiscrete]  # type: ignore[assignment]
    priors: Optional[np.ndarray]
    num_actions: int

    __slots__ = "priors", "num_actions"

    def __init__(
        self,
        state: np.ndarray,
        r: float,
        terminal: bool,
        parent_action: Optional[ActionDiscrete],
        num_actions: int,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        state: np.ndarray
            Environment state for this node.
        r: float
            Reward obtained from transitioning into this state.
        terminal: bool
            Flag indicating whether this is a terminal state or not.
        parent_action: Optional[ActionDiscrete]
            The parent action leading to this node.
        num_actions: int
            The number of available actions in this node.
        """
        self.state = state  # game state
        self.r = r  # reward upon arriving in this node
        self.terminal = terminal  # whether the domain terminated in this node
        self.parent_action = parent_action
        self.n = 0

        # Child actions
        self.num_actions = num_actions

    @property
    def has_children(self) -> bool:
        """Returns TRue if this node has child actions, False else."""
        return cast(bool, self.child_actions)

    def update_visit_counts(self) -> None:
        """Increment the visitation counts in the backup phase."""
        self.n += 1


class NodeContinuous(Node):
    """Node object for the discrete MCTS.

    Attributes
    ----------
    state: np.ndarray
        Environment state this node is associated with.
    r: float
        Reward obtained from transitioning to this state.
    terminal: bool
        Flag indicating whether the state is terminal or not.
    parent_action: Optional[ActionContinuous]
        Pointer to the parent action of this node.
    n: int
        Node visitation count.
    V: float
        Value function estimate of this node.
    child_actions: List[ActionContinuous]
        List of child actions.
    """

    state: np.ndarray
    r: float
    terminal: bool
    parent_action: Optional[ActionContinuous]
    n: int
    V: float
    child_actions: List[ActionContinuous]  # type: ignore[assignment]

    def __init__(
        self,
        state: np.ndarray,
        r: float,
        terminal: bool,
        parent_action: Optional[ActionContinuous],
    ) -> None:
        """Constructor.

        Parameters
        ----------
        state: np.ndarray
            Environment state for this node.
        r: float
            Reward obtained from transitioning into this state.
        terminal: bool
            Flag indicating whether this is a terminal state or not.
        parent_action: Optional[ActionContinuous]
            The parent action leading to this node.
        """
        self.state = state  # game state
        self.r = r  # reward upon arriving in this node
        self.terminal = terminal  # whether the domain terminated in this node
        self.parent_action = parent_action
        self.n = 0

        # Child actions
        self.child_actions = []

    def check_pw(self, c_pw: float, kappa: float) -> bool:
        """Checks whether the criterion for progressive widening is met.

        Calculates c_pw*n(s)^k and checks whether the this quantity is larger than the
        number of children of the node. If it is, the criterion for progressive widening
        is met and a new child action is sampled.

        Parameters
        ----------
        c_pw: float
            Progressive widening factor.
        kappa: float
            Progressive widening exponent.

        Returns
        -------
        bool
            True if progressive widening should be performed.
        """
        pw_actions = ceil(c_pw * ((self.n + 1) ** kappa))
        if 0 < pw_actions - self.num_children:
            return True

        return False

    @property
    def has_children(self) -> bool:
        """Returns True if the node has child actions."""
        return cast(bool, self.child_actions)

    @property
    def num_children(self) -> int:
        """Return the number of child actions as int."""
        return len(self.child_actions)

    def update_visit_counts(self) -> None:
        """Increment the visitation count during the backup phase."""
        self.n += 1


class ActionDiscrete(Action):
    """Action class for the discrete MCTS.

    Attributes
    -------
    action: Union[np.ndarray, int]
        Environment action this class wraps.
    parent_node: NodeDiscrete
        MCTS node this action is associated with.
    W: float
        Total sum of backpropagated rewards for this action.
    n: int
        Visitation count of this action.
    Q: float
        Action-value estimate.
    child_node: NodeDiscrete
        Node obtained when executing this action.
    """

    action: int
    parent_node: NodeDiscrete
    W: float
    n: int
    Q: float
    child_node: NodeDiscrete

    def __init__(self, action: int, parent_node: NodeDiscrete, Q_init: float) -> None:
        """Constructor.

        Parameters
        ----------
        action: int
            Action in the environment this class wraps. Since the action space is 1D,
            it is a single int.
        parent_node: NodeDiscrete
            MCTS node from which this instance has been executed.
        Q_init: float
            Initial UCT value for an action. Usually a high constant.
        """
        self.action = action
        self.parent_node = parent_node
        self.Q = Q_init
        self.W = 0.0
        self.n = 0

    def add_child_node(
        self, state: np.ndarray, r: float, terminal: bool
    ) -> NodeDiscrete:
        """Add a child node for this action.

        Once the action has been executed, the environment transitions into a new state
        which is wrapped by the child node of this action instance.

        Parameters
        ----------
        state: np.ndarray
            Environment state for the new child node.
        r: float
            Reward of the transition into the new state.
        terminal: bool
            Flag indicating whether the new state is terminal or not.

        Returns
        -------
        NodeDiscrete
            Child node for this action instance.
        """
        self.child_node = NodeDiscrete(
            state, r, terminal, self, self.parent_node.num_actions
        )
        return self.child_node


class ActionContinuous(Action):
    """Action class for the continuous MCTS.

    Attributes
    -------
    action: Union[np.ndarray, int]
        Environment action this class wraps.
    parent_node: NodeContinuous
        MCTS node this action is associated with.
    W: float
        Total sum of backpropagated rewards for this action.
    n: int
        Visitation count of this action.
    Q: float
        Action-value estimate.
    child_node: NodeContinuous
        Node obtained when executing this action.
    """

    action: np.ndarray
    paren_node: NodeContinuous
    W: float
    n: int
    Q: float
    child_node: NodeContinuous

    def __init__(self, action: np.ndarray, parent_node: NodeContinuous, Q_init: float):
        """Constructor.

        Parameters
        ----------
        action: int
            Action in the environment this class wraps. Since the action space is 1D,
            it is a single int.
        parent_node: NodeContinuous
            MCTS node from which this instance has been executed.
        Q_init: float
            Initial UCT value for an action. Usually a high constant.
        """
        self.action = action
        self.parent_node = parent_node
        self.Q = Q_init
        self.W = 0.0
        self.n = 0

    def add_child_node(
        self, state: np.ndarray, r: float, terminal: bool
    ) -> NodeContinuous:
        """Add a child node for this action.

        Once the action has been executed, the environment transitions into a new state
        which is wrapped by the child node of this action instance.

        Parameters
        ----------
        state: np.ndarray
            Environment state for the new child node.
        r: float
            Reward of the transition into the new state.
        terminal: bool
            Flag indicating whether the new state is terminal or not.

        Returns
        -------
        NodeContinuous
            Child node for this action instance.
        """
        self.child_node = NodeContinuous(state, r, terminal, self)
        return self.child_node
