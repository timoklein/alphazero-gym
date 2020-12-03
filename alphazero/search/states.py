from __future__ import annotations
from typing import List, Optional, cast
import numpy as np
from math import ceil
from abc import ABC, abstractmethod


class Node(ABC):

    state: np.array
    r: float
    terminal: bool
    parent_action: Action
    n: int
    V: Optional[float]
    child_actions: Optional[List[Action]]

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
        ...

    @abstractmethod
    def update_visit_counts(self) -> None:
        ...


class Action(ABC):

    action: np.array
    parent_node: Node
    W: float
    n: int
    Q: float
    child_node: Optional[Node]

    __slots__ = ("action", "parent_node", "W", "n", "Q", "child_node")

    @abstractmethod
    def add_child_node(self, state: np.array, r: float, terminal: bool) -> Node:
        ...

    @property
    def has_child(self) -> bool:
        return cast(bool, self.child_node)

    def update(self, R: float) -> None:
        self.n += 1
        self.W += R
        self.Q = self.W / self.n


class NodeDiscrete(Node):
    """ Node object """

    state: np.array
    r: float
    terminal: bool
    parent_action: ActionDiscrete
    n: int
    V: Optional[float]
    child_actions: Optional[List[ActionDiscrete]]  # type: ignore[assignment]
    priors: np.array
    num_actions: int

    __slots__ = "priors", "num_actions"

    def __init__(
        self,
        state: np.array,
        r: float,
        terminal: bool,
        parent_action: ActionDiscrete,
        num_actions: int,
    ) -> None:
        """ Initialize a new node """
        self.state = state  # game state
        self.r = r  # reward upon arriving in this node
        self.terminal = terminal  # whether the domain terminated in this node
        self.parent_action = parent_action
        self.n = 0
        self.V = None

        # Child actions
        self.num_actions = num_actions
        self.child_actions = None
        self.priors = None

    @property
    def has_children(self) -> bool:
        return cast(bool, self.child_actions)

    def update_visit_counts(self) -> None:
        """ update count on backward pass """
        self.n += 1


class NodeContinuous(Node):
    """ Node object """

    state: np.array
    r: float
    terminal: bool
    parent_action: ActionContinuous
    n: int
    V: Optional[float]
    child_actions: List[ActionContinuous]  # type: ignore[assignment]

    def __init__(
        self,
        state: np.array,
        r: float,
        terminal: bool,
        parent_action: ActionContinuous,
    ) -> None:
        """ Initialize a new node """
        self.state = state  # game state
        self.r = r  # reward upon arriving in this node
        self.terminal = terminal  # whether the domain terminated in this node
        self.parent_action = parent_action
        self.n = 0
        self.V = None

        # Child actions
        self.child_actions = []

    def check_pw(self, c_pw: float, kappa: float) -> bool:
        pw_actions = ceil(c_pw * ((self.n + 1) ** kappa))
        if 0 < pw_actions - self.num_children:
            return True

        return False

    @property
    def has_children(self) -> bool:
        return cast(bool, self.child_actions)

    @property
    def num_children(self) -> int:
        return len(self.child_actions)

    def update_visit_counts(self) -> None:
        """ update count on backward pass """
        self.n += 1


class ActionDiscrete(Action):
    """ Action object """

    action: np.array
    parent_node: NodeDiscrete
    W: float
    n: int
    Q: float
    child_node: Optional[NodeDiscrete]

    def __init__(self, action: int, parent_node: NodeDiscrete, Q_init: float) -> None:
        self.action = action
        self.parent_node = parent_node
        self.Q = Q_init
        self.W = 0.0
        self.n = 0

        self.child_node = None

    def add_child_node(self, state: np.array, r: float, terminal: bool) -> NodeDiscrete:
        self.child_node = NodeDiscrete(
            state, r, terminal, self, self.parent_node.num_actions
        )
        return self.child_node

    def update(self, R: float) -> None:
        self.n += 1
        self.W += R
        self.Q = self.W / self.n


class ActionContinuous(Action):
    """ Action object """

    action: np.array
    paren_node: NodeContinuous
    W: float
    n: int
    Q: float
    child_node: Optional[NodeContinuous]

    def __init__(self, action: np.array, parent_node: NodeContinuous, Q_init: float):
        self.action = action
        self.parent_node = parent_node
        self.Q = Q_init
        self.W = 0.0
        self.n = 0

        self.child_node = None

    def add_child_node(
        self, state: np.array, r: float, terminal: bool
    ) -> NodeContinuous:
        self.child_node = NodeContinuous(state, r, terminal, self)
        return self.child_node
