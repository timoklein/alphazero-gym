import torch
import numpy as np
from abc import ABC, abstractmethod


class Action(ABC):

    __slots__ = ("index", "parent_node", "W", "n", "Q", "child_node")

    @abstractmethod
    def add_child_node(self):
        ...

    @abstractmethod
    def update(self):
        ...


class Node(ABC):

    _slots__ = (
        "state",
        "r",
        "terminal",
        "parent_action",
        "n",
        "num_actions",
        "V",
        "priors",
        "child_actions",
    )

    @abstractmethod
    def update_visit_counts(self):
        ...


##### MCTS functions #####
class ActionDiscrete(Action):
    """ Action object """

    def __init__(self, index, parent_node, Q_init):
        self.index = index
        self.parent_node = parent_node
        self.W = 0.0
        self.n = 0
        self.Q = Q_init

        self.child_node = None

    def add_child_node(self, state, r, terminal):
        self.child_node = NodeDiscrete(
            state, r, terminal, self, self.parent_node.num_actions
        )
        return self.child_node

    def update(self, R):
        self.n += 1
        self.W += R
        self.Q = self.W / self.n


class NodeDiscrete(Node):
    """ Node object """

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

    def update_visit_counts(self) -> None:
        """ update count on backward pass """
        self.n += 1
