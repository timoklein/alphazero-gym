from __future__ import annotations
import numpy as np
from typing import List, Tuple

from torch._C import namedtuple_solution_cloned_coefficient


class ReplayBuffer:
    """Replay buffer class.

    The buffer holds all training experiences. It is implemented as a FIFO queue
    using a list and an insertion index. The __next__ implemention allows iteration
    over the class in the training process.

    Attributes
    -------
    max_size: int
        Maximum number of experiences in the buffer.
    batch_size: int
        Training batch size.
    sample_array: np.ndarray
        Array holding the indices of all samples.
    sample_index: int
        Index of the current experience.
    insert_index: int
        Index of where the next experience is inserted.
    size: int
        Current size of the buffer.
    experience: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        List of experiences. An experience consists of a state, actions, visitation counts,
        action values and a value target.
    """

    max_size: int
    batch_size: int
    sample_array: np.ndarray
    sample_index: int
    insert_index: int
    size: int
    experience: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]

    def __init__(self, max_size: int, batch_size: int) -> None:
        """Constructor.

        Parameters
        ----------
        max_size: int
            Maximum size of this instance.
        batch_size: int
            Batch size of this instance.
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self.clear()
        self.sample_index = 0

    def clear(self) -> None:
        """Empties the experience list and resets the insertion index."""
        self.experience = []
        self.insert_index = 0
        self.size = 0

    def store(
        self,
        experience: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Store a single experience in the buffer.

        An experience consists of an environment state, its corresponding actions, visitation counts and
        action values as well as a value target

        Parameters
        ----------
        experience: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the state, actions, visitation counts, action values and the value target.
        """
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size += 1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def reshuffle(self) -> None:
        """Reshuffle the buffer and reset its sample index."""
        self.sample_array = np.arange(self.size)
        np.random.shuffle(self.sample_array)
        self.sample_index = 0

    def __iter__(self) -> ReplayBuffer:
        """Return itself as an Iterable."""
        return self

    def __len__(self) -> int:
        """Return the number of experiences in this class."""
        return len(self.experience)

    def __next__(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the next batch of training data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Batch of collated training experiences.
        """
        if (self.sample_index + self.batch_size > self.size) and (
            not self.sample_index == 0
        ):
            self.reshuffle()  # Reset for the next epoch
            raise (StopIteration)

        assert self.sample_array is not None
        if self.sample_index + 2 * self.batch_size > self.size:
            indices = self.sample_array[self.sample_index :]
            batch = [self.experience[i] for i in indices]
        else:
            indices = self.sample_array[
                self.sample_index : self.sample_index + self.batch_size
            ]
            batch = [self.experience[i] for i in indices]
        self.sample_index += self.batch_size

        # reshape experience into batches
        states, actions, counts, Qs, values = map(np.stack, zip(*batch))
        return states, actions, counts, Qs, values

    next = __next__
