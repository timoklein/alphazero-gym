from __future__ import annotations
import random
import numpy as np
from typing import List, Optional, Tuple


class ReplayBuffer:
    """ Experience Replay Buffer  """

    max_size: int
    batch_size: int
    sample_array: np.ndarray
    sample_index: int
    insert_index: int
    size: int
    experience: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]

    def __init__(self, max_size: int, batch_size: int) -> None:
        self.max_size = max_size
        self.batch_size = batch_size
        self.clear()
        self.sample_index = 0

    def clear(self) -> None:
        self.experience = []
        self.insert_index = 0
        self.size = 0

    def store(
        self,
        experience: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size += 1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def reshuffle(self) -> None:
        self.sample_array = np.arange(self.size)
        np.random.shuffle(self.sample_array)
        self.sample_index = 0

    def __iter__(self) -> ReplayBuffer:
        return self

    def __len__(self) -> int:
        return len(self.experience)

    def __next__(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
