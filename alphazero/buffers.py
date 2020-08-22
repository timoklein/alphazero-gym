import random
import numpy as np

# TODO: Simplify this
class ReplayBuffer:
    """ Experience Replay Buffer  """

    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.clear()
        self.sample_array = None
        self.sample_index = 0

    def clear(self):
        # self.buffer = [None] * self.max_size
        self.buffer = []
        self.insert_index = 0
        self.size = 0

    def store_from_array(self, *args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            self.store(entry)

    def store(self, experience):
        if self.size < self.max_size:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def reshuffle(self):
        self.sample_array = np.arange(self.size)
        random.shuffle(self.sample_array)
        self.sample_index = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.buffer)

    def __next__(self):
        """ This methods keeps track of the samples we already fetched.
            The sample array is an array containing the indices which have experiences stored.
            The sample index tracks 
        """
        if (self.sample_index + self.batch_size > self.size) and (
            not self.sample_index == 0
        ):
            # Reset for the next epoch and stop loop
            self.reshuffle()
            raise StopIteration

        if self.sample_index + 2 * self.batch_size > self.size:
            indices = self.sample_array[self.sample_index :]
            batch = [self.buffer[i] for i in indices]
        else:
            indices = self.sample_array[
                self.sample_index : self.sample_index + self.batch_size
            ]
            batch = [self.buffer[i] for i in indices]
        self.sample_index += self.batch_size

        state, value, policy = map(np.stack, zip(*batch)) 
        return state, value, policy

    next = __next__