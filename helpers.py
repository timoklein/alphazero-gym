#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers
@author: thomas
"""
import numpy as np
import random
import os
from shutil import copyfile
from gym import spaces


def stable_normalizer(x, temp):
    """ Computes x[i]**temp/sum_i(x[i]**temp) """
    x = (x / np.max(x)) ** temp
    return np.abs(x / np.sum(x))


def argmax(x):
    """ assumes a 1D vector x """
    x = x.flatten()
    if np.any(np.isnan(x)):
        print("Warning: Cannot argmax when vector contains nans, results will be wrong")
    winners = np.argwhere(x == np.max(x)).flatten()
    winner = random.choice(winners)
    return winner


def check_space(space):
    """ Check the properties of an environment state or action space """
    if isinstance(space, spaces.Box):
        dim = space.shape
        discrete = False
    elif isinstance(space, spaces.Discrete):
        dim = space.n
        discrete = True
    else:
        raise NotImplementedError("This type of space is not supported")
    return dim, discrete


def store_safely(folder, name, to_store):
    """ to prevent losing information due to interruption of process"""
    new_name = folder + name + ".npy"
    old_name = folder + name + "_old.npy"
    if os.path.exists(new_name):
        copyfile(new_name, old_name)
    np.save(new_name, to_store)
    if os.path.exists(old_name):
        os.remove(old_name)


### Atari helpers ###
def get_base_env(env):
    """ removes all wrappers """
    while hasattr(env, "env"):
        env = env.env
    return env


def copy_atari_state(env):
    env = get_base_env(env)
    # return env.ale.cloneSystemState()
    return env.clone_full_state()


def restore_atari_state(env, snapshot):
    env = get_base_env(env)
    # env.ale.restoreSystemState(snapshot)
    env.restore_full_state(snapshot)


def is_atari_game(env):
    """ Verify whether game uses the Arcade Learning Environment """
    env = get_base_env(env)
    return hasattr(env, "ale")


### Replay Buffer ##
# TODO: Add function to persist to disk
class ReplayBuffer:
    """ Experience Replay Buffer  """

    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.clear()
        self.sample_array = None
        self.sample_index = 0

    def clear(self):
        self.buffer = [None] * self.max_size
        self.insert_index = 0
        self.size = 0

    def store(self, experience):
        self.buffer[self.insert_index % self.max_size] = experience
        self.insert_index += 1
        if self.size <= self.max_size:
            self.size += 1
        else:
            self.size = 0

    def store_from_array(self, *args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            self.store(entry)

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
        arrays = []
        for i in range(len(batch[0])):
            to_add = np.array([entry[i] for entry in batch])
            arrays.append(to_add)
        return tuple(arrays)

    next = __next__


### Visualization ##
def symmetric_remove(x, n):
    """ removes n items from beginning and end """
    odd = is_odd(n)
    half = int(n / 2)
    if half > 0:
        x = x[half:-half]
    if odd:
        x = x[1:]
    return x


def is_odd(number):
    """ checks whether number is odd, returns boolean """
    return bool(number & 1)


def smooth(y, window, mode):
    """ smooth 1D vectory y """
    return np.convolve(y, np.ones(window) / window, mode=mode)

