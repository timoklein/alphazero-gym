#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers
@author: thomas
"""
from typing import Tuple, Union
import gym
import numpy as np
import random
import os
from shutil import copyfile
from gym import spaces


def stable_normalizer(x, temp) -> float:
    """ Computes x[i]**temp/sum_i(x[i]**temp) """
    x = (x / np.max(x)) ** temp
    return np.abs(x / np.sum(x))


def argmax(x) -> int:
    """ assumes a 1D vector x """
    x = x.flatten()
    if np.any(np.isnan(x)):
        print("Warning: Cannot argmax when vector contains nans, results will be wrong")
    try:
        winners = np.argwhere(x == np.max(x)).flatten()
        winner = random.choice(winners)
    except:
        winner = np.argmax(x)  # numerical instability ?
    return winner


def check_space(space) -> Tuple[Union[int, Tuple[int]], bool]:
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


def store_safely(folder, name: str, to_store: np.array) -> None:
    """ to prevent losing information due to interruption of process"""
    new_name = folder + name + ".npy"
    old_name = folder + name + "_old.npy"
    if os.path.exists(new_name):
        copyfile(new_name, old_name)
    np.save(new_name, to_store)
    if os.path.exists(old_name):
        os.remove(old_name)


### Atari helpers ###


def get_base_env(env: gym.Env) -> gym.Env:
    """ removes all wrappers """
    while hasattr(env, "env"):
        env = env.env
    return env


def copy_atari_state(env: gym.Env):
    env = get_base_env(env)
    #  return env.ale.cloneSystemState()
    return env.clone_full_state()


def restore_atari_state(env: gym.Env, snapshot) -> None:
    env = get_base_env(env)
    # env.ale.restoreSystemState(snapshot)
    env.restore_full_state(snapshot)


def is_atari_game(env: gym.Env) -> bool:
    """ Verify whether game uses the Arcade Learning Environment """
    env = get_base_env(env)
    return hasattr(env, "ale")


### Visualization ##
def symmetric_remove(x: np.array, n: int) -> np.array:
    """ removes n items from beginning and end """
    odd = is_odd(n)
    half = int(n / 2)
    if half > 0:
        x = x[half:-half]
    if odd:
        x = x[1:]
    return x


def is_odd(number: int) -> bool:
    """ checks whether number is odd, returns boolean """
    return bool(number & 1)


def smooth(y: np.array, window: int, mode: str) -> np.array:
    """ smooth 1D vectory y """
    return np.convolve(y, np.ones(window) / window, mode=mode)

