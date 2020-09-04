from pathlib import Path
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


def argmax(x):
    """ assumes a 1D vector x """
    x = x.flatten()
    if np.any(np.isnan(x)):
        print("Warning: Cannot argmax when vector contains nans, results will be wrong")

    winners = np.where(x == np.max(x))
    winner = random.choice(winners[0])
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


def store_actions(name: str, to_store: np.array) -> None:
    """ to prevent losing information due to interruption of process"""
    path = Path("runs/")
    if not path.exists():
        path.mkdir()

    actions_path = path / f"{name}.npy"

    np.save(actions_path, to_store)


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

