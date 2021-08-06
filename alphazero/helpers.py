from pathlib import Path
from typing import Any, List, Tuple, Union
import gym
import numpy as np
import random
from gym import spaces


def stable_normalizer(x: np.ndarray, temp: float) -> np.ndarray:
    """Computes x[i]**temp/sum_i(x[i]**temp).

    Normalize an input array such that the sum of its elements is one.

    Parameters
    ----------
    x: np.ndarray
        Input array.
    temp: float
        Temperature parameter for the normalization.

    Returns
    -------
    np.ndarray
          Normalized array summing to one.
    """
    x = (x / np.max(x)) ** temp
    return np.abs(x / np.sum(x))


def argmax(x: np.ndarray) -> int:
    """Compute the argmax of an array.

    The difference between this argmax function and the numpy one is that this function
    will break ties by returning a random element instead of the first one.

    Parameters
    ----------
    x: np.ndarray
        Input array.

    Returns
    -------
    int
        Index of the maximum value.
    """
    x = x.flatten()
    if np.any(np.isnan(x)):
        print("Warning: Cannot argmax when vector contains nans, results will be wrong")

    winners = np.where(x == np.max(x))
    winner = random.choice(winners[0])
    return winner


def check_space(space: Any) -> Tuple[Tuple[int], bool]:
    """Check the properties of an environment state or action space.

    Parameters
    ----------
    space : Any
        Environment state space or action space.

    Returns
    -------
    Tuple[Tuple[int], bool]
        Tuple where the first element is the dimensionality and the second element
        is a boolean that is True when the space is discrete.
    """
    if isinstance(space, spaces.Box):
        dim = space.shape
        discrete = False
    elif isinstance(space, spaces.Discrete):
        dim = space.n
        discrete = True
    else:
        raise NotImplementedError("This type of space is not supported")
    dim = (dim,) if isinstance(dim, int) else dim
    return dim, discrete


def store_actions(name: str, to_store: List[Any]) -> None:
    """ to prevent losing information due to interruption of process"""
    path = Path("runs/")
    if not path.exists():
        path.mkdir()

    actions_path = path / f"{name}.npy"

    np.save(actions_path, np.array(to_store))


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
