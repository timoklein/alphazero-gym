from pathlib import Path
import argparse
import numpy as np
from gym import wrappers
from rl.make_game import make_game


# TODO: Something's wrong with the seed -> Fix it
def visualize(game: str) -> None:
    # NOTE: Has to be run from a terminal, not from VS Code!
    cwd = Path.cwd()
    run_vals = np.load(cwd / f"runs/{game}.npy")
    seed = run_vals[0]
    actions = run_vals[1:]
    Env = make_game(game)
    Env.reset()
    Env.seed(int(seed))
    for a in actions:
        try:
            Env.step(a)
        except AssertionError:
            Env.step(int(a))
        Env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="CartPole-v0", help="Env name")
    args = parser.parse_args()
    visualize(args.game)
