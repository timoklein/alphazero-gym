import numpy as np
from tqdm import trange
import argparse
import time
import git
import wandb

from alphazero.agents import AlphaZeroAgent
from alphazero.losses import A0CLoss, A0CLossTuned, AlphaZeroLoss
from alphazero.buffers import ReplayBuffer
from alphazero.helpers import store_actions
from alphazero.helpers import is_atari_game
from rl.make_game import make_game


def run_discrete_agent(
    game: str,
    n_ep: int,
    n_traces: int,
    max_ep_len: int,
    lr: float,
    c: float,
    gamma: float,
    buffer_size: int,
    batch_size: int,
    temp: float,
    n_hidden_layers: int,
    n_hidden_units: int,
    value_loss_ratio: float,
    seed: int,
):
    episode_returns = []  # storage
    R_max = -np.inf
    best_actions = None
    actions_list = []

    # Environments
    Env = make_game(game)
    is_atari = is_atari_game(Env)
    mcts_env = make_game(game) if is_atari else None

    # set seeds
    np.random.seed(seed)
    Env.seed(seed)

    buffer = ReplayBuffer(max_size=buffer_size, batch_size=batch_size)
    t_total = 0  # total steps

    loss = AlphaZeroLoss(1, value_loss_ratio, "mean")
    agent = AlphaZeroAgent(
        Env,
        n_hidden_layers=n_hidden_layers,
        n_hidden_units=n_hidden_units,
        n_traces=n_traces,
        lr=lr,
        temperature=temp,
        c_uct=c,
        gamma=gamma,
        loss=loss,
    )

    repo = git.Repo(search_parent_directories=True)
    config = {
        "Commit": repo.head.object.hexsha,
        "Environment": Env.unwrapped.spec.id,
        "Environment seed": seed,
        "Discrete Env": agent.action_discrete,
        "MCTS_traces": agent.n_traces,
        "UCT constant": agent.c_uct,
        "Discount factor": agent.gamma,
        "Softmax temperature": agent.temperature,
        "Network hidden layers": agent.n_hidden_layers,
        "Network hidden units": agent.n_hidden_units,
        "Learning rate": agent.lr,
        "Batch size": buffer.batch_size,
        "Replay buffer size": buffer.max_size,
        "Policy Coefficient": agent.loss.policy_coeff,
        "Value loss ratio": agent.loss.value_coeff,
    }
    if isinstance(agent.loss, A0CLoss):
        config["Log counts scaling factor [tau]"] = agent.loss.tau
        config["Entropy Parameter alpha"] = agent.loss.alpha
    elif isinstance(agent.loss, A0CLossTuned):
        config["Log counts scaling factor [tau]"] = agent.loss.tau
        config["Automatic Entropy tuning"] = True

    run = wandb.init(name="AlphaZero Discrete", project="a0c", config=config)

    pbar = trange(n_ep)
    for ep in pbar:
        start = time.time()
        state = Env.reset()
        R = 0.0  # Total return counter
        if is_atari:
            mcts_env.reset()
            mcts_env.seed(int(Env.seed()))

        agent.reset_mcts(root_state=state)
        for t in range(max_ep_len):
            # MCTS step
            # run mcts and extract the root output
            action, s, actions, counts, V = agent.act(Env=Env, mcts_env=mcts_env)
            buffer.store((s, actions, counts, V))

            # Make the true step
            state, step_reward, terminal, _ = Env.step(action)
            actions_list.append(action)
            R += step_reward
            t_total += (
                n_traces  # total number of environment steps (counts the mcts steps)
            )

            if terminal or t == max_ep_len - 1:
                if R_max < R:
                    actions_list.insert(0, Env.seed()[0])
                    best_actions = actions_list
                    R_max = R
                    store_actions(game, best_actions)
                actions_list.clear()
                break
            else:
                agent.mcts_forward(action, state)

        # store the total episode return
        episode_returns.append(R)

        # Train
        info_dict = agent.train(buffer)
        info_dict["Episode reward"] = R
        if isinstance(agent.loss, A0CLossTuned):
            info_dict["alpha"] = agent.loss.alpha

        # agent.save_checkpoint(env=Env)

        run.log(
            info_dict, step=ep,
        )

        reward = np.round(R, 2)
        e_time = np.round((time.time() - start), 1)
        pbar.set_description(f"{ep=}, {reward=}, {e_time=}s")
    # Return results
    return episode_returns, best_actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="CartPole-v0", help="Training environment")
    parser.add_argument("--n_ep", type=int, default=300, help="Number of episodes")
    parser.add_argument(
        "--n_traces", type=int, default=25, help="Number of MCTS traces per step"
    )
    parser.add_argument(
        "--max_ep_len",
        type=int,
        default=200,
        help="Maximum number of steps per episode",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--c", type=float, default=1.5, help="UCT constant")
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Temperature in normalization of counts to policy target",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount parameter")
    parser.add_argument(
        "--buffer_size", type=int, default=1000, help="Size of the FIFO replay buffer"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Minibatch size")
    parser.add_argument(
        "--n_hidden_layers", type=int, default=1, help="Number of hidden layers in NN"
    )
    parser.add_argument(
        "--n_hidden_units",
        type=int,
        default=128,
        help="Number of units per hidden layers in NN",
    )
    parser.add_argument(
        "--value_ratio",
        type=float,
        default=1,
        help="Value loss ratio in the AlphaZero loss",
    )
    parser.add_argument(
        "--env_seed", type=int, default=34, help="Random seed for the environment",
    )

    args = parser.parse_args()
    episode_returns, best_actions = run_discrete_agent(
        game=args.game,
        n_ep=args.n_ep,
        n_traces=args.n_traces,
        max_ep_len=args.max_ep_len,
        lr=args.lr,
        c=args.c,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        temp=args.temp,
        n_hidden_layers=args.n_hidden_layers,
        n_hidden_units=args.n_hidden_units,
        value_loss_ratio=args.value_ratio,
        seed=args.env_seed,
    )

