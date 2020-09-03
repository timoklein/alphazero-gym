import numpy as np
from tqdm import trange
import argparse
import time
import git
import wandb

from alphazero.mcts import PENDULUM_R_SCALE
from alphazero.agents import A0CAgent
from alphazero.buffers import ReplayBuffer
from alphazero.helpers import is_atari_game
from rl.make_game import make_game

def run_continuous_agent(
    game: str,
    n_ep: int,
    n_traces: int,
    max_ep_len: int,
    lr: float,
    c_uct: float,
    c_pw: float,
    kappa: float,
    tau: float,
    alpha: float,
    gamma: float,
    buffer_size: int,
    batch_size: int,
    temp: float,
    n_hidden_layers: int,
    n_hidden_units: int,
    value_loss_ratio: float,
    seed: int,
):
    """ Outer training loop """
    episode_returns = []  # storage
    timepoints = []
    # Environments
    Env = make_game(game)
    is_atari = is_atari_game(Env)
    mcts_env = make_game(game) if is_atari else None

    buffer = ReplayBuffer(max_size=buffer_size, batch_size=batch_size)
    t_total = 0  # total steps

    agent = A0CAgent(
        Env,
        n_hidden_layers=n_hidden_layers,
        n_hidden_units=n_hidden_units,
        value_loss_ratio=value_loss_ratio,
        n_traces=n_traces,
        lr=lr,
        temperature=temp,
        c_uct=c_uct,
        c_pw=c_pw,
        kappa=kappa,
        tau=tau,
        alpha=alpha,
        gamma=gamma,
    )

    repo = git.Repo(search_parent_directories=True)
    config = {
        "Commit": repo.head.object.hexsha,
        "Environment": Env.unwrapped.spec.id,
        "Discrete Env": agent.action_discrete,
        "MCTS_traces": agent.n_traces,
        "UCT constant": agent.c_uct,
        "Discount factor": agent.gamma,
        "Softmax temperature": agent.temperature,
        "Network hidden layers": agent.n_hidden_layers,
        "Network hidden units": agent.n_hidden_units,
        "Value loss ratio": agent.value_loss_ratio,
        "Learning rate": agent.lr,
        "Batch size": buffer.batch_size,
        "Replay buffer size": buffer.max_size,
        "Environment seed": seed,
    }

    run = wandb.init(name="A0C", project="a0c", config=config)

    pbar = trange(n_ep)
    for ep in pbar:
        start = time.time()
        state = Env.reset()
        R = 0.0  # Total return counter
        Env.seed(seed)
        if is_atari:
            mcts_env.reset()
            mcts_env.seed(seed)

        agent.reset_mcts(root_state=state)
        for t in range(max_ep_len):
            # MCTS step
            # run mcts and extract the root output
            action, actions, s, log_counts, V = agent.act(Env=Env, mcts_env=mcts_env)
            buffer.store((actions, s, log_counts, V))

            # Make the true step
            state , step_reward, terminal, _ = Env.step(action)
            step_reward /= PENDULUM_R_SCALE

            R += step_reward
            t_total += (
                n_traces  # total number of environment steps (counts the mcts steps)
            )

            if terminal:
                break
            else:
                # reset the mcts as we can't reuse the tree
                agent.reset_mcts(root_state=state)

        # store the total episode return
        episode_returns.append(R)

        # Train
        episode_loss = agent.train(buffer)

        # agent.save_checkpoint(env=Env)

        run.log(
            {
                "Episode reward": R,
                "Total loss": episode_loss["loss"],
                "Policy loss": episode_loss["policy_loss"],
                "Entropy loss": episode_loss["entropy_loss"],
                "Value loss": episode_loss["value_loss"],
            },
            step=ep,
        )

        reward = np.round(R, 2)
        e_time = np.round((time.time() - start), 1)
        pbar.set_description(f"{ep=}, {reward=}, {e_time=}s")
    # Return results
    return episode_returns, timepoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Pendulum-v0", help="Training environment")
    parser.add_argument("--n_ep", type=int, default=300, help="Number of episodes")
    parser.add_argument(
        "--n_traces", type=int, default=10, help="Number of MCTS traces per step"
    )
    parser.add_argument(
        "--max_ep_len",
        type=int,
        default=200,
        help="Maximum number of steps per episode",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--c_uct", type=float, default=0.05, help="UCT constant")
    parser.add_argument("--c_pw", type=float, default=1, help="Progressive widening constant")
    parser.add_argument("--kappa", type=float, default=0.5, help="Progressive widening exponent")
    parser.add_argument("--tau", type=float, default=0.1, help="Log visit counts scaling factor")
    parser.add_argument("--alpha", type=float, default=0.1, help="Entropy temperature parameter")
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
        "--window", type=int, default=25, help="Smoothing window for visualization"
    )
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
    episode_returns, timepoints = run_continuous_agent(
        game=args.game,
        n_ep=args.n_ep,
        n_traces=args.n_traces,
        max_ep_len=args.max_ep_len,
        lr=args.lr,
        c_uct=args.c_uct,
        c_pw=args.c_pw,
        kappa=args.kappa,
        tau=args.tau,
        alpha=args.alpha,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        temp=args.temp,
        n_hidden_layers=args.n_hidden_layers,
        n_hidden_units=args.n_hidden_units,
        value_loss_ratio=args.value_ratio,
        seed=args.env_seed,
    )

#    print('Showing best episode with return {}'.format(R_best))
#    Env = make_game(args.game)
#    Env = wrappers.Monitor(Env,os.getcwd() + '/best_episode',force=True)
#    Env.reset()
#    Env.seed(seed_best)
#    for a in a_best:
#        Env.step(a)
#        Env.render()
