from torch import optim
import numpy as np
from tqdm import trange
import argparse
import os
import time
from torch.utils.tensorboard import SummaryWriter

from alphazero import AlphaZeroAgent
from helpers import (
    argmax,
    check_space,
    is_atari_game,
    copy_atari_state,
    store_safely,
    restore_atari_state,
    stable_normalizer,
    smooth,
    symmetric_remove,
    ReplayBuffer,
)
from rl.make_game import make_game


#### Run the agentAgent ##
# TODO: refactor this
def run(
    game,
    n_ep,
    n_traces,
    max_ep_len,
    lr,
    c,
    gamma,
    data_size,
    batch_size,
    temp,
    n_hidden_layers,
    n_hidden_units,
):
    """ Outer training loop """
    episode_returns = []  # storage
    timepoints = []
    # Environments
    Env = make_game(game)
    is_atari = is_atari_game(Env)
    mcts_env = make_game(game) if is_atari else None

    tb = SummaryWriter(log_dir="logs")
    Buffer = ReplayBuffer(max_size=data_size, batch_size=batch_size)
    t_total = 0  # total steps
    R_best = -np.Inf

    agent = AlphaZeroAgent(
        Env,
        n_hidden_layers=n_hidden_layers,
        n_hidden_units=n_hidden_units,
        n_traces=n_traces,
        lr=lr,
        temperature=temp,
        c_uct=c,
        gamma=gamma,
    )

    pbar = trange(n_ep)
    for ep in pbar:
        start = time.time()
        state = Env.reset()
        R = 0.0  # Total return counter
        action_store = []
        seed = np.random.randint(1e7)  # draw some Env seed
        Env.seed(seed)
        if is_atari:
            mcts_env.reset()
            mcts_env.seed(seed)

        agent.reset_mcts(root_index=state)
        for t in range(max_ep_len):
            # MCTS step
            # run mcts and extract the root output
            s, pi, V = agent.search(Env=Env, mcts_env=mcts_env)
            Buffer.store((s, V, pi))

            # Make the true step
            # We sample here from the policy according tot he policy's probabilities
            action = np.random.choice(len(pi), p=pi)
            action_store.append(action)
            new_state, step_reward, terminal, _ = Env.step(action)
            R += step_reward
            t_total += (
                n_traces  # total number of environment steps (counts the mcts steps)
            )

            if terminal:
                break
            else:
                agent.mcts_forward(action, new_state)

        # store the total episode return
        tb.add_scalar("Episode reward", R, ep)
        episode_returns.append(R)
        timepoints.append(t_total)  # store the timestep count of the episode return
        store_safely(os.getcwd(), "result", {"R": episode_returns, "t": timepoints})

        if R > R_best:
            a_best = action_store
            seed_best = seed
            R_best = R

        # Train
        Buffer.reshuffle()
        running_loss = []
        batches = 0
        for epoch in range(1):
            for obs in Buffer:
                loss = agent.update(obs)
                running_loss.append(loss)
                batches += 1
        episode_loss = sum(running_loss) / batches
        tb.add_scalar("Training loss", episode_loss, ep)

        reward = np.round(R, 2)
        e_time = np.round((time.time() - start), 1)
        pbar.set_description(f"{ep=}, {reward=}, {e_time=}s")
    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="CartPole-v0", help="Training environment")
    parser.add_argument("--n_ep", type=int, default=500, help="Number of episodes")
    parser.add_argument(
        "--n_traces", type=int, default=25, help="Number of MCTS traces per step"
    )
    parser.add_argument(
        "--max_ep_len",
        type=int,
        default=300,
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
        "--data_size", type=int, default=1000, help="Dataset size (FIFO)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Minibatch size")
    parser.add_argument(
        "--window", type=int, default=25, help="Smoothing window for visualization"
    )

    parser.add_argument(
        "--n_hidden_layers", type=int, default=2, help="Number of hidden layers in NN"
    )
    parser.add_argument(
        "--n_hidden_units",
        type=int,
        default=128,
        help="Number of units per hidden layers in NN",
    )

    args = parser.parse_args()
    episode_returns, timepoints, a_best, seed_best, R_best = run(
        game=args.game,
        n_ep=args.n_ep,
        n_traces=args.n_traces,
        max_ep_len=args.max_ep_len,
        lr=args.lr,
        c=args.c,
        gamma=args.gamma,
        data_size=args.data_size,
        batch_size=args.batch_size,
        temp=args.temp,
        n_hidden_layers=args.n_hidden_layers,
        n_hidden_units=args.n_hidden_units,
    )

#    print('Showing best episode with return {}'.format(R_best))
#    Env = make_game(args.game)
#    Env = wrappers.Monitor(Env,os.getcwd() + '/best_episode',force=True)
#    Env.reset()
#    Env.seed(seed_best)
#    for a in a_best:
#        Env.step(a)
#        Env.render()
