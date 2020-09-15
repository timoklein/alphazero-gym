import numpy as np
from tqdm import trange
import wandb
import hydra
from omegaconf.dictconfig import DictConfig

from alphazero.losses import A0CLossTuned, A0CLoss
from alphazero.helpers import check_space, store_actions
from rl.make_game import make_game


@hydra.main(config_path="config", config_name="run_continuous")
def run_continuous_agent(cfg: DictConfig):
    episode_returns = []  # storage
    R_max = -np.inf
    best_actions = None
    actions_list = []

    # Environments
    Env = make_game(cfg.game)

    # set seeds
    np.random.seed(cfg.seed)
    Env.seed(cfg.seed)

    buffer = hydra.utils.instantiate(cfg.buffer)
    t_total = 0  # total steps

    # get environment info
    state_dim, _ = check_space(Env.observation_space)
    action_dim, action_discrete = check_space(Env.action_space)

    assert (
        action_discrete == False
    ), "Using continuous agent for a discrete action space!"

    # set config environment values
    cfg.network.state_dim = state_dim[0]
    cfg.network.action_dim = action_dim[0]
    # assumes that all dimensions of the action space are equally bound
    cfg.network.act_limit = float(Env.action_space.high[0])

    agent = hydra.utils.instantiate(cfg.agent)

    config = {
        "Environment": Env.unwrapped.spec.id,
        "Environment seed": cfg.seed,
        "Batch size": cfg.buffer.batch_size,
        "Replay buffer size": cfg.buffer.max_size,
        "MCTS rollouts": cfg.mcts.n_rollouts,
        "UCT constant": cfg.mcts.c_uct,
        "Discount factor": cfg.mcts.gamma,
        "Progressive widening factor [c_pw]": cfg.mcts.c_pw,
        "Progressive widening exponent [kappa]": cfg.mcts.kappa,
        "Network hidden layers": cfg.network.n_hidden_layers,
        "Network hidden units": cfg.network.n_hidden_units,
        "Number of GMM components": cfg.network.num_components,
        "Learning rate": cfg.network_optimizer.lr,
        "Log counts scaling factor [tau]": cfg.agent.loss_cfg.tau,
        "Policy Coefficient": cfg.agent.loss_cfg.policy_coeff,
        "Value coefficient": cfg.agent.loss_cfg.value_coeff,
        "Loss reduction": cfg.agent.loss_cfg.reduction,
    }

    if isinstance(agent.loss, A0CLossTuned):
        config.update({"Loss lr": 0.001, "Loss type": "A0C loss tuned"})
    elif isinstance(agent.loss, A0CLoss):
        config.update(
            {
                "Entropy coeff [alpha]": cfg.agent.loss_cfg.alpha,
                "Loss type": "A0C loss untuned",
            }
        )

    run = wandb.init(name="A0C", project="a0c", config=config)

    pbar = trange(cfg.num_train_episodes)
    for ep in pbar:
        state = Env.reset()
        R = 0.0  # Total return counter

        agent.reset_mcts(root_state=state)
        for t in range(cfg.max_episode_length):
            # MCTS step
            # run mcts and extract the root output
            action, s, actions, counts, V = agent.act(Env=Env, mcts_env=None)
            buffer.store((s, actions, counts, V))

            # Make the true step
            state, step_reward, terminal, _ = Env.step(action)
            actions_list.append(action)

            R += step_reward
            t_total += (
                agent.n_rollouts  # total number of environment steps (counts the mcts steps)
            )

            if terminal or t == cfg.max_episode_length - 1:
                if R_max < R:
                    actions_list.insert(0, Env.seed())
                    best_actions = actions_list
                    R_max = R
                    store_actions(cfg.game, best_actions)
                actions_list.clear()
                break
            else:
                # reset the mcts as we can't reuse the tree
                agent.reset_mcts(root_state=state)

        # store the total episode return
        episode_returns.append(R)

        # Train
        info_dict = agent.train(buffer)
        # agent.save_checkpoint(env=Env)

        info_dict["Episode reward"] = R
        if isinstance(agent.loss, A0CLossTuned):
            info_dict["alpha"] = agent.loss.alpha

        run.log(
            info_dict, step=ep,
        )

        reward = np.round(R, 2)
        pbar.set_description(f"{ep=}, {reward=}, {t_total=}")
    # Return results
    return episode_returns, best_actions


if __name__ == "__main__":
    episode_returns, actions_list = run_continuous_agent()
