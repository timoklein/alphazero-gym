from copy import deepcopy
from datetime import datetime
import numpy as np
from tqdm import trange
import wandb
import hydra
from omegaconf.dictconfig import DictConfig

from alphazero.agent.losses import A0CLossTuned, A0CLoss
from alphazero.helpers import store_actions
from alphazero.helpers import check_space
from rl.make_game import make_game


@hydra.main(config_path="config", config_name="run_discrete")
def run_discrete_agent(cfg: DictConfig):
    episode_returns = []  # storage
    R_max = -np.inf
    best_actions = None
    actions_list = []

    # Environments
    Env = make_game(cfg.game)
    mcts_env = None

    # set seeds
    np.random.seed(cfg.seed)
    Env.seed(cfg.seed)

    buffer = hydra.utils.instantiate(cfg.buffer)
    t_total = 0  # total steps

    # get environment info
    state_dim, _ = check_space(Env.observation_space)
    num_actions, action_discrete = check_space(Env.action_space)

    assert (
        action_discrete == True
    ), "Can't use discrete agent for continuous action spaces!"

    # set config environment values
    cfg.policy.representation_dim = state_dim[0]
    cfg.policy.num_actions = num_actions[0]

    agent = hydra.utils.instantiate(cfg.agent)

    config = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Environment": Env.unwrapped.spec.id,
        "Environment seed": cfg.seed,
        "Training episodes": cfg.num_train_episodes,
        "Episode length": cfg.max_episode_length,
        "Training epochs": cfg.num_train_epochs,
        "Batch size": cfg.buffer.batch_size,
        "Replay buffer size": cfg.buffer.max_size,
        "Policy temperature": cfg.agent.temperature,
        "MCTS rollouts": cfg.mcts.n_rollouts,
        "UCT constant": cfg.mcts.c_uct,
        "Discount factor": cfg.mcts.gamma,
        "MCTS epsilon greedy": cfg.mcts.epsilon,
        "V target policy": cfg.mcts.V_target_policy,
        "Final selection policy": cfg.agent.final_selection,
        "Network hidden layers": cfg.policy.hidden_dimensions,
        "Optimizer": "Adam"
        if cfg.optimizer._target_ == "torch.optim.Adam"
        else "RMSProp",
        "Learning rate": cfg.optimizer.lr,
        "Policy coefficient": cfg.agent.loss_cfg.policy_coeff,
        "Value coefficient": cfg.agent.loss_cfg.value_coeff,
        "Loss reduction": cfg.agent.loss_cfg.reduction,
    }

    if isinstance(agent.loss, A0CLossTuned):
        config.update(
            {
                "Temperature tau": cfg.agent.loss_cfg.tau,
                "Loss lr": 0.001,
                "Loss type": "A0C loss tuned",
            }
        )
    elif isinstance(agent.loss, A0CLoss):
        config.update(
            {
                "Temperature tau": cfg.agent.loss_cfg.tau,
                "Entropy coeff [alpha]": cfg.agent.loss_cfg.alpha,
                "Loss type": "A0C loss untuned",
            }
        )
    else:
        config["Loss type"] = "AlphaZero loss"

    run = wandb.init(name="AlphaZero Discrete", project="a0c", config=config)

    pbar = trange(cfg.num_train_episodes)
    for ep in pbar:
        state = Env.reset()
        R = 0.0  # Total return counter

        agent.reset_mcts(root_state=state)
        for t in range(cfg.max_episode_length):
            # MCTS step
            # run mcts and extract the root output
            action, s, actions, counts, Qs, V = agent.act(Env=Env, mcts_env=mcts_env)
            buffer.store((s, actions, counts, Qs, V))

            # Make the true step
            state, step_reward, terminal, _ = Env.step(action)
            actions_list.append(action)
            R += step_reward
            t_total += (
                agent.n_rollouts  # total number of environment steps (counts the mcts steps)
            )

            if terminal or t == cfg.max_episode_length - 1:
                if R_max < R:
                    actions_list.insert(0, Env.seed()[0])
                    best_actions = deepcopy(actions_list)
                    R_max = R
                actions_list.clear()
                break
            else:
                agent.mcts_forward(action, state)

        # store the total episode return
        episode_returns.append(R)

        # Train
        info_dict = agent.train(buffer)
        info_dict["Episode reward"] = R
        info_dict["Episode reward"] = R
        if isinstance(agent.loss, A0CLossTuned):
            info_dict["alpha"] = agent.loss.alpha.detach().cpu().item()

        # agent.save_checkpoint(env=Env)

        if run is not None:
            run.log(
                info_dict,
                step=ep,
            )

        reward = np.round(R, 2)
        pbar.set_description(f"{ep=}, {reward=}, {t_total=}")
    # Return results
    return episode_returns, best_actions


if __name__ == "__main__":
    episode_returns, best_actions = run_discrete_agent()
