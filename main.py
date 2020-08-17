from torch import optim
import numpy as np
from tqdm import trange
import argparse
import os
import time
from torch.utils.tensorboard import SummaryWriter

from alphazero import Model, alphaZero_loss, train, MCTS
from helpers import (argmax,check_space,is_atari_game,copy_atari_state,store_safely,
restore_atari_state,stable_normalizer,smooth,symmetric_remove,ReplayBuffer)
from rl.make_game import make_game


#### Run the agentAgent ##
# TODO: refactor this
# TODO: Add tensorboard for loss, reward and 
def run(game,n_ep,n_mcts,max_ep_len,lr,c,gamma,data_size,batch_size,temp,n_hidden_layers,n_hidden_units):
    ''' Outer training loop '''
    episode_returns = [] # storage
    timepoints = []
    # Environments
    Env = make_game(game)
    is_atari = is_atari_game(Env)
    mcts_env = make_game(game) if is_atari else None

    tb = SummaryWriter(log_dir="logs")
    D = ReplayBuffer(max_size=data_size,batch_size=batch_size)
    model = Model(Env,n_hidden_layers,n_hidden_units)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=.9, eps=1e-07) # initialize w tf defaults
    t_total = 0 # total steps   
    R_best = -np.Inf
    
    pbar = trange(n_ep)
    for ep in pbar:    
        start = time.time()
        s = Env.reset() 
        R = 0.0 # Total return counter
        a_store = []
        seed = np.random.randint(1e7) # draw some Env seed
        Env.seed(seed)      
        if is_atari: 
            mcts_env.reset()
            mcts_env.seed(seed)                                

        mcts = MCTS(root_index=s,root=None,model=model,na=model.action_dim,gamma=gamma) # the object responsible for MCTS searches                             
        for t in range(max_ep_len):
            # MCTS step
            mcts.search(n_mcts=n_mcts,c=c,Env=Env,mcts_env=mcts_env) # perform a forward search
            state,pi,V = mcts.return_results(temp) # extract the root output
            D.store((state,V,pi))

            # Make the true step
            a = np.random.choice(len(pi),p=pi)
            a_store.append(a)
            s1,r,terminal,_ = Env.step(a)
            R += r
            t_total += n_mcts # total number of environment steps (counts the mcts steps)                

            if terminal:
                break
            else:
                mcts.forward(a,s1)
        
        # store the total episode return
        tb.add_scalar("Episode reward", R, ep)
        episode_returns.append(R) 
        timepoints.append(t_total) # store the timestep count of the episode return
        store_safely(os.getcwd(),'result',{'R':episode_returns,'t':timepoints})  

        if R > R_best:
            a_best = a_store
            seed_best = seed
            R_best = R
        # print(f"Finished episode {ep}, total return: {np.round(R,2)}, total time: {np.round((time.time()-start),1)} sec")
        # Train
        loss = train(model, optimizer, D, alphaZero_loss)
        tb.add_scalar("Training loss", loss, ep)


        reward = np.round(R,2)
        e_time = np.round((time.time()-start),1)
        pbar.set_description(f"{ep=}, {reward=}, {e_time=}s")
    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best

#### Command line call, parsing and plotting ##
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='CartPole-v0',help='Training environment')
    parser.add_argument('--n_ep', type=int, default=500, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=25, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=300, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')

    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=128, help='Number of units per hidden layers in NN')

    
    args = parser.parse_args()
    episode_returns,timepoints,a_best,seed_best,R_best = run(game=args.game,n_ep=args.n_ep,n_mcts=args.n_mcts,
                                        max_ep_len=args.max_ep_len,lr=args.lr,c=args.c,gamma=args.gamma,
                                        data_size=args.data_size,batch_size=args.batch_size,temp=args.temp,
                                        n_hidden_layers=args.n_hidden_layers,n_hidden_units=args.n_hidden_units)

    # Finished training: Visualize
    # fig,ax = plt.subplots(1,figsize=[7,5])
    # total_eps = len(episode_returns)
    # episode_returns = smooth(episode_returns,args.window,mode='valid') 
    # ax.plot(symmetric_remove(np.arange(total_eps),args.window-1),episode_returns,linewidth=4,color='darkred')
    # ax.set_ylabel('Return')
    # ax.set_xlabel('Episode',color='darkred')
    # plt.savefig(os.getcwd()+'/learning_curve.png',bbox_inches="tight",dpi=300)
    
#    print('Showing best episode with return {}'.format(R_best))
#    Env = make_game(args.game)
#    Env = wrappers.Monitor(Env,os.getcwd() + '/best_episode',force=True)
#    Env.reset()
#    Env.seed(seed_best)
#    for a in a_best:
#        Env.step(a)
#        Env.render()
