
import gym
import argparse

import sys
sys.path.append("../..")

from spinnup_local.utils.run_utils import setup_logger_kwargs
from spinnup_local.ddpg import ddpg
from spinnup_local.ddpg import core

#from spinup.utils.run_utils import setup_logger_kwargs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    output_dir = "/home/ella-lovise/Dokumenter/master_auto_docking/spinnup_local/ddpg/model/ddpg"
    export_name = "v2"
    parser.add_argument('--output-dir', type=str, default=output_dir)
    parser.add_argument('--exp-name', type=str, default=export_name)
    parser.add_argument('--noise-type', type=str, default="norm")
    parser.add_argument('--batch_normalization', type=bool, default=True)
    #parser.add_argument('--act-noise', type=int, default=1)
    args = parser.parse_args()


    logger_kwargs = setup_logger_kwargs(args.output_dir,args.exp_name, args.seed)

    env = gym.make(args.env)

    ddpg.ddpg(lambda : env, actor_critic=core.mlp_actor_critic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs,noise_type = args.noise_type,
         batch_normalization = args.batch_normalization)


