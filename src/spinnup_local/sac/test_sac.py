

import core
from sac import sac
import gym



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    output_dir = "/home/ella-lovise/Dokumenter/master_auto_docking/spinnup_local/sac/model/sac"
    export_name = "v1"
    parser.add_argument('--output-dir', type=str, default=output_dir)
    parser.add_argument('--exp-name', type=str, default=export_name)
    parser.add_argument('--layer-norm', type=bool, default=True)
    parser.add_argument('--update_each_time', type=bool, default=True)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.output_dir,args.exp_name, args.seed)

    sac(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs,layer_norm = args.layer_norm,
        update_each_time = args.update_each_time)