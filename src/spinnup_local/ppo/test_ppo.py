if __name__ == '__main__':

    output_dir = "/home/ella-lovise/Dokumenter/overview_RL_spescialization_assignment/ppo_spinningup/model_ppo/"
    export_name = "v3"

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    #CartPole-v0
    #parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=2000)

    parser.add_argument('--output-dir', type=str, default=output_dir)
    parser.add_argument('--exp-name', type=str, default=export_name)
    parser.add_argument('--stats-path', type=str, default=output_dir + export_name + "/stats.txt" )
    parser.add_argument('--normalize', type = bool, default = True)


    args = parser.parse_args()

    #mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, args.output_dir)

    print(gym.make(args.env))



    ppo(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        normalize = args.normalize,
        stats_path = args.stats_path)
