

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ant OrcaGym environment with APPO training.')
    parser.add_argument('--config_file', type=str, default='legged_local_config.yaml', help='The path of the config file')
    parser.add_argument('--run_mode', type=str, help='The mode to run (training / testing)')
    parser.add_argument('--checkpoint', type=str, help='The path to the checkpoint file for testing. no need for training')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config_file)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.run_mode == 'training':
        config = config['train_legged_local']
    elif args.run_mode == 'testing':
        config = config['test_legged_local']
    else:
        raise ValueError("Invalid run mode. Use 'training' or 'testing'.")

    main(
        config=config,
        run_mode=args.run_mode,
        checkpoint_path=args.checkpoint
    )
