import os
import sys

current_file_path = os.path.realpath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)

print(f'Project root: {project_root}')

from envs.gymloong import LEGGED_GYM_ROOT_DIR
from envs.gymloong.utils.wandb_helper import wandb_helper
from envs.gymloong.utils.task_registry import task_registry
from envs.gymloong.utils.helpers import get_args
from datetime import datetime
import wandb


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    if args.wandb_name:
        experiment_name = args.wandb_name
    else:
        experiment_name = f'{args.task}'

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

    # Check if we specified that we want to use wandb
    do_wandb = train_cfg.do_wandb if hasattr(train_cfg, 'do_wandb') else False
    # Do the logging only if wandb requirements have been fully specified
    do_wandb = do_wandb and None not in (args.wandb_project, args.wandb_entity)

    if do_wandb:
        wandb.config = {}

        if hasattr(train_cfg, 'wandb'):
            what_to_log = train_cfg.wandb.what_to_log
            wandb_helper.craft_log_config(env_cfg, train_cfg, wandb.config, what_to_log)

        print(f'Received WandB project name: {args.wandb_project}\nReceived WandB entitiy name: {args.wandb_entity}\n')
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   group=args.wandb_group,
                   config=wandb.config,
                   name=experiment_name)

        ppo_runner.configure_wandb(wandb)
        ppo_runner.configure_learn(train_cfg.runner.max_iterations, True)
        ppo_runner.learn()

        wandb.finish()
    else:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args)