import argparse
import subprocess
import sys

import os
import sys
import time
import subprocess
import signal


current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))
finetune_script_path = os.path.join(project_root, '3rd_party', 'openvla', 'vla-scripts', 'finetune.py')

def main():
    parser = argparse.ArgumentParser(description="Train VLA model with LoRA using torchrun")

    # 添加所有原始命令行参数
    parser.add_argument("--standalone", action="store_true", help="Run in standalone mode")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Processes per node")
    parser.add_argument("--vla_path", type=str, default="openvla/openvla-7b", help="Path to pre-trained VLA model")
    parser.add_argument("--data_root_dir", type=str, default="orca_gym_dataset", help="Dataset root directory")
    parser.add_argument("--dataset_name", type=str, default="bridge_orig", help="Dataset name")
    parser.add_argument("--run_root_dir", type=str, default="train_lora", help="Training output root")
    parser.add_argument("--adapter_tmp_dir", type=str, default="train_lora/tmp", help="Adapter temp directory")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--image_aug", action="store_false", help="Disable image augmentation")
    parser.add_argument("--wandb_project", type=str, default="orca_gym_openvla_test", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="", help="WandB entity name")
    parser.add_argument("--save_steps", type=int, default=None, help="Checkpoint save interval")

    args = parser.parse_args()

    # 构建完整的 torchrun 命令
    command = [
        "torchrun",
        "--standalone",
        "--nnodes", str(args.nnodes),
        "--nproc_per_node", str(args.nproc_per_node),
        finetune_script_path,
        f'--vla_path "{args.vla_path}"',
        f'--data_root_dir "{args.data_root_dir}"',
        f'--dataset_name "{args.dataset_name}"',
        f'--run_root_dir "{args.run_root_dir}"',
        f'--adapter_tmp_dir "{args.adapter_tmp_dir}"',
        f'--lora_rank {args.lora_rank}',
        f'--batch_size {args.batch_size}',
        f'--grad_accumulation_steps {args.grad_accumulation_steps}',
        f'--learning_rate {args.learning_rate}',
        f'--image_aug {str(args.image_aug).lower()}',
        f'--wandb_project "{args.wandb_project}"',
        f'--wandb_entity "{args.wandb_entity}"',
    ]

    if args.save_steps is not None:
        command.append(f'--save_steps {args.save_steps}')

    # 执行命令
    try:
        result = subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()