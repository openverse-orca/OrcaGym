torchrun \
  --standalone \
  --nnodes 1 \
  --nproc_per_node 1 \
  ../../3rd_party/openvla/vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /home/huangwei1/tensorflow_datasets \
  --dataset_name orca_gym_dataset \
  --run_root_dir train_lora \
  --adapter_tmp_dir train_lora/tmp \
  --lora_rank 32 \
  --batch_size 10 \
  --grad_accumulation_steps 1 \
  --learning_rate 0.0005 \
  --image_aug true \
  --wandb_project orca_gym_openvla_test \
  --wandb_entity ""