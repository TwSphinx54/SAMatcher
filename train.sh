accelerate launch --config_file configs/accelerate_config.yaml --num_processes 2 train.py \
    --data_cfg_path configs/train_config.py \
    --exp_name SAMatcher \
    --batch_size 2 \
    --num_workers 4 \
    --true_lr 0.0001 \
    --num_epochs 40 \
    --log_every_n_steps 5 \
    --output_dir outputs \
    --mixed_precision no