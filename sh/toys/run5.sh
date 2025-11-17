python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/sports \
    semantic_id_path=logs/inference/runs/2025-09-21/12-26-52/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 \
    train=true \
    trainer.max_steps=0 \
    ckpt_path=logs/train/runs/2025-09-21/12-42-18/checkpoints/checkpoint_epoch_000_step_003300.ckpt
