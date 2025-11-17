python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/sports \
    semantic_id_path=logs/inference/runs/2025-09-21/12-48-27/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 \
    train=true \
    trainer.max_steps=0 \
    ckpt_path=logs/train/runs/2025-09-21/16-05-02/checkpoints/checkpoint_epoch=000_step=004300.ckpt
