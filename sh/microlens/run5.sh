python -m src.train experiment=tiger_train_flat \
    data_dir=data/microlens \
    semantic_id_path=logs/inference/runs/2025-09-30/12-16-27/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 \
    train=true \
    trainer.max_steps=0 \
    ckpt_path=logs/train/runs/2025-09-30/14-10-49/checkpoints/checkpoint_epoch_000_step_003600.ckpt

# text
# python -m src.train experiment=tiger_train_flat \
#     data_dir=data/microlens \
#     semantic_id_path=logs/inference/runs/2025-09-30/20-15-41/pickle/merged_predictions_tensor.pt \
#     num_hierarchies=4 \
#     train=true \
#     trainer.max_steps=0 \
#     ckpt_path=logs/train/runs/2025-10-01/00-23-06/checkpoints/checkpoint_epoch_000_step_003600.ckpt


# text-cv
# python -m src.train experiment=tiger_train_flat \
#     data_dir=data/microlens \
#     semantic_id_path=logs/inference/runs/2025-09-30/20-19-31/pickle/merged_predictions_tensor.pt \
#     num_hierarchies=4 \
#     train=true \
#     trainer.max_steps=0 \
#     ckpt_path=logs/train/runs/2025-10-01/00-18-47/checkpoints/checkpoint_epoch=000_step=004100.ckpt
