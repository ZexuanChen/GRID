python -m src.inference \
  experiment=tiger_inference_flat \
  data_dir=data/microlens \
  semantic_id_path=logs/inference/runs/2025-09-30/20-15-41/pickle/merged_predictions_tensor.pt \
  num_hierarchies=4 \
  ckpt_path=logs/train/runs/2025-09-30/14-10-49/checkpoints/checkpoint_epoch_000_step_003600.ckpt
