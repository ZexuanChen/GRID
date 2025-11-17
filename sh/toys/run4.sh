python -m src.inference \
  experiment=tiger_inference_flat \
  data_dir=data/amazon_data/toys \
  semantic_id_path=logs/inference/runs/2025-09-21/12-48-45/pickle/merged_predictions_tensor.pt \
  num_hierarchies=4 \
  ckpt_path=logs/train/runs/2025-09-21/12-42-18/checkpoints/checkpoint_000_000030.ckpt
