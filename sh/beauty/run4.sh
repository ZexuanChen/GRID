python -m src.inference \
  experiment=tiger_inference_flat \
  data_dir=data/amazon_data/beauty \
  semantic_id_path=logs/inference/runs/2025-10-28/17-18-13/pickle/merged_predictions_tensor.pt \
  num_hierarchies=4 \
  ckpt_path=logs/train/runs/2025-10-29/03-04-32/checkpoints/checkpoint_epoch_000_step_003300.ckpt
