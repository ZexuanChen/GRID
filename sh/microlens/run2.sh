# python -m src.inference \
#   experiment=rkmeans_inference_flat \
#   data_dir=data/microlens \
#   embedding_path=logs/inference/runs/2025-09-29/21-10-01/pickle/merged_predictions_tensor.pt \
#   embedding_dim=2048 \
#   num_hierarchies=3 \
#   codebook_width=256 \
#   ckpt_path=logs/train/runs/2025-09-29/21-35-07/checkpoints/checkpoint_000_000030.ckpt


# python -m src.inference \
#   experiment=rkmeans_inference_flat \
#   data_dir=data/microlens \
#   embedding_path=data/microlens/items/text_cv_emb.pt \
#   embedding_dim=1152 \
#   num_hierarchies=3 \
#   codebook_width=256 \
#   ckpt_path=logs/train/runs/2025-09-30/15-48-54/checkpoints/checkpoint_000_000030.ckpt

# python -m src.inference \
#   experiment=rkmeans_inference_flat \
#   data_dir=data/microlens \
#   embedding_path=data/microlens/items/text_cv_emb2.pt \
#   embedding_dim=2048 \
#   num_hierarchies=3 \
#   codebook_width=256 \
#   ckpt_path=logs/train/runs/2025-09-30/18-16-27/checkpoints/checkpoint_000_000030.ckpt


python -m src.inference \
  experiment=rkmeans_inference_flat \
  data_dir=data/microlens \
  embedding_path=data/microlens/items/text_emb.pt \
  embedding_dim=1024 \
  num_hierarchies=3 \
  codebook_width=256 \
  ckpt_path=logs/train/runs/2025-09-30/19-35-42/checkpoints/checkpoint_000_000030.ckpt
