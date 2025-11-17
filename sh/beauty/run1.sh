python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=logs/inference/runs/2025-10-28/11-10-30/pickle/merged_predictions_tensor.pt \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256

# python -m src.train experiment=rqvae_train_flat \
#     data_dir=data/amazon_data/beauty \
#     embedding_path=logs/inference/runs/2025-09-19/20-05-32/pickle/merged_predictions_tensor.pt \
#     embedding_dim=2048 \
#     num_hierarchies=3 \
#     codebook_width=256 \
#     trainer.max_steps=30