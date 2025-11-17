python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=logs/inference/runs/2025-10-29/08-22-38/pickle/merged_predictions_tensor.pt \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256