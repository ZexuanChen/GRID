python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/sports \
    embedding_path=logs/inference/runs/2025-09-21/12-26-52/pickle/merged_predictions_tensor.pt \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256