python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/sports \
    semantic_id_path=logs/inference/runs/2025-09-21/12-48-27/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4
