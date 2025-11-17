python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/inference/runs/2025-11-04/16-32-38/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4
