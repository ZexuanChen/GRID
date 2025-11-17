python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/beauty \
    semantic_id_path=logs/inference/runs/2025-10-28/17-18-13/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4

# python -m src.train experiment=tiger_train_flat \
#     data_dir=data/amazon_data/beauty \
#     semantic_id_path=logs/inference/runs/2025-09-19/20-58-18/pickle/merged_predictions_tensor.pt \
#     num_hierarchies=4 \
#     trainer.val_check_interval=1

# 上面是多少个step验证一次