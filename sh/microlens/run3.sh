# python -m src.train experiment=tiger_train_flat \
#     data_dir=data/microlens \
#     semantic_id_path=logs/inference/runs/2025-09-30/12-16-27/pickle/merged_predictions_tensor.pt \
#     num_hierarchies=4  


# python -m src.train experiment=tiger_train_flat \
#     data_dir=data/microlens \
#     semantic_id_path=logs/inference/runs/2025-09-30/15-53-19/pickle/merged_predictions_tensor.pt \
#     num_hierarchies=4

# text-cv2
# python -m src.train experiment=tiger_train_flat \
#     data_dir=data/microlens \
#     semantic_id_path=logs/inference/runs/2025-09-30/20-19-31/pickle/merged_predictions_tensor.pt \
#     num_hierarchies=4

# text
# python -m src.train experiment=tiger_train_flat \
#     data_dir=data/microlens \
#     semantic_id_path=logs/inference/runs/2025-09-30/20-15-41/pickle/merged_predictions_tensor.pt \
#     num_hierarchies=4

python -m src.train experiment=tiger_train_flat \
    data_dir=data/microlens \
    semantic_id_path=logs/inference/runs/2025-09-30/20-15-41/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 \
    train=true \
    trainer.max_steps=100 \
    