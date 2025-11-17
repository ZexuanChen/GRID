# python -m src.train experiment=rkmeans_train_flat \
#     data_dir=data/microlens \
#     embedding_path=logs/inference/runs/2025-09-29/21-10-01/pickle/merged_predictions_tensor.pt \
#     embedding_dim=2048 \
#     num_hierarchies=3 \
#     codebook_width=256

# # data/microlens/items/text_cv_emb.pt
# # 1024,1152
# python -m src.train experiment=rkmeans_train_flat \
#     data_dir=data/microlens \
#     embedding_path=data/microlens/items/text_cv_emb.pt \
#     embedding_dim=1152 \
#     num_hierarchies=3 \
#     codebook_width=256

# python -m src.train experiment=rkmeans_train_flat \
#     data_dir=data/microlens \
#     embedding_path=data/microlens/items/text_cv_emb2.pt \
#     embedding_dim=2048 \
#     num_hierarchies=3 \
#     codebook_width=256

python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/microlens \
    embedding_path=data/microlens/items/text_emb.pt \
    embedding_dim=1024 \
    num_hierarchies=3 \
    codebook_width=256