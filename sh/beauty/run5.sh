# 这个文件和run3.sh有什么区别？
# 这个文件是用来训练tiger模型的，但是只训练少量步数，用来测试是否能正常运行

python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/beauty \
    semantic_id_path=logs/inference/runs/2025-09-19/20-58-18/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 \
    train=true \
    trainer.max_steps=0
    ckpt_path=logs/train/runs/2025-09-21/00-27-07/checkpoints/checkpoint_epoch=000_step=003400.ckpt
# 统计 sh 文件夹里所有脚本中出现过的文件路径，按时间排序
# 注意：以下命令会递归扫描 sh/ 下全部 .sh 文件，提取其中看起来像路径的字符串，
#       然后去重、过滤掉不存在的路径，最后按文件修改时间（mtime）升序输出。

# find sh -type f -name '*.sh' -print0 \
# | xargs -0 grep -oEh '\b[-./a-zA-Z0-9_]+\b' \
# | grep -E '^[./].*' \
# | sort -u \
# | while read -r p; do
#     # 如果路径存在，则打印“修改时间 路径”
#     if [[ -e "$p" ]]; then
#       printf '%s %s\n' "$(stat -c %Y "$p")" "$p"
#     fi
#   done \
# | sort -n \
# | awk '{print $2}'
