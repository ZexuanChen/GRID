# 这个脚本用于将Microlens-100k的sequence数据转换为tfrecord文件
import tensorflow as tf
import numpy as np
import os
from collections import defaultdict

# ----------------- SparseTensor 特征 -----------------
def create_sparse_tensor_feature(values):
    """
    将Python列表转换为 TFRecord Feature
    """
    if not isinstance(values, (list, np.ndarray)):
        values = [values]
    values = [int(v) for v in values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

# ----------------- 创建 Example -----------------
def create_tfrecord_example(sequence_data, user_id):
    """
    创建单个 TFRecord Example
    """
    features = {
        "sequence_data": create_sparse_tensor_feature(sequence_data),
        "user_id": create_sparse_tensor_feature([user_id])
    }
    return tf.train.Example(features=tf.train.Features(feature=features))

# ----------------- 将 sequences 写入多个 TFRecord partition -----------------
def convert_sequences_to_tfrecord_partitions(sequences, output_dir, split_name="train", num_partitions=4):
    """
    将序列数据拆分成多个 partition 并保存为 TFRecord.gz 文件
    """
    os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)
    
    data = sequences[split_name]
    user_ids = data["userId"]
    item_sequences = data["sequence_data"]
    total_records = len(user_ids)
    
    # 每个 partition 的大小
    partition_size = (total_records + num_partitions - 1) // num_partitions
    
    for p in range(num_partitions):
        start_idx = p * partition_size
        end_idx = min((p + 1) * partition_size, total_records)
        partition_user_ids = user_ids[start_idx:end_idx]
        partition_sequences = item_sequences[start_idx:end_idx]
        
        partition_file = os.path.join(output_dir, split_name, f"partition_{p}.tfrecord.gz")
        with tf.io.TFRecordWriter(partition_file, options="GZIP") as writer:
            for user_id, sequence_data in zip(partition_user_ids, partition_sequences):
                example = create_tfrecord_example(sequence_data, user_id)
                writer.write(example.SerializeToString())
        
        print(f"{split_name} partition {p}: 保存 {len(partition_user_ids)} 条记录到 {partition_file}")

# ----------------- 读取并验证 TFRecord -----------------
def read_and_verify_tfrecord_partitions(output_dir, split_name="train", max_examples_per_partition=5, num_partitions=4):
    print(f"验证 {split_name} 数据")
    feature_description = {
        "user_id": tf.io.VarLenFeature(tf.int64),
        "sequence_data": tf.io.VarLenFeature(tf.int64)
    }
    
    for p in range(num_partitions):
        partition_file = os.path.join(output_dir, split_name, f"partition_{p}.tfrecord.gz")
        if not os.path.exists(partition_file):
            print(f"Partition 文件不存在: {partition_file}")
            continue
        
        raw_dataset = tf.data.TFRecordDataset([partition_file], compression_type="GZIP")
        print(f"\n--- {split_name} partition {p} ---")
        for i, raw_record in enumerate(raw_dataset.take(max_examples_per_partition)):
            parsed = tf.io.parse_single_example(raw_record, feature_description)
            user_id = tf.sparse.to_dense(parsed["user_id"]).numpy()
            seq = tf.sparse.to_dense(parsed["sequence_data"]).numpy()
            print(f"Example {i}: user_id={user_id}, sequence_data={seq}")

# ----------------- 原始数据处理 -----------------
def train_test_split_and_save(raw_dir, output_dir, max_seq_len=20, num_partitions=4):
    splits = ["training", "evaluation", "testing"]
    sequences = {sp: defaultdict(list) for sp in splits}
    user_ids = []

    input_file = os.path.join(raw_dir, "MicroLens-100k_pairs.tsv")
    print(f"读取数据文件: {input_file}")
    
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 2:
                continue
                
            user_id = int(parts[0])
            items = [int(x)-1 for x in parts[1].strip().split()] # -1 从0开始
            user_ids.append(user_id)

            sequences["training"]["sequence_data"].append(items[:-2])
            sequences["evaluation"]["sequence_data"].append(items[-(max_seq_len+2):-1])
            sequences["testing"]["sequence_data"].append(items[-(max_seq_len+1):])

    for sp in splits:
        sequences[sp]["userId"] = user_ids

    print("数据统计:")
    for sp in splits:
        print(f"  {sp}: {len(sequences[sp]['userId'])} 条记录")
        convert_sequences_to_tfrecord_partitions(sequences, output_dir, sp, num_partitions=num_partitions)

    return sequences

# # 把序列信息转换成指定的tfrecords格式
if __name__ == "__main__":
    raw_dir = "/home/yfu/public_data/MRS/MicroLens/MicroLens-100k"
    output_dir = "/home/yfu/code/GR/GRID/data/microlens"

    # 生成 partition TFRecord
    sequences = train_test_split_and_save(raw_dir, output_dir, max_seq_len=20, num_partitions=32) # 必须得分区存储不然也会报错stop
    
    # 验证生成的 partition
    for split in ["training", "evaluation", "testing"]:
        read_and_verify_tfrecord_partitions(output_dir, split_name=split, max_examples_per_partition=5, num_partitions=4)
