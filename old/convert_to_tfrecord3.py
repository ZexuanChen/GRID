import tensorflow as tf
import numpy as np
import os
from collections import defaultdict

# ----------------- SparseTensor 特征 -----------------
def create_sparse_tensor_feature(values):
    """
    将Python列表转换为 SparseTensor 并创建 TFRecord Feature
    保持原始 TFRecord 的存储方式（VarLenFeature / SparseTensor）
    """
    if not isinstance(values, (list, np.ndarray)):
        values = [values]
    # 转 int64
    values = [int(v) for v in values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values)) # 不用存成稀疏张量

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

# ----------------- 将 sequences 写入 TFRecord -----------------
def convert_sequences_to_tfrecord(sequences, output_file, split_name="train"):
    """
    将序列数据转换并保存为 TFRecord.gz 文件
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    data = sequences[split_name]
    user_ids = data["userId"]
    item_sequences = data["sequence_data"]
    
    with tf.io.TFRecordWriter(output_file, options="GZIP") as writer:
        for i, (user_id, sequence_data) in enumerate(zip(user_ids, item_sequences)):
            example = create_tfrecord_example(sequence_data, user_id)
            writer.write(example.SerializeToString())
            if i % 1000 == 0:
                print(f"已处理 {i} 条记录...")
    
    print(f"成功保存到 {output_file}")

# ----------------- 批量写入自定义数据 -----------------
def write_custom_tfrecord(filename, data_list):
    """
    data_list: list of dict, 每个 dict 包含 user_id 和 sequence_data
        e.g., [{"user_id":123, "sequence_data":[1,2,3]}, ...]
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with tf.io.TFRecordWriter(filename, options="GZIP") as writer:
        for data in data_list:
            example = create_tfrecord_example(data["sequence_data"], data["user_id"])
            writer.write(example.SerializeToString())
    print(f"Saved {len(data_list)} examples to {filename}")

# ----------------- 读取 TFRecord 验证 -----------------
def read_and_verify_tfrecord(tfrecord_file, max_examples=5):
    print(f"验证文件: {tfrecord_file}")
    
    raw_dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP")
    
    # 手动写 feature_description
    feature_description = {
        "user_id": tf.io.VarLenFeature(tf.int64),
        "sequence_data": tf.io.VarLenFeature(tf.int64)
    }
    
    for i, raw_record in enumerate(raw_dataset.take(max_examples)):
        parsed = tf.io.parse_single_example(raw_record, feature_description)
        user_id = tf.sparse.to_dense(parsed["user_id"]).numpy()
        seq = tf.sparse.to_dense(parsed["sequence_data"]).numpy()
        print(f"\n--- Example {i} ---")
        print("user_id:", user_id, "sequence_data:", seq)

# ----------------- 原始数据处理 -----------------
def train_test_split_and_save(raw_dir, output_dir, max_seq_len=20):
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
            if len(parts) < 3:
                print('序列长度小于3')
                exit()
                continue
                
            user_id = int(parts[0])
            items = [int(x)-1 for x in parts[1].strip().split()] # -1从0开始！
            user_ids.append(user_id)

            sequences["training"]["sequence_data"].append(items[:-2])
            sequences["evaluation"]["sequence_data"].append(items[-(max_seq_len+2):-1])
            sequences["testing"]["sequence_data"].append(items[-(max_seq_len+1):])

    for sp in splits:
        sequences[sp]["userId"] = user_ids
        
    print("数据统计:")
    for sp in splits:
        print(f"  {sp}: {len(sequences[sp]['userId'])} 条记录")

    for split_name in splits:
        output_file = os.path.join(output_dir, split_name, "partition_0.tfrecord.gz")
        print(f"转换 {split_name} 数据到 {output_file}")
        convert_sequences_to_tfrecord(sequences, output_file, split_name)

    return sequences

# ----------------- 主程序 -----------------
if __name__ == "__main__":
    raw_dir = "/home/yfu/public_data/MRS/MicroLens/MicroLens-100k"
    output_dir = "/home/yfu/code/GR/GRID/data/microlens"

    # 原始数据处理
    # sequences = train_test_split_and_save(raw_dir, output_dir, max_seq_len=20)
    
    # 验证生成的文件
    for split in ["training", "evaluation", "testing"]:
        print(split)
        tfrecord_file = os.path.join(output_dir, split, "partition_0.tfrecord.gz")
        if os.path.exists(tfrecord_file):
            read_and_verify_tfrecord(tfrecord_file)

    # # ----------------- 自定义数据示例 -----------------
    # custom_data = [
    #     {"user_id": 123, "sequence_data": [1,2,3,4,22]},
    #     {"user_id": 124, "sequence_data": [5,6,7,8]}
    # ]
    # custom_tfrecord = os.path.join(output_dir, "custom", "custom_data.tfrecord.gz")
    # write_custom_tfrecord(custom_tfrecord, custom_data)
    # read_and_verify_tfrecord(custom_tfrecord)
