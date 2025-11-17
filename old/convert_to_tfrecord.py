import tensorflow as tf
import numpy as np
import os
from collections import defaultdict


def create_sparse_tensor_feature(values):
    """
    将Python列表转换为SparseTensor并创建TFRecord特征
    """
    if not isinstance(values, (list, np.ndarray)):
        values = [values]
    
    # 创建SparseTensor
    # indices: 非零元素的索引 (对于1D tensor，就是 [[0], [1], [2], ...])
    indices = [[i] for i in range(len(values))]
    
    # 转换为int64
    values = [int(v) for v in values]
    
    # dense_shape: tensor的完整形状
    dense_shape = [len(values)]
    
    # 创建SparseTensor
    sparse_tensor = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape
    )
    
    # 序列化SparseTensor
    serialized = tf.io.serialize_sparse(sparse_tensor)
    
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized.numpy().tobytes()]))


def create_tfrecord_example(sequence_data, user_id):
    """
    创建单个TFRecord样本
    """
    features = {}
    
    # 处理sequence_data (列表转sparse tensor)
    if sequence_data:
        features["sequence_data"] = create_sparse_tensor_feature(sequence_data)
    
    # 处理user_id (单个值转sparse tensor)
    features["user_id"] = create_sparse_tensor_feature([user_id])
    
    # 创建Example
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def convert_sequences_to_tfrecord(sequences, output_file, split_name="train"):
    """
    将序列数据转换并保存为TFRecord.gz文件
    
    Args:
        sequences: 包含数据的字典，格式如 sequences[split_name]
        output_file: 输出文件路径
        split_name: 数据集分割名称 ("train", "eval", "test")
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 获取对应split的数据
    data = sequences[split_name]
    
    # 直接使用字典格式的数据
    user_ids = data["userId"]
    item_sequences = data["itemId"]
    
    # 写入TFRecord文件
    with tf.io.TFRecordWriter(output_file, options="GZIP") as writer:
        for i, (user_id, sequence_data) in enumerate(zip(user_ids, item_sequences)):
            # 过滤掉padding值(-1)
            filtered_sequence = [item for item in sequence_data if item != -1]
            
            if filtered_sequence:  # 只保存非空序列
                example = create_tfrecord_example(filtered_sequence, user_id)
                writer.write(example.SerializeToString())
                
                if i % 1000 == 0:
                    print(f"已处理 {i} 条记录...")
    
    print(f"成功保存到 {output_file}")


def train_test_split_and_save(raw_dir, output_dir, max_seq_len=20):
    """
    完整的数据处理流程：读取原始数据 -> 分割 -> 转换为TFRecord
    """
    splits = ["train", "eval", "test"]
    sequences = {sp: defaultdict(list) for sp in splits}
    user_ids = []

    # 读取数据
    input_file = os.path.join(raw_dir, "MicroLens-100k_pairs.tsv")
    print(f"读取数据文件: {input_file}")
    
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            # 1. 按制表符分割：第一列是用户，第二列是整串物品
            parts = line.split('\t')
            if len(parts) < 2:
                continue
                
            user_id = int(parts[0])
            item_str = parts[1].strip()

            # 2. 物品串按空格分割
            try:
                # items = [int(x) - 1 for x in item_str.split()]  # 从0开始
                items = [int(x) for x in item_str.split()]  # 从1开始
            except ValueError:
                print(f"跳过第 {line_num} 行，无法解析物品ID: {item_str}")
                continue
            
            if len(items) < 3:  # 至少需要3个物品才能做训练/验证/测试
                continue

            user_ids.append(user_id)

            # 训练集：前 n-2 预测 n-2
            train_items = items[:-2]
            sequences["train"]["itemId"].append(train_items)
            sequences["train"]["itemId_fut"].append(items[-2])

            # 验证集：最后 max_seq_len 段预测 n-2
            eval_items = items[-(max_seq_len + 2):-2]
            padded_eval = eval_items + [-1] * (max_seq_len - len(eval_items))
            sequences["eval"]["itemId"].append(padded_eval)
            sequences["eval"]["itemId_fut"].append(items[-2])

            # 测试集：最后 max_seq_len 段预测 n-1
            test_items = items[-(max_seq_len + 1):-1]
            padded_test = test_items + [-1] * (max_seq_len - len(test_items))
            sequences["test"]["itemId"].append(padded_test)
            sequences["test"]["itemId_fut"].append(items[-1])

    # 为每个split添加userId
    for sp in splits:
        sequences[sp]["userId"] = user_ids
        
    print(f"数据统计:")
    for sp in splits:
        print(f"  {sp}: {len(sequences[sp]['userId'])} 条记录")

    # 转换并保存为TFRecord文件
    for split_name in splits:
        output_file = os.path.join(output_dir, f"{split_name}_partition_0.tfrecord.gz")
        print(f"转换 {split_name} 数据到 {output_file}")
        convert_sequences_to_tfrecord(sequences, output_file, split_name)

    return sequences


def read_and_verify_tfrecord(tfrecord_file, max_examples=3):
    """
    读取并验证TFRecord文件内容
    """
    print(f"验证文件: {tfrecord_file}")
    
    # 读取TFRecord
    raw_dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP")
    
    for i, raw_record in enumerate(raw_dataset.take(max_examples)):
        # 解析样本
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print(f"\n--- Example {i} ---")
        
        for key, feature in example.features.feature.items():
            if feature.HasField("bytes_list"):
                # 反序列化SparseTensor
                serialized_sparse = feature.bytes_list.value[0]
                sparse_tensor = tf.io.deserialize_sparse(
                    serialized_sparse, 
                    out_type=tf.int64
                )
                print(f"{key}-sparsetensor: {sparse_tensor.values.numpy()}")
            else:
                print(f"{key}: {feature}")


if __name__ == "__main__":
    # 配置路径
    raw_dir = "/home/yfu/public_data/MRS/MicroLens/MicroLens-100k"
    output_dir = "/home/yfu/code/GR/GRID/data/microlens/processed"
    
    # 执行完整流程
    sequences = train_test_split_and_save(raw_dir, output_dir, max_seq_len=20)
    
    # 验证生成的文件
    for split in ["train", "eval", "test"]:
        tfrecord_file = os.path.join(output_dir, f"{split}_partition_0.tfrecord.gz")
        if os.path.exists(tfrecord_file):
            read_and_verify_tfrecord(tfrecord_file)