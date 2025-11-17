try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import numpy as np
import os
from collections import defaultdict


def create_int64_list_feature(values):
    """
    创建 int64 列表特征
    """
    if not isinstance(values, (list, np.ndarray)):
        values = [values]
    
    # 过滤掉无效值（-1）
    values = [int(v) for v in values if v != -1]
    
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def create_tfrecord_example(sequence_data, user_id):
    """
    创建单个TFRecord样本，只包含 sequence_data 和 user_id
    """
    features = {}
    
    # 处理sequence_data (列表)
    features["sequence_data"] = create_int64_list_feature(sequence_data)
    
    # 处理user_id (单个值)
    features["user_id"] = create_int64_list_feature([user_id])
    
    # 创建Example
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def convert_sequences_to_tfrecord(sequences, output_file, split_name="train"):
    """
    将序列数据转换并保存为TFRecord.gz文件
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 获取对应split的数据
    data = sequences[split_name]
    user_ids = data["userId"]
    item_sequences = data["itemId"]
    
    # 写入TFRecord文件
    with tf.io.TFRecordWriter(output_file, options="GZIP") as writer:
        for i, (user_id, sequence_data) in enumerate(zip(user_ids, item_sequences)):
            # 创建TFRecord样本
            example = create_tfrecord_example(sequence_data, user_id)
            writer.write(example.SerializeToString())
                
            if i % 1000 == 0:
                print(f"已处理 {i} 条记录...")
    
    print(f"成功保存到 {output_file}，共 {len(user_ids)} 条记录")


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
                items = [int(x) - 1 for x in item_str.split()]  # 从0开始
            except ValueError:
                print(f"跳过第 {line_num} 行，无法解析物品ID: {item_str}")
                continue
            
            if len(items) < 3:  # 至少需要3个物品才能做训练/验证/测试
                continue

            user_ids.append(user_id)

            # 训练集：前 n-2 个物品作为序列
            train_items = items[:-2]
            sequences["train"]["itemId"].append(train_items)

            # 验证集：最后 max_seq_len 段预测 n-2
            eval_items = items[-(max_seq_len + 2):-2]
            # 如果序列长度不足，用 -1 填充
            if len(eval_items) < max_seq_len:
                eval_items = eval_items + [-1] * (max_seq_len - len(eval_items))
            sequences["eval"]["itemId"].append(eval_items)

            # 测试集：最后 max_seq_len 段预测 n-1
            test_items = items[-(max_seq_len + 1):-1]
            # 如果序列长度不足，用 -1 填充
            if len(test_items) < max_seq_len:
                test_items = test_items + [-1] * (max_seq_len - len(test_items))
            sequences["test"]["itemId"].append(test_items)

    # 为每个split添加userId
    for sp in splits:
        sequences[sp]["userId"] = user_ids
        
    print(f"数据统计:")
    for sp in splits:
        print(f"  {sp}: {len(sequences[sp]['userId'])} 条记录")

    # 转换并保存为TFRecord文件
    for split_name in splits:
        output_file = os.path.join(output_dir, f"{split_name}", "partition_0.tfrecord.gz")
        print(f"转换 {split_name} 数据到 {output_file}")
        convert_sequences_to_tfrecord(sequences, output_file, split_name)

    return sequences


def read_and_verify_tfrecord(tfrecord_file, max_examples=3):
    """
    读取并验证TFRecord文件内容
    """
    print(f"验证文件: {tfrecord_file}")
    
    # 创建feature描述
    feature_description = {
        'sequence_data': tf.VarLenFeature(tf.int64),
        'user_id': tf.VarLenFeature(tf.int64)
    }
    
    # 读取TFRecord
    raw_dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP")
    parsed_dataset = raw_dataset.map(lambda x: tf.parse_single_example(x, feature_description))
    
    # 创建迭代器
    iterator = parsed_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    
    with tf.Session() as sess:
        for i in range(max_examples):
            try:
                features = sess.run(next_element)
                print(f"\n--- Example {i} ---")
                
                for key, value in features.items():
                    # VarLenFeature 返回 SparseTensorValue
                    if hasattr(value, 'values'):
                        print(f"{key}: {value.values}")
                    else:
                        print(f"{key}: {value}")
                        
            except tf.errors.OutOfRangeError:
                print(f"已读取完所有数据，共 {i} 条")
                break


if __name__ == "__main__":
    # 配置路径
    raw_dir = "/home/yfu/public_data/MRS/MicroLens/MicroLens-100k"
    output_dir = "/home/yfu/code/GR/GRID/data/microlens"
    
    # 执行完整流程
    sequences = train_test_split_and_save(raw_dir, output_dir, max_seq_len=20)
    
    # 验证生成的文件
    for split in ["train", "eval", "test"]:
        tfrecord_file = os.path.join(output_dir, split, "partition_0.tfrecord.gz")
        if os.path.exists(tfrecord_file):
            read_and_verify_tfrecord(tfrecord_file)