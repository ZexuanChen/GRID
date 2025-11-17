"""
整合的数据处理脚本，将用户序列数据和物品数据都转换为 TFRecord 格式
模拟您的 process 方法的完整流程
"""

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import numpy as np
import pandas as pd
import os
from collections import defaultdict


class MicroLensDataProcessor:
    """
    MicroLens 数据处理器，将数据转换为 TFRecord 格式
    """
    
    def __init__(self, raw_dir, output_dir):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
    
    def train_test_split(self, max_seq_len=20):
        """
        用户序列数据分割
        """
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []

        # 读取数据
        input_file = os.path.join(self.raw_dir, "MicroLens-100k_pairs.tsv")
        print(f"读取用户序列数据: {input_file}")
        
        with open(input_file, "r") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                    
                user_id = int(parts[0])
                item_str = parts[1].strip()

                try:
                    items = [int(x) - 1 for x in item_str.split()]  # 从0开始
                except ValueError:
                    continue
                
                if len(items) < 3:
                    continue

                user_ids.append(user_id)

                # 训练集：前 n-2 个物品作为序列
                train_items = items[:-2]
                sequences["train"]["itemId"].append(train_items)

                # 验证集：最后 max_seq_len 段预测 n-2
                eval_items = items[-(max_seq_len + 2):-2]
                if len(eval_items) < max_seq_len:
                    eval_items = eval_items + [-1] * (max_seq_len - len(eval_items))
                sequences["eval"]["itemId"].append(eval_items)

                # 测试集：最后 max_seq_len 段预测 n-1
                test_items = items[-(max_seq_len + 1):-1]
                if len(test_items) < max_seq_len:
                    test_items = test_items + [-1] * (max_seq_len - len(test_items))
                sequences["test"]["itemId"].append(test_items)

        # 为每个split添加userId
        for sp in splits:
            sequences[sp]["userId"] = user_ids
            
        print(f"用户序列数据统计:")
        for sp in splits:
            print(f"  {sp}: {len(sequences[sp]['userId'])} 条记录")

        return sequences
    
    def create_int64_list_feature(self, values):
        """创建 int64 列表特征"""
        if not isinstance(values, (list, np.ndarray)):
            values = [values]
        values = [int(v) for v in values if v != -1]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def create_float_list_feature(self, values):
        """创建 float 列表特征"""
        if not isinstance(values, (list, np.ndarray)):
            values = [values]
        values = [float(v) for v in values]
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def create_bytes_list_feature(self, values):
        """创建 bytes 列表特征"""
        if isinstance(values, str):
            values = [values]
        byte_values = []
        for v in values:
            if isinstance(v, str):
                byte_values.append(v.encode('utf-8'))
            else:
                byte_values.append(v)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=byte_values))
    
    def save_sequence_data_to_tfrecord(self, sequences):
        """
        保存用户序列数据到 TFRecord
        """
        print("\n=== 保存用户序列数据到 TFRecord ===")
        
        for split_name in ["train", "eval", "test"]:
            data = sequences[split_name]
            user_ids = data["userId"]
            item_sequences = data["itemId"]
            
            # 创建输出目录
            split_output_dir = os.path.join(self.output_dir, "sequences", split_name)
            os.makedirs(split_output_dir, exist_ok=True)
            
            output_file = os.path.join(split_output_dir, "partition_0.tfrecord.gz")
            
            print(f"保存 {split_name} 序列数据到: {output_file}")
            
            with tf.python_io.TFRecordWriter(output_file, options=tf.python_io.TFRecordOptions(compression_type="GZIP")) as writer:
                for user_id, sequence_data in zip(user_ids, item_sequences):
                    features = {}
                    features["sequence_data"] = self.create_int64_list_feature(sequence_data)
                    features["user_id"] = self.create_int64_list_feature([user_id])
                    
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())
            
            print(f"  完成，共保存 {len(user_ids)} 条序列记录")
    
    def save_item_data_to_tfrecord(self, items_per_partition=5000):
        """
        保存物品数据到 TFRecord (模拟您的 process 方法)
        """
        print("\n=== 保存物品数据到 TFRecord ===")
        
        # 1. 读取标题数据
        titles_path = os.path.join(self.raw_dir, "MicroLens-100k_title_en.csv")
        print(f"读取标题文件: {titles_path}")
        titles_df = pd.read_csv(titles_path, header=None, names=["id", "title"])
        titles_df = titles_df.sort_values("id")
        
        # 2. 读取嵌入数据
        emb_path = os.path.join(self.raw_dir, "text_embeddings_BgeM3.npz")
        print(f"读取嵌入文件: {emb_path}")
        
        with np.load(emb_path) as data_npz:
            ids_array = data_npz['ids']
            embeddings_array = data_npz['embeddings']
        
        print(f"ID 数组形状: {ids_array.shape}")
        print(f"嵌入向量数组形状: {embeddings_array.shape}")
        
        # 3. 创建输出目录
        items_output_dir = os.path.join(self.output_dir, "items")
        os.makedirs(items_output_dir, exist_ok=True)
        
        # 4. 分批保存
        total_items = len(titles_df)
        num_partitions = (total_items + items_per_partition - 1) // items_per_partition
        
        print(f"总共 {total_items} 个物品，将分成 {num_partitions} 个分区文件")
        
        for partition_idx in range(num_partitions):
            start_idx = partition_idx * items_per_partition
            end_idx = min((partition_idx + 1) * items_per_partition, total_items)
            
            output_file = os.path.join(items_output_dir, f"data_{partition_idx}.tfrecord.gz")
            
            print(f"  创建分区 {partition_idx}: 物品 {start_idx}-{end_idx-1}")
            
            with tf.python_io.TFRecordWriter(output_file, options=tf.python_io.TFRecordOptions(compression_type="GZIP")) as writer:
                for i in range(start_idx, end_idx):
                    item_id = int(titles_df.iloc[i]['id'])
                    title = titles_df.iloc[i]['title']
                    embedding = embeddings_array[i]
                    
                    features = {}
                    features["id"] = self.create_int64_list_feature([item_id])
                    features["embedding"] = self.create_float_list_feature(embedding)
                    features["text"] = self.create_bytes_list_feature([title])
                    
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())
            
            print(f"    分区 {partition_idx} 保存完成，包含 {end_idx - start_idx} 个物品")
    
    def process(self, max_seq_len=20):
        """
        完整的数据处理流程 (模拟您的 process 方法)
        """
        print("开始数据处理流程...")
        
        # 1. 构造用户序列
        sequences = self.train_test_split(max_seq_len=max_seq_len)
        
        # 2. 保存用户序列数据到 TFRecord
        self.save_sequence_data_to_tfrecord(sequences)
        
        # 3. 保存物品数据到 TFRecord
        self.save_item_data_to_tfrecord()
        
        print("\n数据处理完成！")
        print("生成的文件结构:")
        print(f"{self.output_dir}/")
        print("├── sequences/")
        print("│   ├── train/partition_0.tfrecord.gz")
        print("│   ├── eval/partition_0.tfrecord.gz")
        print("│   └── test/partition_0.tfrecord.gz")
        print("└── items/")
        print("    ├── data_0.tfrecord.gz")
        print("    ├── data_1.tfrecord.gz")
        print("    └── ...")
    
    def verify_output(self):
        """
        验证输出文件
        """
        print("\n=== 验证输出文件 ===")
        
        # 验证序列数据
        sequence_file = os.path.join(self.output_dir, "sequences", "train", "partition_0.tfrecord.gz")
        if os.path.exists(sequence_file):
            print(f"\n验证序列文件: {sequence_file}")
            self._verify_sequence_file(sequence_file, max_examples=2)
        
        # 验证物品数据
        item_file = os.path.join(self.output_dir, "items", "data_0.tfrecord.gz")
        if os.path.exists(item_file):
            print(f"\n验证物品文件: {item_file}")
            self._verify_item_file(item_file, max_examples=2)
    
    def _verify_sequence_file(self, tfrecord_file, max_examples=2):
        """验证序列文件"""
        feature_description = {
            'sequence_data': tf.VarLenFeature(tf.int64),
            'user_id': tf.VarLenFeature(tf.int64)
        }
        
        dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP")
        
        with tf.Session() as sess:
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            
            for i in range(max_examples):
                try:
                    raw_record = sess.run(next_element)
                    parsed_example = tf.parse_single_example(raw_record, feature_description)
                    parsed_result = sess.run(parsed_example)
                    
                    print(f"  --- Example {i} ---")
                    print(f"  sequence_data-sparsetensor: {parsed_result['sequence_data'].values}")
                    print(f"  user_id-sparsetensor: {parsed_result['user_id'].values}")
                    
                except tf.errors.OutOfRangeError:
                    break
    
    def _verify_item_file(self, tfrecord_file, max_examples=2):
        """验证物品文件"""
        feature_description = {
            'id': tf.VarLenFeature(tf.int64),
            'embedding': tf.VarLenFeature(tf.float32),
            'text': tf.VarLenFeature(tf.string)
        }
        
        dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP")
        
        with tf.Session() as sess:
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            
            for i in range(max_examples):
                try:
                    raw_record = sess.run(next_element)
                    parsed_example = tf.parse_single_example(raw_record, feature_description)
                    parsed_result = sess.run(parsed_example)
                    
                    print(f"  --- Example {i} ---")
                    print(f"  id: {parsed_result['id'].values}")
                    print(f"  embedding: shape({len(parsed_result['embedding'].values)})")
                    
                    text_value = parsed_result['text'].values[0].decode('utf-8') if len(parsed_result['text'].values) > 0 else ""
                    print(f"  text: [b\"{text_value}\"]")
                    
                except tf.errors.OutOfRangeError:
                    break


if __name__ == "__main__":
    # 配置路径
    raw_dir = "/home/yfu/public_data/MRS/MicroLens/MicroLens-100k"
    output_dir = "/home/yfu/code/GR/GRID/data/microlens_processed"
    
    # 创建处理器并执行完整流程
    processor = MicroLensDataProcessor(raw_dir, output_dir)
    
    # 执行 process 方法
    processor.process(max_seq_len=20)
    
    # 验证输出
    processor.verify_output()