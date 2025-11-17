try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import numpy as np
import pandas as pd
import os

import torch

def create_int64_list_feature(values):
    """
    创建 int64 列表特征
    """
    if not isinstance(values, (list, np.ndarray)):
        values = [values]
    
    values = [int(v) for v in values]
    
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def create_float_list_feature(values):
    """
    创建 float 列表特征
    """
    if not isinstance(values, (list, np.ndarray)):
        values = [values]
    
    values = [float(v) for v in values]
    
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def create_bytes_list_feature(values):
    """
    创建 bytes 列表特征
    """
    if isinstance(values, str):
        values = [values]
    
    # 确保是 bytes 格式
    byte_values = []
    for v in values:
        if isinstance(v, str):
            byte_values.append(v.encode('utf-8'))
        else:
            byte_values.append(v)
    
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=byte_values))


def create_item_tfrecord_example(item_id, embedding, text):
    """
    创建单个物品的TFRecord样本
    包含：id, embedding, text
    """
    features = {}
    
    # 处理 id
    features["id"] = create_int64_list_feature([item_id])
    
    # 处理 embedding (浮点数数组)
    features["embedding"] = create_float_list_feature(embedding)
    
    # 处理 text (字符串)
    features["text"] = create_bytes_list_feature([text])
    
    # 创建Example
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def convert_item_data_to_tfrecord(raw_dir, output_dir, items_per_partition=1000):
    """
    将物品数据转换并保存为TFRecord.gz文件
    """
    # 1. 读取 title 数据
    titles_path = os.path.join(raw_dir, "MicroLens-100k_title_en.csv")
    print(f"读取标题文件: {titles_path}")
    titles_df = pd.read_csv(titles_path, header=None, names=["id", "title"]) # id是从1开始的嘛？
    titles_df = titles_df.sort_values("id")

    print(titles_df.head(3))
    
    # 2. 读取 embedding 数据
    # emb_path = os.path.join(raw_dir, "text_embeddings_BgeM3.npz")
    # print(f"读取嵌入文件: {emb_path}")
    
    # with np.load(emb_path) as data_npz:
    #     ids_array = data_npz['ids']
    #     embeddings_array = data_npz['embeddings']

    with np.load(os.path.join(raw_dir, "text_embeddings_BgeM3.npz")) as data_npz:
        embeddings_array = data_npz['embeddings']
        embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32)
        tensor_file = os.path.join(output_dir, f"text_emb.pt")
        torch.save(embeddings_tensor, tensor_file)
        print(f"Saved text_emb to {tensor_file}")
        print(f"嵌入向量数组形状: {embeddings_array.shape}")


    with np.load(os.path.join(raw_dir, "fused_SwinV2-T_vs_Sentence-BERT_1152dim.npz")) as data_npz:
        embeddings_array = data_npz['embeddings']
        embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32)
        tensor_file = os.path.join(output_dir, f"text_cv_emb.pt")
        torch.save(embeddings_tensor, tensor_file)
        print(f"Saved text_cv_emb to {tensor_file}")
        print(f"嵌入向量数组形状: {embeddings_array.shape}")

    
    # print(f"ID 数组形状: {ids_array.shape}")
    # print(f"嵌入向量数组形状: {embeddings_array.shape}")
    
    # 3. 验证数据对齐
    # assert len(titles_df) == len(ids_array), f"标题数量 {len(titles_df)} 与ID数量 {len(ids_array)} 不匹配"
    
    # 4. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. 分批保存到多个分区文件
    total_items = len(titles_df)
    num_partitions = (total_items + items_per_partition - 1) // items_per_partition
    
    print(f"总共 {total_items} 个物品，将分成 {num_partitions} 个分区文件")
    
    for partition_idx in range(num_partitions):
        start_idx = partition_idx * items_per_partition
        end_idx = min((partition_idx + 1) * items_per_partition, total_items)
        
        output_file = os.path.join(output_dir, f"data_{partition_idx}.tfrecord.gz")
        
        print(f"创建分区 {partition_idx}: 物品 {start_idx}-{end_idx-1} -> {output_file}")
        
        with tf.python_io.TFRecordWriter(output_file, options=tf.python_io.TFRecordOptions(compression_type="GZIP")) as writer:
            for i in range(start_idx, end_idx):
                # 获取当前物品的数据
                item_id = int(titles_df.iloc[i]['id']) - 1
                title = titles_df.iloc[i]['title']
                embedding = embeddings_array[i]  # 注意：这里假设 embedding 数组的索引与 titles_df 的索引对应
                
                # 创建 TFRecord 样本
                example = create_item_tfrecord_example(item_id, embedding, title)
                # example=tf.train.Features(
                #     feature={
                #         'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[title.encode('utf-8')])),
                #         'embedding': tf.train.Feature(float_list=tf.train.FloatList(value=embedding)),
                #         'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[item_id]))
                #     }
                # )
                writer.write(example.SerializeToString())
        
        print(f"分区 {partition_idx} 保存完成，包含 {end_idx - start_idx} 个物品")


def read_and_verify_item_tfrecord(tfrecord_file, max_examples=3):
    """
    读取并验证物品TFRecord文件内容
    """
    print(f"验证文件: {tfrecord_file}")
    
    # 定义特征描述
    feature_description = {
        'id': tf.VarLenFeature(tf.int64),
        'embedding': tf.VarLenFeature(tf.float32),
        'text': tf.VarLenFeature(tf.string)
    }
    
    # 创建数据集
    dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP")
    
    # 在 TensorFlow 1.x 中需要使用 Session
    with tf.Session() as sess:
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        
        for i in range(max_examples):
            try:
                raw_record = sess.run(next_element)
                
                # 解析记录
                parsed_example = tf.parse_single_example(raw_record, feature_description)
                
                # 运行解析操作
                parsed_result = sess.run(parsed_example)
                
                print(f"\n--- Example {i} ---")
                
                # 打印 id
                id_sparse = parsed_result['id']
                print(f"id: {id_sparse.values}")
                
                # 打印 embedding 形状（不打印具体值，因为太长）
                embedding_sparse = parsed_result['embedding']
                print(f"embedding: shape({len(embedding_sparse.values)})")
                
                # 打印 text
                text_sparse = parsed_result['text']
                if len(text_sparse.values) > 0:
                    text_value = text_sparse.values[0].decode('utf-8')
                    print(f"text: [b\"{text_value}\"]")
                else:
                    print("text: []")
                    
            except tf.errors.OutOfRangeError:
                print("数据集结束")
                break
            except Exception as e:
                print(f"处理第 {i} 条记录时出错: {e}")
                break


if __name__ == "__main__":
    # 配置路径
    raw_dir = "/home/yfu/public_data/MRS/MicroLens/MicroLens-100k"
    output_dir = "/home/yfu/code/GR/GRID/data/microlens/items"
    
    # 转换物品数据
    convert_item_data_to_tfrecord(raw_dir, output_dir, items_per_partition=5000)
    
    # 验证生成的第一个文件
    first_file = os.path.join(output_dir, "data_0.tfrecord.gz")
    if os.path.exists(first_file):
        read_and_verify_item_tfrecord(first_file)