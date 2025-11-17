# 这个脚本用于将Microlens-100k的item数据转换为tfrecord文件 # 什么是tfrecord文件？它是tensorflow的一种数据格式，用于存储大规模数据集
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import os
import torch

def create_int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in values]))

def create_float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v) for v in values]))

def create_bytes_list_feature(values):
    byte_values = [(v.encode('utf-8') if isinstance(v, str) else v) for v in (values if isinstance(values, list) else [values])]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=byte_values))

def create_item_tfrecord_example(item_id, embedding, text):
    features = {
        "id": create_int64_list_feature([item_id]),
        "embedding": create_float_list_feature(embedding),
        "text": create_bytes_list_feature([text])
    }
    return tf.train.Example(features=tf.train.Features(feature=features))

def convert_item_data_to_tfrecord(raw_dir, output_dir, items_per_partition=1000):
    titles_df = pd.read_csv(os.path.join(raw_dir, "MicroLens-100k_title_en.csv"), header=None, names=["id","title"]).sort_values("id")
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
    
    with np.load(os.path.join(raw_dir, "fused_CLIPRN50_vs_BgeM3_2048dim.npz")) as data_npz:
        embeddings_array = data_npz['embeddings']
        embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32)
        tensor_file = os.path.join(output_dir, f"text_cv_emb2.pt")
        torch.save(embeddings_tensor, tensor_file)
        print(f"Saved text_cv_emb2 to {tensor_file}")
        print(f"嵌入向量数组形状: {embeddings_array.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    total_items = len(titles_df)
    num_partitions = (total_items + items_per_partition - 1) // items_per_partition

    for partition_idx in range(num_partitions):
        start, end = partition_idx*items_per_partition, min((partition_idx+1)*items_per_partition, total_items)
        output_file = os.path.join(output_dir, f"data_{partition_idx}.tfrecord.gz")
        with tf.python_io.TFRecordWriter(output_file, options=tf.python_io.TFRecordOptions(compression_type="GZIP")) as writer:
            for i in range(start, end):
                # example = create_item_tfrecord_example(int(titles_df.iloc[i]['id']), embeddings_array[i], titles_df.iloc[i]['title'])
                example = create_item_tfrecord_example(int(titles_df.iloc[i]['id']-1), embeddings_array[i], titles_df.iloc[i]['title'])
                writer.write(example.SerializeToString())
        print(f"Saved partition {partition_idx} with {end-start} items")

def read_and_verify_item_tfrecord(tfrecord_file, max_examples=3):
    feature_description = {'id': tf.VarLenFeature(tf.int64),
                           'embedding': tf.VarLenFeature(tf.float32),
                           'text': tf.VarLenFeature(tf.string)}
    dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP")
    with tf.Session() as sess:
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        for _ in range(max_examples):
            raw_record = sess.run(next_element)
            parsed = tf.parse_single_example(raw_record, feature_description)
            result = sess.run(parsed)
            print("id:", result['id'].values)
            print("embedding length:", len(result['embedding'].values))
            print("text:", result['text'].values[0].decode('utf-8') if len(result['text'].values)>0 else "")

# 把物品信息转换成指定的tfrecords格式
if __name__ == "__main__":
    raw_dir = "/home/yfu/public_data/MRS/MicroLens/MicroLens-100k"
    output_dir = "/home/yfu/code/GR/GRID/data/microlens/items"
    convert_item_data_to_tfrecord(raw_dir, output_dir, items_per_partition=5000)
    read_and_verify_item_tfrecord(os.path.join(output_dir, "data_0.tfrecord.gz"))
