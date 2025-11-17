# 这个脚本用于检查tfrecord文件中每个sequence的特征  # 什么叫检查特征？检查每个sequence的长度，id范围等
import tensorflow as tf
import glob
import numpy as np

# --------- 配置 ---------
# tfrecord_dir = "data/amazon_data/beauty/training"  # 文件夹
tfrecord_dir = "data/microlens/evaluation"  # 文件夹
tfrecord_files = glob.glob(tfrecord_dir + "/*.tfrecord*")
print(f"Found {len(tfrecord_files)} files")

# --------- 辅助函数 ---------
def parse_tfrecord_example(example_proto, feature_description):
    return tf.io.parse_single_example(example_proto, feature_description)

def infer_feature_description(sample_record):
    feature_description = {}
    example = tf.train.Example()
    example.ParseFromString(sample_record.numpy())
    for key, feature in example.features.feature.items():
        if feature.HasField("bytes_list"):
            feature_description[key] = tf.io.VarLenFeature(tf.string)
        elif feature.HasField("float_list"):
            feature_description[key] = tf.io.VarLenFeature(tf.float32)
        elif feature.HasField("int64_list"):
            feature_description[key] = tf.io.VarLenFeature(tf.int64)
        else:
            raise ValueError(f"Unknown feature type for key: {key}")
    return feature_description

# --------- 推断 feature_description (用第一条样本) ---------
raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP")
dataset_iterator = iter(raw_dataset)
sample_record = next(dataset_iterator)
feature_description = infer_feature_description(sample_record)

# --------- 遍历所有样本 ---------
min_seq_len = float("inf")
max_seq_len = 0
min_id_global = float("inf")
max_id_global = 0
count = 0
len_list = []

for raw_example in tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP"):
    parsed = parse_tfrecord_example(raw_example, feature_description)
    seq = parsed.get("sequence_data", None)
    if seq is not None:
        if isinstance(seq, tf.SparseTensor):
            ids = seq.values.numpy()
        else:
            ids = seq.numpy()

        length = len(ids)
        len_list.append(length)

        min_seq_len = min(min_seq_len, length)
        max_seq_len = max(max_seq_len, length)
        min_id_global = min(min_id_global, ids.min())
        max_id_global = max(max_id_global, ids.max())

        count += 1

print(f"Scanned {count} examples")
print(f"Min sequence_data length: {min_seq_len}")
print(f"Max sequence_data length: {max_seq_len}")
print(f"Min ID in sequence_data: {min_id_global}")
print(f"Max ID in sequence_data: {max_id_global}")

# microlens
# Scanned 100000 examples
# Max sequence_data length: 216
# Min ID in sequence_data: 0
# Max ID in sequence_data: 19737

# Scanned 22363 examples
# Max sequence_data length: 202
# Min ID in sequence_data: 0 # 最小是从0开始的！不是从1开始
# Max ID in sequence_data: 12100
 # [117, 121, 129, 147, 147, 148, 152, 180, 190, 202]

# microlens
#[:-2],[-21-1(包含最后一个要预测的物品):-1],[-20-1:]
# train: 202 # 最长202是限制过后的嘛？
# eval: 21 # 只有训练和测试用了20的滑动窗口？？？但是训练的时候不用？20+1
# test: 21