import tensorflow as tf

# --------- 配置 ---------
# tfrecord_file = "/home/yfu/code/GR/GRID/data/microlens/items/data_0.tfrecord.gz"
tfrecord_file = "/home/yfu/code/GR/GRID/data/amazon_data/sports/training/partition_0.tfrecord.gz"
max_examples_to_print = 2

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

# --------- 读取 TFRecord ---------
raw_dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP")
dataset_iterator = iter(raw_dataset)

# 推断 feature_description
sample_record = next(dataset_iterator)
feature_description = infer_feature_description(sample_record)

print("Feature description:")
for key, desc in feature_description.items():
    print(f"  {key}: {desc}")

# --------- 打印前几条样本并检查类型 ---------
dataset_iterator = iter(raw_dataset)
for i in range(max_examples_to_print):
    try:
        raw_example = next(dataset_iterator)
    except StopIteration:
        break
    parsed = parse_tfrecord_example(raw_example, feature_description)
    
    print(f"\n--- Example {i} ---")
    for key, value in parsed.items():
        print(f"{key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, tf.SparseTensor):
            print(f"  SparseTensor values shape: {value.values.shape}")
            print(f"  SparseTensor dense shape: {value.dense_shape}")
            print(f"  Is SparseTensor: True")
            if key == 'embedding':
                print(f"  Values (first 5): {value.values.numpy()[:5]}")
        else:
            print(f"  Regular tensor shape: {value.shape}")
            print(f"  Is SparseTensor: False")