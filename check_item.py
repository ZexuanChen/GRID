import tensorflow as tf
# 这个脚本用于检查tfrecord文件中每个item的特征

# --------- 配置 ---------
tfrecord_file = "data/microlens/items/data_0.tfrecord.gz"  # 替换成你的文件路径
max_examples_to_print = 4  # 打印前几条样本

# --------- 辅助函数 ---------
def parse_tfrecord_example(example_proto, feature_description):
    """
    解析单条 TFRecord
    """
    return tf.io.parse_single_example(example_proto, feature_description)

def infer_feature_description(sample_record):
    """
    根据一条样本推断 feature 类型
    """
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
        print(key,feature_description[key])
    exit()
    return feature_description

# --------- 读取 TFRecord ---------
raw_dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP")
dataset_iterator = iter(raw_dataset)

# 推断 feature_description
sample_record = next(dataset_iterator)
feature_description = infer_feature_description(sample_record)

# --------- 打印前几条样本 ---------
dataset_iterator = iter(raw_dataset)  # 重新创建 iterator
for i in range(max_examples_to_print):
    try:
        raw_example = next(dataset_iterator)
    except StopIteration:
        break
    parsed = parse_tfrecord_example(raw_example, feature_description)
    
    print(f"--- Example {i} ---")
    for key, value in parsed.items():
        # 对 VarLenFeature 需要转换成稠密 tensor 或直接打印 sparse
        print(key,type(value))
        if isinstance(value, tf.SparseTensor): # 为什么存的时候必须存成sparse？？？
            if(key=='embedding'): print(f"{key}: {value.values.shape}")
            else: print(f"{key}: {value.values.numpy()}") 
        else:
            print(f"{key}: {value.numpy()}") 

# 
# --- Example 1 ---
# embedding
# id: [1] # 1
# text: [b"Title: OPI Red Shatter Crackle Nail Polish E55 New; Brand: OPI; Categories: ['Beauty', 'Makeup', 'Nails', 'Nail Polish']; Price: 3.04; "]
# --- Example 2 ---
# embedding # 
# id: [2]
# text: [b"Title: SKIN79 The Prestige Beblesh Balm BB Cream Diamond Collection; Brand: Unknown; Categories: ['Beauty', 'Skin Care', 'Face', 'Creams & Moisturizers']; Price: 14.96; "]


# training
# --- Example 4 --- 
# embedding-sparsetensor
# sequence_data-sparsetensor: [ 3 24 25 26 27 28 29] # 1, 21
# text-sparsetensor: [b"Title: WAWO 15 Color Professionl Makeup Eyeshadow Camouflage Facial Concealer Neutral Palette; Brand: COKA; Categories: ['Beauty', 'Makeup', 'Face', 'Concealers & Neutralizers']; Price: 5.04; "
#  b"Title: World Pride Fashionable 23&quot; Straight Full Head Clip in Hair Extensions - Light Brown; Brand: Unknown; Categories: ['Beauty', 'Hair Care', 'Styling Products', 'Hair Extensions & Wigs', 'Hair Extensions']; Price: nan; "
#  b"Title: MapofBeauty Charming Synthetic Fiber Long Wavy Hair Wig Women's Party Full Wigs (Wine Red); Brand: MapofBeauty; Categories: ['Beauty', 'Hair Care', 'Styling Products', 'Hair Extensions & Wigs', 'Wigs']; Price: 8.79; "
#  b"Title: World Pride Pefect Eyebrow Shaping Stencils; Brand: niceEshop; Categories: ['Beauty', 'Makeup', 'Eyes', 'Eyebrow Color']; Price: 1.5; "
#  b"Title: New Leopard Shell Waterproof Liquid Eye Liner Eyeliner Pen Makeup Cosmetic Black; Brand: Evermarket; Categories: ['Beauty', 'Makeup', 'Eyes', 'Eyeliner']; Price: 2.02; "
#  b"Title: Ardell Lashgrip Adhesive Dark .236 oz. Tube (Black Package); Brand: Ardell; Categories: ['Beauty', 'Tools & Accessories', 'Makeup Brushes & Tools', 'Eyelash Tools', 'Fake Eyelashes & Adhesives']; Price: 4.7; "
#  b"Title: 10 Pair Long Black False Eyelashes Eye Lashes Makeup; Brand: Fashionwu; Categories: ['Beauty', 'Tools & Accessories', 'Makeup Brushes & Tools', 'Eyelash Tools', 'Fake Eyelashes & Adhesives']; Price: 1.89; "]
# user_id-sparsetensor: [4]

# train/eval/test
# sequence_data
# user_id

#text VarLenFeature(dtype=tf.string)
# embedding VarLenFeature(dtype=tf.float32)
# id VarLenFeature(dtype=tf.int64)