try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import numpy as np

def test_tfrecord_text_encoding():
    """测试 TFRecord 文件中文本的编码格式"""
    
    tfrecord_file = "data/microlens/items/data_0.tfrecord.gz"
    
    # 定义特征描述
    feature_description = {
        'id': tf.VarLenFeature(tf.int64),
        'embedding': tf.VarLenFeature(tf.float32),
        'text': tf.VarLenFeature(tf.string)
    }
    
    # 读取数据
    dataset = tf.data.TFRecordDataset([tfrecord_file], compression_type="GZIP")
    
    print("检查 TFRecord 文件中的文本编码...")
    
    # 使用 Session 进行 TensorFlow 1.x 兼容
    with tf.Session() as sess:
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        
        for i in range(5):
            try:
                raw_record = sess.run(next_element)
                parsed_example = tf.parse_single_example(raw_record, feature_description)
                parsed_result = sess.run(parsed_example)
                
                # 检查文本字段
                text_sparse = parsed_result['text']
                id_sparse = parsed_result['id']
                
                print(f"\n--- Example {i} ---")
                print(f"ID: {id_sparse.values}")
                
                if len(text_sparse.values) > 0:
                    text_bytes = text_sparse.values[0]
                    print(f"Text bytes type: {type(text_bytes)}")
                    print(f"Text bytes: {text_bytes}")
                    
                    try:
                        # 尝试解码为 UTF-8
                        decoded_text = text_bytes.decode('utf-8')
                        print(f"Decoded UTF-8 text: {decoded_text}")
                        print("✓ UTF-8 解码成功")
                    except UnicodeDecodeError as e:
                        print(f"✗ UTF-8 解码失败: {e}")
                        try:
                            # 尝试其他编码
                            decoded_text = text_bytes.decode('latin-1', errors='ignore')
                            print(f"Fallback decoded text: {decoded_text}")
                        except Exception as e2:
                            print(f"✗ 所有解码尝试都失败: {e2}")
                else:
                    print("✗ 文本字段为空")
                    
            except tf.errors.OutOfRangeError:
                print(f"数据集结束，共处理 {i} 条记录")
                break
            except Exception as e:
                print(f"处理第 {i} 条记录时出错: {e}")
                break

if __name__ == "__main__":
    test_tfrecord_text_encoding()