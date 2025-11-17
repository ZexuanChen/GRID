import sys
sys.path.append('/home/yfu/code/GR/GRID')

from src.data.loading.components.pre_processing import convert_bytes_to_string
import numpy as np

# 测试修复后的 convert_bytes_to_string 函数
def test_convert_bytes_to_string():
    print("测试 convert_bytes_to_string 函数...")
    
    # 模拟从 TFRecord 读取的数据（bytes 格式）
    test_data = {
        'text': np.array([b' Gu long song gaga bad'], dtype=object),
        'id': np.array([1], dtype=np.int64)
    }
    
    print(f"输入数据: {test_data}")
    
    # 模拟 dataset_config
    class MockConfig:
        pass
    
    config = MockConfig()
    
    try:
        # 调用函数
        result = convert_bytes_to_string(
            test_data, 
            config, 
            features_to_apply=['text']
        )
        
        print(f"输出数据: {result}")
        print(f"文本类型: {type(result['text'][0])}")
        print(f"文本内容: {result['text'][0]}")
        print("✓ convert_bytes_to_string 测试成功")
        
    except Exception as e:
        print(f"✗ convert_bytes_to_string 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_convert_bytes_to_string()