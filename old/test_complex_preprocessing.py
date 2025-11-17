import sys
sys.path.append('/home/yfu/code/GR/GRID')

from src.data.loading.components.pre_processing import convert_bytes_to_string
import numpy as np

def test_complex_bytes():
    print("测试包含特殊字符的 bytes 转换...")
    
    # 测试包含中文和特殊字符的文本
    test_texts = [
        "Hello World!",
        "测试中文文本",
        "Café français", 
        "🎉 Emoji test",
        "Mixed 中英文 text"
    ]
    
    # 转换为 bytes 并创建测试数据
    bytes_texts = [text.encode('utf-8') for text in test_texts]
    
    test_data = {
        'text': np.array(bytes_texts, dtype=object),
        'id': np.array([1, 2, 3, 4, 5], dtype=np.int64)
    }
    
    print(f"输入 bytes 数据: {[b.decode('utf-8') for b in bytes_texts]}")
    
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
        
        print(f"输出字符串数据: {list(result['text'])}")
        
        # 验证转换是否正确
        for i, (original, converted) in enumerate(zip(test_texts, result['text'])):
            if original == converted:
                print(f"✓ 第{i+1}个文本转换正确: {converted}")
            else:
                print(f"✗ 第{i+1}个文本转换错误: 期望 '{original}', 得到 '{converted}'")
        
        print("✓ 复杂字符 convert_bytes_to_string 测试完成")
        
    except Exception as e:
        print(f"✗ convert_bytes_to_string 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complex_bytes()