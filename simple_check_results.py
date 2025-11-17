import os
import pickle
import torch

# 指定要查看的推理结果目录
default_dir = "/home/zxchen/GRID/logs/inference/runs/2025-10-29/07-57-28/pickle"

# 读取并显示pkl文件内容
def read_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n=== {os.path.basename(file_path)} 文件信息 ===")
        print(f"数据类型: {type(data).__name__}")
        print(f"数据长度: {len(data)}")
        
        if isinstance(data, list) and len(data) > 0:
            print(f"\n第一个元素类型: {type(data[0]).__name__}")
            print(f"第一个元素内容: {data[0]}")
            
            if isinstance(data[0], dict):
                print(f"\n字典键: {list(data[0].keys())}")
                # 只显示部分键值对，避免输出过多
                for key, value in list(data[0].items())[:3]:  # 只显示前3个键值对
                    print(f"  {key}: {type(value).__name__}")
                    if hasattr(value, 'shape'):
                        print(f"    形状: {value.shape}")
        
        return data
    except Exception as e:
        print(f"读取pkl文件时出错: {e}")
        return None

# 读取并显示pt文件内容
def read_pt_file(file_path):
    try:
        tensor = torch.load(file_path)
        
        print(f"\n=== {os.path.basename(file_path)} 文件信息 ===")
        print(f"张量类型: {type(tensor).__name__}")
        print(f"张量形状: {tensor.shape if hasattr(tensor, 'shape') else 'N/A'}")
        print(f"张量数据类型: {tensor.dtype if hasattr(tensor, 'dtype') else 'N/A'}")
        
        return tensor
    except Exception as e:
        print(f"读取pt文件时出错: {e}")
        return None

# 查看run4.sh的主要功能
def explain_run4_functionality():
    print("\n=== run4.sh脚本功能说明 ===")
    print("run4.sh执行的是tiger_inference_flat实验，这是一个推理任务，主要功能包括：")
    print("1. 使用预训练的模型进行推理")
    print("2. 使用指定的semantic_id_path作为输入")
    print("3. 设置num_hierarchies=4，表示使用4层层次结构")
    print("4. 加载指定的模型检查点进行推理")
    print("\n推理结果存储在：")
    print(f"- {default_dir}/merged_predictions.pkl (原始预测结果列表)")
    print(f"- {default_dir}/merged_predictions_tensor.pt (转换后的PyTorch张量)")
    print("\n这些结果不是常规的评估指标（如NDCG、Recall等），而是模型生成的语义表示或预测结果。")

# 主函数
def main():
    print("=== run4.sh推理结果分析 ===")
    
    # 显示run4.sh的功能说明
    explain_run4_functionality()
    
    # 读取pkl文件
    pkl_path = os.path.join(default_dir, "merged_predictions.pkl")
    if os.path.exists(pkl_path):
        read_pkl_file(pkl_path)
    else:
        print(f"\n未找到 {pkl_path}")
    
    # 读取pt文件
    pt_path = os.path.join(default_dir, "merged_predictions_tensor.pt")
    if os.path.exists(pt_path):
        read_pt_file(pt_path)
    else:
        print(f"\n未找到 {pt_path}")

if __name__ == "__main__":
    main()