import os
import pickle
import torch
from datetime import datetime

# 查找最新的推理结果目录
def find_latest_inference_dir():
    base_dir = "/home/zxchen/GRID/logs/inference/runs"
    dates = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    dates.sort(reverse=True)
    
    if not dates:
        return None
    
    latest_date_dir = os.path.join(base_dir, dates[0])
    times = [t for t in os.listdir(latest_date_dir) if os.path.isdir(os.path.join(latest_date_dir, t))]
    times.sort(reverse=True)
    
    if not times:
        return None
    
    return os.path.join(latest_date_dir, times[0], "pickle")

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
        
        # 显示部分数据
        if hasattr(tensor, 'shape') and len(tensor.shape) > 0:
            print(f"\n张量前10个元素: {tensor[:10] if tensor.numel() > 10 else tensor}")
        
        return tensor
    except Exception as e:
        print(f"读取pt文件时出错: {e}")
        return None

# 查找run4.sh相关的配置文件
def find_run4_configs():
    print("\n=== 查找run4.sh脚本信息 ===")
    domains = ["beauty", "microlens", "sports", "toys"]
    
    for domain in domains:
        script_path = f"/home/zxchen/GRID/sh/{domain}/run4.sh"
        if os.path.exists(script_path):
            with open(script_path, 'r') as f:
                content = f.read()
            print(f"\n{domain} 领域的run4.sh:")
            print(content)
            
            # 提取关键参数
            import re
            ckpt_match = re.search(r'ckpt_path=(.*)', content)
            sem_match = re.search(r'semantic_id_path=(.*)', content)
            
            if ckpt_match:
                print(f"使用的检查点: {ckpt_match.group(1)}")
            if sem_match:
                print(f"使用的语义ID路径: {sem_match.group(1)}")

# 主函数
def main():
    print("=== 检查run4.sh推理结果 ===")
    
    # 首先显示run4.sh的配置信息
    find_run4_configs()
    
    # 查找最新的推理结果
    latest_dir = find_latest_inference_dir()
    if latest_dir:
        print(f"\n=== 最新的推理结果目录 ===")
        print(latest_dir)
        
        # 检查是否存在merged_predictions.pkl
        pkl_path = os.path.join(latest_dir, "merged_predictions.pkl")
        if os.path.exists(pkl_path):
            read_pkl_file(pkl_path)
        else:
            print(f"\n未找到 {pkl_path}")
        
        # 检查是否存在merged_predictions_tensor.pt
        pt_path = os.path.join(latest_dir, "merged_predictions_tensor.pt")
        if os.path.exists(pt_path):
            read_pt_file(pt_path)
        else:
            print(f"\n未找到 {pt_path}")
    else:
        print("未找到推理结果目录")

if __name__ == "__main__":
    main()