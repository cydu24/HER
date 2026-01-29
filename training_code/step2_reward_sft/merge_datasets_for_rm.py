#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RM训练数据合并脚本
将除了sft_roleplay外的其他数据集合并，并添加用途标注字段
"""

import json
import argparse
from pathlib import Path

def load_data(input_file):
    """加载JSONL数据"""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_data(data, output_file):
    """保存数据到JSONL文件"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def merge_datasets_for_rm(split_datasets_dir, output_file):
    """合并除sft_roleplay外的数据集，添加用途标注"""
    split_datasets_path = Path(split_datasets_dir)
    
    # 定义需要合并的数据集及其用途标注
    datasets_to_merge = {
        "test.jsonl": "test",
        "sft_single.jsonl": "sft_single", 
        "rl.jsonl": "rl",
        "roleplay_rl.jsonl": "roleplay_rl"
    }
    
    merged_data = []
    stats = {}
    
    print("开始合并数据集...")
    
    for filename, dataset_type in datasets_to_merge.items():
        file_path = split_datasets_path / filename
        
        if not file_path.exists():
            print(f"警告: 文件不存在 {file_path}")
            continue
            
        print(f"加载 {filename}...")
        data = load_data(file_path)
        
        # 为每条数据添加用途标注字段
        for item in data:
            item['dataset_type'] = dataset_type
            merged_data.append(item)
        
        stats[dataset_type] = len(data)
        print(f"  - {dataset_type}: {len(data)}条")
    
    # 保存合并后的数据
    print(f"\n保存合并数据到: {output_file}")
    save_data(merged_data, Path(output_file))
    
    # 生成统计报告
    stats['总计'] = len(merged_data)
    stats_file = Path(output_file).parent / "merge_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"统计信息已保存到: {stats_file}")
    print("\n=== 数据合并完成 ===")
    for key, value in stats.items():
        print(f"{key}: {value:,}条")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='合并RM训练数据集')
    parser.add_argument('--input', '-i', required=True, 
                       help='拆分数据集目录路径 (包含test.jsonl, sft_single.jsonl等文件)')
    parser.add_argument('--output', '-o', required=True, 
                       help='输出合并文件路径 (JSONL格式)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return
    
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {args.output}")
    print()
    
    merge_datasets_for_rm(input_dir, args.output)

if __name__ == '__main__':
    main()

# 使用示例命令
python_cmd = '''
python /path/to/data/example \
--input /path/to/data/example \
--output /path/to/data/example
'''
