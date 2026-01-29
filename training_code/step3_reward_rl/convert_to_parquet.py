#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL转Parquet格式转换脚本 V2
"""
import random
import json
import pandas as pd
from pathlib import Path
import sys

def jsonl_to_parquet(jsonl_file: str, parquet_file: str):
    """
    将JSONL文件转换为Parquet格式
    
    Args:
        jsonl_file: 输入的JSONL文件路径
        parquet_file: 输出的Parquet文件路径
    """
    print(f"=" * 70)
    print(f"开始转换文件: {jsonl_file}")
    print(f"=" * 70)
    
    # 检查输入文件是否存在
    if not Path(jsonl_file).exists():
        print(f"错误: 输入文件不存在 {jsonl_file}")
        return False
    
    try:
        # 读取JSONL文件
        records = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 5000 == 0:
                    print(f"处理进度: {line_num} 行")
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析错误: {e}")
                    continue
        
        print(f"共读取 {len(records)} 条记录")
        
        # shuffle records
        seed = 42
        random.seed(seed)
        random.shuffle(records)
        print(f"数据已随机打乱（seed={seed}）")
        
        # 转换为DataFrame
        print("转换为DataFrame...")
        df = pd.DataFrame(records)
        print(f"DataFrame形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 保存为Parquet
        print(f"保存为Parquet格式: {parquet_file}")
        df.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
        
        # 显示文件大小
        jsonl_size = Path(jsonl_file).stat().st_size / (1024 * 1024)  # MB
        parquet_size = Path(parquet_file).stat().st_size / (1024 * 1024)  # MB
        
        print("=" * 70)
        print("转换完成!")
        print(f"原始文件大小: {jsonl_size:.2f} MB")
        print(f"Parquet文件大小: {parquet_size:.2f} MB")
        print(f"压缩率: {(1 - parquet_size/jsonl_size)*100:.1f}%")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_parquet_file(parquet_file: str):
    """验证Parquet文件的内容"""
    try:
        print(f"\n验证Parquet文件: {parquet_file}")
        print("=" * 70)
        
        # 读取Parquet文件
        df = pd.read_parquet(parquet_file)
        
        print(f"记录数量: {len(df)}")
        print(f"列数量: {len(df.columns)}")
        print(f"列名: {list(df.columns)}")
        
        # 检查reward_model.answer的分布
        if 'reward_model' in df.columns:
            # 提取answer字段
            answers = df['reward_model'].apply(lambda x: x.get('answer', 'unknown') if isinstance(x, dict) else 'unknown')
            answer_counts = answers.value_counts()
            print(f"\n标准答案分布:")
            for answer, count in answer_counts.items():
                percentage = count / len(df) * 100
                print(f"  {answer}: {count} ({percentage:.1f}%)")
        
        # 显示第一条记录的结构
        print(f"\n第一条记录的字段:")
        first_record = df.iloc[0].to_dict()
        for key in first_record.keys():
            print(f"  {key}: {type(first_record[key])}")
        
        print("=" * 70)
        print("✅ Parquet文件验证通过！")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"验证Parquet文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    output_dir = "/path/to/data/example"
    
    # 处理RL和Test数据
    datasets = [
        {
            'name': 'RL',
            'jsonl': f"{output_dir}/v3_roleplay_rl_data.jsonl",
            'parquet': f"{output_dir}/v3_roleplay_rl_data.parquet"
        },
        {
            'name': 'Test',
            'jsonl': f"{output_dir}/v3_roleplay_test_data.jsonl",
            'parquet': f"{output_dir}/v3_roleplay_test_data.parquet"
        }
    ]
    
    for dataset in datasets:
        jsonl_file = dataset['jsonl']
        parquet_file = dataset['parquet']
        
        if not Path(jsonl_file).exists():
            print(f"⚠️ 输入文件不存在: {jsonl_file}")
            continue
        
        print(f"\n处理{dataset['name']}数据...")
        
        # 转换文件
        success = jsonl_to_parquet(jsonl_file, parquet_file)
        
        if success:
            # 验证转换结果
            verify_parquet_file(parquet_file)
        else:
            print(f"❌ {dataset['name']}数据转换失败")

if __name__ == "__main__":
    main()

