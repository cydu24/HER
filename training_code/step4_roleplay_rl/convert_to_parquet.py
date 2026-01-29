#!/usr/bin/env python3
"""
将JSONL文件转换为Parquet格式 (修复版本)
保持复杂字段为原始字典格式，而不是JSON字符串
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any
import os

def jsonl_to_parquet_fixed(input_jsonl: str, output_parquet: str, batch_size: int = 1000):
    """
    将JSONL文件转换为Parquet格式 (修复版本)
    保持复杂字段为字典格式以兼容训练代码
    
    Args:
        input_jsonl: 输入的JSONL文件路径
        output_parquet: 输出的Parquet文件路径
        batch_size: 批处理大小
    """
    
    print(f"开始转换: {input_jsonl} -> {output_parquet}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")
    
    # 读取数据并转换
    all_data = []
    total_count = 0
    
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                line = line.strip()
                if not line:
                    continue
                
                # 解析JSON - 保持原始数据结构
                data = json.loads(line)
                all_data.append(data)
                total_count += 1
                
                if total_count % 1000 == 0:
                    print(f"已处理 {total_count} 条数据")
                    
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"第{line_num}行处理错误: {e}")
                continue
    
    # 创建DataFrame并保存
    print("正在创建DataFrame...")
    df = pd.DataFrame(all_data)
    
    print("正在保存Parquet文件...")
    df.to_parquet(output_parquet, index=False, engine='pyarrow')
    
    print(f"转换完成！总共处理了 {total_count} 条数据")
    
    # 验证输出文件
    if os.path.exists(output_parquet):
        file_size = os.path.getsize(output_parquet) / (1024 * 1024)  # MB
        print(f"输出文件大小: {file_size:.2f} MB")
        
        # 读取验证
        try:
            df_verify = pd.read_parquet(output_parquet)
            print(f"验证成功：Parquet文件包含 {len(df_verify)} 行数据")
            print(f"列名: {list(df_verify.columns)}")
            
            # 验证数据类型
            print("\n数据类型验证:")
            sample_row = df_verify.iloc[0]
            for col in df_verify.columns:
                value = sample_row[col]
                print(f"  {col}: {type(value).__name__}")
                
        except Exception as e:
            print(f"验证失败: {e}")

def create_sample_parquet(input_parquet: str, output_parquet: str, n_samples: int = 100):
    """创建小样本Parquet文件"""
    print(f"创建 {n_samples} 条样本数据...")
    
    df = pd.read_parquet(input_parquet)
    df_sample = df.head(n_samples)
    df_sample.to_parquet(output_parquet, index=False, engine='pyarrow')
    
    print(f"样本文件已保存: {output_parquet}")
    print(f"样本数据量: {len(df_sample)} 条")
    
    file_size = os.path.getsize(output_parquet) / (1024 * 1024)
    print(f"文件大小: {file_size:.2f} MB")

def main():
    # 使用新生成的 roleplay RL 训练数据
    input_file = "/path/to/data/example"
    output_file = "/path/to/data/example"
    sample_file = "/path/to/data/example"
    
    try:
        # 转换完整数据集
        jsonl_to_parquet_fixed(input_file, output_file)
        print("✅ 完整数据集转换成功！")
        
        # 创建100条样本
        create_sample_parquet(output_file, sample_file, 100)
        print("✅ 样本数据集创建成功！")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")

if __name__ == "__main__":
    main()
