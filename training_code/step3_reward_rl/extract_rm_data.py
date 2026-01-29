#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据抽取脚本V2：从messages格式数据中抽取标准答案并转换为RL数据格式
用于RL和Test数据
"""

import json
import re
import sys
from typing import Dict, Any
from pathlib import Path

def extract_better_response_regex(content: str) -> str:
    """使用正则表达式直接提取better_response"""
    try:
        # 使用正则表达式匹配 "better_response": "cand_1", "cand_2" 或 "tie"
        pattern = r'"better_response":\s*"(cand_[12]|tie)"'
        match = re.search(pattern, content)
        if match:       
            return match.group(1)
        else:
            return None
            
    except Exception as e:
        print(f"正则提取出错: {e}")
        return None

def convert_to_rl_format(item: Dict[str, Any]) -> Dict[str, Any]:
    """将messages格式数据项转换为RL数据格式"""
    
    # 从messages中提取用户提示内容
    messages = item.get("messages", [])
    if not messages:
        return None
    
    # 找到用户的prompt内容和assistant回复
    user_prompt = ""
    assistant_response = ""
    
    for msg in messages:
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
        elif msg.get("role") == "assistant":
            assistant_response = msg.get("content", "")
    
    if not user_prompt or not assistant_response:
        return None
    
    # 从assistant的回复中提取标准答案
    better_response = extract_better_response_regex(assistant_response)
    
    if not better_response:
        return None
    
    # 系统指导信息
    sys_info = '''
You always first think about the reasoning process in the mind and then provides the user with the answer.
The reasoning process are enclosed within '<think>' '</think>' e.g.,
<think>
A detailed reasoning process here
</think>
Reply the json format answer to user here.
Please reason step by step, and put your final answer within only json format.
Now, start your reasoning process.
    '''
    
    # 构造RL数据格式
    rl_item = {
        "data_source": "v3_tx_sft/tx_rl4rm",
        "prompt": [
            {
                "content": user_prompt + '\n' + sys_info,
                "role": "user"
            }
        ],
        "ability": "roleplay",
        "reward_model": {
            "answer": better_response,
            "problem": "",
            "solution": better_response,
            "style": "rule"
        },
        "extra_info": item  # 把整个item放到extra_info中
    }
    
    return rl_item

def process_data(input_file: str, output_file: str, data_type: str):
    """处理数据文件"""
    print(f"=" * 70)
    print(f"开始处理{data_type.upper()}数据: {input_file}")
    print(f"=" * 70)
    
    processed_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            if line_num % 10000 == 0:
                print(f"处理进度: {line_num} 行")
            
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析JSON
                item = json.loads(line)
                
                # 转换为RL格式
                rl_item = convert_to_rl_format(item)
                
                if rl_item:
                    # 写入输出文件
                    f_out.write(json.dumps(rl_item, ensure_ascii=False) + '\n')
                    processed_count += 1
                else:
                    skipped_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"第{line_num}行处理错误: {e}")
                skipped_count += 1
    
    print(f"\n处理完成!")
    print(f"✅ 成功处理: {processed_count} 条")
    print(f"❌ 跳过: {skipped_count} 条")
    print(f"📁 输出文件: {output_file}")
    print(f"=" * 70)
    
    return processed_count, skipped_count

def main():
    base_dir = "/path/to/data/example"
    output_dir = "/path/to/data/example"
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 处理RL和Test数据
    datasets = [
        {
            'type': 'rl',
            'input': f"{base_dir}/rl_training_data.jsonl",
            'output': f"{output_dir}/roleplay_rl_data.jsonl"
        },
        {
            'type': 'test',
            'input': f"{base_dir}/test_data.jsonl",
            'output': f"{output_dir}/roleplay_test_data.jsonl"
        }
    ]
    
    all_stats = {}
    
    for dataset in datasets:
        data_type = dataset['type']
        input_file = dataset['input']
        output_file = dataset['output']
        
        if not Path(input_file).exists():
            print(f"⚠️ 输入文件不存在: {input_file}")
            continue
        
        # 处理数据
        success, failed = process_data(input_file, output_file, data_type)
        all_stats[data_type] = {'success': success, 'failed': failed}
    
    # 总体统计
    print(f"\n{'=' * 70}")
    print("📊 总体统计")
    print(f"{'=' * 70}")
    for data_type, stats in all_stats.items():
        print(f"{data_type.upper()}: {stats['success']}/{stats['success']+stats['failed']} 条成功")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()

