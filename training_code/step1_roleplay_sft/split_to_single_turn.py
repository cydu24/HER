#!/usr/bin/env python3
"""
Step 1: 将多轮对话拆分为单轮样本

逻辑：
- 原本一个多轮对话: system + [user1, asst1, user2, asst2, ..., userN, asstN]
- 拆成 N 个单轮样本:
  - 样本1: system + user1 + asst1 (加 system_thinking)
  - 样本2: system + user1 + asst1 + user2 + asst2 (加 system_thinking)
  - ...
  - 样本N: system + 完整历史 + userN + asstN (加 system_thinking)

注意：
- 每个单轮样本的最后一个 assistant 回复需要加上 system_thinking
- 历史对话中的 assistant 回复不需要 system_thinking

输入: sft_train.jsonl, sft_val.jsonl (多轮)
输出: sft_train_single.jsonl, sft_val_single.jsonl (单轮)
"""

import json
import argparse
import os
from tqdm import tqdm
from collections import Counter


def split_to_single_turns(messages):
    """
    将多轮对话拆分为多个单轮样本
    
    Args:
        messages: [system, user1, asst1, user2, asst2, ...]
    
    Returns:
        list of single-turn samples
    """
    if not messages or messages[0]['role'] != 'system':
        return []
    
    system_msg = messages[0]
    samples = []
    
    # 遍历所有 assistant 位置
    for i, msg in enumerate(messages):
        if msg['role'] == 'assistant':
            # 构建单轮样本：system + 历史对话 + 当前 user/assistant
            single_messages = [system_msg] + messages[1:i+1]
            
            # 确保最后一个是 assistant
            if single_messages[-1]['role'] == 'assistant':
                # 检查是否有 system_thinking（如果原本没有，保持原样）
                # 注意：我们假设原数据只有最后一轮有 system_thinking
                # 拆分后，每个样本的最后一轮都应该有 system_thinking
                # 但如果原数据没有，我们就不加
                samples.append(single_messages)
    
    return samples


def process_file(input_file, output_file):
    """处理单个文件"""
    
    total_input = 0
    total_output = 0
    turns_dist = Counter()
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc=f"处理 {os.path.basename(input_file)}"):
            data = json.loads(line)
            total_input += 1
            
            trace_id = data.get('trace_id', '')
            messages = data.get('messages', [])
            
            # 拆分为单轮
            single_samples = split_to_single_turns(messages)
            
            for idx, single_messages in enumerate(single_samples):
                # 生成新的 trace_id
                new_trace_id = f"{trace_id}_turn_{idx}"
                
                # 统计轮数（历史轮数）
                history_turns = sum(1 for m in single_messages if m['role'] == 'assistant') - 1
                turns_dist[history_turns] += 1
                
                single_sample = {
                    "trace_id": new_trace_id,
                    "messages": single_messages,
                    "original_trace_id": trace_id,
                    "turn_index": idx,
                    "history_turns": history_turns
                }
                
                f_out.write(json.dumps(single_sample, ensure_ascii=False) + '\n')
                total_output += 1
    
    return total_input, total_output, turns_dist


def main():
    parser = argparse.ArgumentParser(description='将多轮对话拆分为单轮样本')
    parser.add_argument('--input_dir', type=str,
                        default='/path/to/project/code/step1_roleplay_sft/split_data',
                        help='输入目录')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/project/code/step1_roleplay_sft/single_turn_data',
                        help='输出目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Step 1: 拆分多轮对话为单轮样本")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 处理 train 和 val
    results = {}
    
    for split in ['train', 'val']:
        input_file = os.path.join(args.input_dir, f'sft_{split}.jsonl')
        output_file = os.path.join(args.output_dir, f'sft_{split}_single.jsonl')
        
        if not os.path.exists(input_file):
            print(f"跳过 {split}：文件不存在")
            continue
        
        total_input, total_output, turns_dist = process_file(input_file, output_file)
        results[split] = {
            'input': total_input,
            'output': total_output,
            'expansion': total_output / total_input if total_input > 0 else 0,
            'turns_dist': turns_dist
        }
        
        print(f"\n{split}:")
        print(f"  输入: {total_input} 个多轮样本")
        print(f"  输出: {total_output} 个单轮样本")
        print(f"  扩展倍数: {total_output / total_input:.2f}x")
        print(f"  输出文件: {output_file}")
    
    # 保存统计信息
    stats_file = os.path.join(args.output_dir, 'split_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        # 转换 Counter 为 dict
        for split in results:
            results[split]['turns_dist'] = dict(results[split]['turns_dist'])
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计信息已保存: {stats_file}")
    
    # 打印历史轮数分布
    print("\n历史轮数分布 (train):")
    if 'train' in results:
        for turns, count in sorted(results['train']['turns_dist'].items(), key=lambda x: int(x[0])):
            print(f"  {turns} 轮历史: {count} 个样本")


if __name__ == "__main__":
    main()

