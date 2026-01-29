#!/usr/bin/env python3
"""
Step 2: 按用途分流数据

将单轮数据按以下比例分流:
- sft_roleplay: ~33% (角色扮演 SFT)
- roleplay_rl: ~8% (角色扮演 RL)
- rm_sft: ~34% (RM SFT)
- rm_rl: ~25% (RM RL)

分流策略:
1. 先按历史轮数分层抽样，保证分布均衡
2. 随机打乱后按比例分配

输入: sft_train_single.jsonl (单轮数据)
输出: sft_roleplay.jsonl, sft_single.jsonl, rl.jsonl, roleplay_rl.jsonl
"""

import json
import argparse
import os
import random
from tqdm import tqdm
from collections import defaultdict


def load_data(input_file):
    """加载数据并按历史轮数分组"""
    data_by_turns = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载数据"):
            item = json.loads(line)
            history_turns = item.get('history_turns', 0)
            data_by_turns[history_turns].append(item)
    
    return data_by_turns


def stratified_sample(data_by_turns, total_size, seed=42):
    """分层抽样"""
    random.seed(seed)
    
    # 计算每层的抽样数量（按比例）
    total_count = sum(len(v) for v in data_by_turns.values())
    
    sampled = []
    for turns, items in data_by_turns.items():
        # 按比例计算该层应抽取的数量
        layer_ratio = len(items) / total_count
        layer_size = int(total_size * layer_ratio)
        
        # 抽样
        if layer_size >= len(items):
            sampled.extend(items)
        else:
            sampled.extend(random.sample(items, layer_size))
    
    # 如果不够，补充抽样
    if len(sampled) < total_size:
        all_items = [item for items in data_by_turns.values() for item in items]
        remaining = [item for item in all_items if item not in sampled]
        additional = min(total_size - len(sampled), len(remaining))
        sampled.extend(random.sample(remaining, additional))
    
    # 如果太多，截断
    if len(sampled) > total_size:
        sampled = random.sample(sampled, total_size)
    
    return sampled


def split_data(input_file, output_dir, sizes, seed=42):
    """分流数据"""
    
    random.seed(seed)
    
    # 加载所有数据
    print("加载数据...")
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取"):
            all_data.append(json.loads(line))
    
    print(f"总数据量: {len(all_data)}")
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 计算各部分大小
    total_requested = sum(sizes.values())
    
    if len(all_data) < total_requested:
        print(f"警告: 数据量 ({len(all_data)}) 小于请求量 ({total_requested})")
        print("将按比例缩放...")
        scale = len(all_data) / total_requested
        sizes = {k: int(v * scale) for k, v in sizes.items()}
        print(f"调整后: {sizes}")
    
    # 分配数据
    results = {}
    idx = 0
    
    for name, size in sizes.items():
        end_idx = min(idx + size, len(all_data))
        results[name] = all_data[idx:end_idx]
        idx = end_idx
    
    # 保存各部分
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {}
    for name, data in results.items():
        output_file = os.path.join(output_dir, f'{name}.jsonl')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 统计历史轮数分布
        turns_dist = defaultdict(int)
        for item in data:
            turns_dist[item.get('history_turns', 0)] += 1
        
        stats[name] = {
            'count': len(data),
            'file': output_file,
            'turns_dist': dict(turns_dist)
        }
        
        print(f"\n{name}:")
        print(f"  数量: {len(data)}")
        print(f"  文件: {output_file}")
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'split_by_purpose_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计信息已保存: {stats_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='按用途分流数据')
    parser.add_argument('--input', type=str,
                        default='/path/to/project/code/step1_roleplay_sft/single_turn_data/sft_train_single.jsonl',
                        help='输入文件')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/project/code/step1_roleplay_sft/final_split',
                        help='输出目录')
    # 全部用完单轮数据，按比例分配
    parser.add_argument('--sft_roleplay', type=int, default=107800,
                        help='sft_roleplay 数量 (角色扮演 SFT) ~33%')
    parser.add_argument('--roleplay_rl', type=int, default=26800,
                        help='roleplay_rl 数量 (角色扮演 RL) ~8%')
    parser.add_argument('--rm_sft', type=int, default=108800,
                        help='rm_sft 数量 (RM SFT) ~34%')
    parser.add_argument('--rm_rl', type=int, default=80000,
                        help='rm_rl 数量 (RM RL) ~25%')
    parser.add_argument('--rl_test', type=int, default=200,
                        help='rl_test 数量 (RL 测试集)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step 2: 按用途分流数据")
    print("=" * 60)
    print(f"输入: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print()
    print("目标分配:")
    print(f"  sft_roleplay: {args.sft_roleplay} (角色扮演 SFT)")
    print(f"  roleplay_rl: {args.roleplay_rl} (角色扮演 RL)")
    print(f"  rm_sft: {args.rm_sft} (RM SFT)")
    print(f"  rm_rl: {args.rm_rl} (RM RL)")
    print(f"  rl_test: {args.rl_test} (RL 测试集)")
    print()
    
    sizes = {
        'sft_roleplay': args.sft_roleplay,
        'roleplay_rl': args.roleplay_rl,
        'rm_sft': args.rm_sft,
        'rm_rl': args.rm_rl,
        'rl_test': args.rl_test
    }
    
    split_data(args.input, args.output_dir, sizes, args.seed)


if __name__ == "__main__":
    main()

