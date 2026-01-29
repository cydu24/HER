#!/usr/bin/env python3
"""
分割 SFT 数据为 训练集 和 测试集

逻辑：
1. 读取 test_set_coser.json，获取测试集涉及的书籍列表
2. 从 sft_train_data.jsonl 中分离出这些书籍的数据作为测试集
3. 剩余数据作为训练集
4. 训练集可以进一步按比例分为 train/val

输入: sft_train_data.jsonl, test_set_coser.json
输出: sft_train.jsonl, sft_test.jsonl (可选: sft_val.jsonl)
"""

import json
import random
import argparse
from tqdm import tqdm
from collections import defaultdict


def load_test_cases(test_file):
    """从测试集文件中提取精确的测试 case (book, i_p, i_c)"""
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_keys = set()
    for item in data:
        book = item.get('book_name', item.get('book', ''))
        i_p = item.get('i_p')
        i_c = item.get('i_c')
        if book and i_p is not None and i_c is not None:
            # 标准化书名
            normalized_book = book.replace(' ', '_')
            test_keys.add((normalized_book, i_p, i_c))
    
    return test_keys


def extract_key_from_trace_id(trace_id):
    """从 trace_id 中提取 (book, i_p, i_c)
    
    trace_id 格式: 书名_章节_i_p_i_c_角色名
    例如: Pride_and_Prejudice_Chapter_57_70_0_Elizabeth_Bennet
    
    需要从末尾往前解析
    """
    import re
    
    # trace_id 格式: {book_name}_{chapter}_{i_p}_{i_c}_{character}
    # 从末尾开始匹配: ..._数字_数字_角色名
    # 角色名可能包含空格（已转为下划线）
    
    # 更简单的方法：找到最后两个连续的数字部分
    parts = trace_id.rsplit('_', 10)  # 最多从末尾拆分10个部分
    
    # 从后往前找两个连续的数字
    i_p = None
    i_c = None
    book_end_idx = len(parts)
    
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].isdigit() and i > 0 and parts[i-1].isdigit():
            i_c = int(parts[i])
            i_p = int(parts[i-1])
            book_end_idx = i - 1
            break
    
    if i_p is None or i_c is None:
        return None
    
    # 书名是前面的部分，去掉章节名（章节名在 i_p 之前）
    # 需要找到书名的结束位置
    # 书名格式: Pride_and_Prejudice 或 2001_A_Space_Odyssey_(Space_Odyssey,_#1)
    # 章节名格式: Chapter_57 或 Anomaly 等
    
    # 简化处理：返回整个前缀作为 key 的一部分
    prefix = '_'.join(parts[:book_end_idx])
    
    return (prefix, i_p, i_c)


def split_dataset(input_file, test_file, output_dir, val_ratio=0.05, seed=42):
    """分割数据集"""
    
    random.seed(seed)
    
    # 加载测试集精确 case
    test_keys = load_test_cases(test_file)
    print(f"测试集共 {len(test_keys)} 个 case")
    
    print(f"\n示例测试 case:")
    for i, key in enumerate(sorted(test_keys)[:5]):
        print(f"  {key}")
    print("  ...")
    
    # 读取并分类数据
    train_data = []
    test_data = []
    
    # 为了精确匹配，我们需要把 test_keys 转换为可以匹配 trace_id 的格式
    # test_keys: (normalized_book, i_p, i_c)
    # trace_id: {book}_{chapter}_{i_p}_{i_c}_{character}
    
    # 简化：只用 (i_p, i_c) + book 包含关系来匹配
    test_book_ip_ic = {}
    for book, i_p, i_c in test_keys:
        test_book_ip_ic[(i_p, i_c)] = test_book_ip_ic.get((i_p, i_c), set())
        test_book_ip_ic[(i_p, i_c)].add(book)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="分类数据"):
            data = json.loads(line)
            trace_id = data.get('trace_id', '')
            
            # 提取 i_p, i_c
            key_info = extract_key_from_trace_id(trace_id)
            
            is_test = False
            if key_info:
                prefix, i_p, i_c = key_info
                if (i_p, i_c) in test_book_ip_ic:
                    # 检查书名是否匹配
                    for test_book in test_book_ip_ic[(i_p, i_c)]:
                        if test_book in prefix:
                            is_test = True
                            break
            
            if is_test:
                test_data.append(data)
            else:
                train_data.append(data)
    
    print(f"\n分类结果:")
    print(f"  训练集: {len(train_data)}")
    print(f"  测试集 (测试书籍): {len(test_data)}")
    
    # 从训练集中分出验证集
    random.shuffle(train_data)
    val_size = int(len(train_data) * val_ratio)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]
    
    print(f"\n最终分割:")
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    print(f"  测试集: {len(test_data)}")
    
    # 保存
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'sft_train.jsonl')
    val_file = os.path.join(output_dir, 'sft_val.jsonl')
    test_file = os.path.join(output_dir, 'sft_test.jsonl')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n输出文件:")
    print(f"  {train_file}")
    print(f"  {val_file}")
    print(f"  {test_file}")
    
    # 统计测试集涉及的 (i_p, i_c) 分布
    test_case_counts = defaultdict(int)
    for item in test_data:
        trace_id = item.get('trace_id', '')
        key_info = extract_key_from_trace_id(trace_id)
        if key_info:
            _, i_p, i_c = key_info
            test_case_counts[(i_p, i_c)] += 1
    
    print(f"\n测试集 case 分布 (前10):")
    for case, count in sorted(test_case_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {case}: {count} 个样本")


def main():
    parser = argparse.ArgumentParser(description='分割 SFT 数据为训练集和测试集')
    parser.add_argument('--input', type=str,
                        default='/path/to/project/code/step1_roleplay_sft/sft_train_data.jsonl',
                        help='输入文件')
    parser.add_argument('--test_file', type=str,
                        default='/path/to/project/her_eval/data/coser/test_set_coser.json',
                        help='测试集文件')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/project/code/step1_roleplay_sft/split_data',
                        help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='验证集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    split_dataset(args.input, args.test_file, args.output_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()

