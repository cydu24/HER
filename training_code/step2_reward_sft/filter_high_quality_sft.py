#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建最终数据集（SFT、RL、Test）
筛选条件：
1. principle >= 3个
2. 按照winner分布分配到不同数据集
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import re
from multiprocessing import Pool, cpu_count
from functools import partial

def extract_json_from_response(response_text):
    """从response文本中提取JSON"""
    if not response_text:
        return None
    
    # 尝试提取```json和```之间的内容
    match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    
    # 如果没有markdown格式，直接尝试解析
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None

def check_quality(data):
    """
    检查数据是否满足高质量条件
    返回: (is_high_quality, num_principles, cand_1_count, cand_2_count, tie_count)
    """
    response_text = data.get('response', '')
    
    # 解析response JSON
    response_json = extract_json_from_response(response_text)
    if not response_json:
        return False, 0, 0, 0, 0
    
    # 获取result
    result_list = response_json.get('result', [])
    if not result_list:
        return False, 0, 0, 0, 0
    
    result = result_list[0] if isinstance(result_list, list) else result_list
    
    # 获取principle数量
    principles = result.get('principle', {})
    num_principles = len(principles)
    
    # 获取analysis中的principle_comparisons
    analysis = result.get('analysis', {})
    comparisons = analysis.get('principle_comparisons', [])
    
    # 统计winner分布
    cand_1_count = 0
    cand_2_count = 0
    tie_count = 0
    
    for comp in comparisons:
        winner = comp.get('winner', '')
        if winner == 'cand_1':
            cand_1_count += 1
        elif winner == 'cand_2':
            cand_2_count += 1
        elif winner == 'tie':
            tie_count += 1
    
    # 判断是否高质量
    # 条件1: principle >= 3
    # 条件2: 既有cand_1也有cand_2（不能只选一边）
    is_high_quality = (num_principles >= 1) and (cand_1_count > 0) and (cand_2_count > 0)
    
    return is_high_quality, num_principles, cand_1_count, cand_2_count, tie_count

def process_line(line):
    """
    处理单行数据，返回分类结果和统计信息
    返回: (data, is_high_quality, num_principles, cand_1_count, cand_2_count, tie_count, error)
    """
    if not line.strip():
        return None
    
    try:
        data = json.loads(line)
        is_hq, num_prin, c1, c2, tie = check_quality(data)
        return (data, is_hq, num_prin, c1, c2, tie, False)
    except Exception as e:
        return (None, False, 0, 0, 0, 0, True)

def filter_high_quality_data(input_file, output_dir, num_workers=256, 
                            cand1_only_limit=10000, cand2_only_limit=10000,
                            both_sides_limit=20000, tie_for_rl=50, tie_for_test=5):
    """
    构建最终数据集（并行版本）
    策略：
    SFT数据（principle >= 3）：
    1. 双边数据：按比例采样
    2. cand_1_only: 按比例采样
    3. cand_2_only: 按比例采样
    4. tie_only: 剩余部分（扣除给RL和Test的）
    5. no_winner: 全部
    
    RL数据：
    - both_sides剩余的
    - cand_1_only剩余的
    - cand_2_only剩余的
    - tie_only的50条
    
    Test数据（200条）：
    - 从RL数据中随机抽取195条
    - 从tie_only中随机抽取5条
    """
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 输出文件
    sft_final_file = os.path.join(output_dir, "sft_final.jsonl")
    rl_final_file = os.path.join(output_dir, "rl_final.jsonl")
    test_final_file = os.path.join(output_dir, "test_final.jsonl")
    
    # 统计信息
    stats = {
        'total': 0,
        'parse_error': 0,
        'principle_distribution': {},  # principle数量分布
        'winner_patterns': {
            'both_sides': 0,  # 既有cand_1也有cand_2
            'cand_1_only': 0,  # 只有cand_1
            'cand_2_only': 0,  # 只有cand_2
            'tie_only': 0,    # 只有tie
            'no_winner': 0    # 没有winner
        }
    }
    
    print("=" * 70)
    print(f"🚀 构建最终数据集（使用 {num_workers} 个进程）...")
    print("=" * 70)
    
    # 读取所有行
    print("📖 读取数据...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    
    total_lines = len(lines)
    print(f"总计 {total_lines} 条数据")
    
    # 并行处理
    print(f"⚙️ 使用 {num_workers} 个进程并行处理...")
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_line, lines, chunksize=100),
            total=total_lines,
            desc="筛选数据"
        ))
    
    # 分类收集数据
    print("📦 分类数据...")
    both_sides_data = []
    cand1_only_data = []
    cand2_only_data = []
    tie_only_data = []
    no_winner_data = []
    low_quality_data = []
    
    for result in results:
        if result is None:
            continue
        
        data, is_hq, num_prin, c1, c2, tie, error = result
        
        if error:
            stats['parse_error'] += 1
            continue
        
        stats['total'] += 1
        
        # 统计principle分布
        prin_key = f"{num_prin}个principle"
        stats['principle_distribution'][prin_key] = stats['principle_distribution'].get(prin_key, 0) + 1
        
        # 统计winner模式
        if c1 > 0 and c2 > 0:
            stats['winner_patterns']['both_sides'] += 1
        elif c1 > 0 and c2 == 0:
            stats['winner_patterns']['cand_1_only'] += 1
        elif c2 > 0 and c1 == 0:
            stats['winner_patterns']['cand_2_only'] += 1
        elif tie > 0 and c1 == 0 and c2 == 0:
            stats['winner_patterns']['tie_only'] += 1
        else:
            stats['winner_patterns']['no_winner'] += 1
        
        # 分类：principle >= 3 且符合不同的winner模式
        if num_prin >= 3:
            if c1 > 0 and c2 > 0:
                both_sides_data.append(data)
            elif c1 > 0 and c2 == 0:
                cand1_only_data.append(data)
            elif c2 > 0 and c1 == 0:
                cand2_only_data.append(data)
            elif tie > 0 and c1 == 0 and c2 == 0:
                tie_only_data.append(data)
            elif c1 == 0 and c2 == 0 and tie == 0:
                no_winner_data.append(data)
            else:
                low_quality_data.append(data)
        else:
            low_quality_data.append(data)
    
    # 步骤1: 数据分类统计
    print("\n【步骤1】数据分类统计")
    print(f"  ✓ Both_sides（双边数据）: {len(both_sides_data)} 条")
    print(f"  ✓ Cand_1_only: {len(cand1_only_data)} 条")
    print(f"  ✓ Cand_2_only: {len(cand2_only_data)} 条")
    print(f"  ✓ Tie_only: {len(tie_only_data)} 条")
    print(f"  ✓ No_winner: {len(no_winner_data)} 条")
    print(f"  ⚠️ 低质量数据: {len(low_quality_data)} 条")
    
    # 步骤2: 采样SFT数据
    print("\n【步骤2】采样数据（SFT、RL、Test）")
    import random
    random.seed(42)
    
    # 1. 双边数据：20000条给SFT，剩余给RL
    if len(both_sides_data) >= both_sides_limit:
        both_sides_sft = random.sample(both_sides_data, both_sides_limit)
        both_sides_rl = [d for d in both_sides_data if d not in both_sides_sft]
        print(f"  Both_sides: 抽取{both_sides_limit}条给SFT，剩余{len(both_sides_rl)}条给RL")
    else:
        both_sides_sft = both_sides_data
        both_sides_rl = []
        print(f"  Both_sides: 全部{len(both_sides_sft)}条给SFT")
    
    # 2. Cand_1_only：10000条给SFT，剩余给RL
    if len(cand1_only_data) >= cand1_only_limit:
        cand1_sft = random.sample(cand1_only_data, cand1_only_limit)
        cand1_rl = [d for d in cand1_only_data if d not in cand1_sft]
        print(f"  Cand_1_only: 抽取{cand1_only_limit}条给SFT，剩余{len(cand1_rl)}条给RL")
    else:
        cand1_sft = cand1_only_data
        cand1_rl = []
        print(f"  Cand_1_only: 全部{len(cand1_sft)}条给SFT")
    
    # 3. Cand_2_only：10000条给SFT，剩余给RL
    if len(cand2_only_data) >= cand2_only_limit:
        cand2_sft = random.sample(cand2_only_data, cand2_only_limit)
        cand2_rl = [d for d in cand2_only_data if d not in cand2_sft]
        print(f"  Cand_2_only: 抽取{cand2_only_limit}条给SFT，剩余{len(cand2_rl)}条给RL")
    else:
        cand2_sft = cand2_only_data
        cand2_rl = []
        print(f"  Cand_2_only: 全部{len(cand2_sft)}条给SFT")
    
    # 4. Tie_only：分配给Test(5条)、RL(50条)、SFT(剩余)
    if len(tie_only_data) >= tie_for_test + tie_for_rl:
        # 先抽取给Test的
        tie_test = random.sample(tie_only_data, tie_for_test)
        tie_remaining = [d for d in tie_only_data if d not in tie_test]
        
        # 再从剩余中抽取给RL的
        if len(tie_remaining) >= tie_for_rl:
            tie_rl = random.sample(tie_remaining, tie_for_rl)
            tie_sft = [d for d in tie_remaining if d not in tie_rl]
        else:
            tie_rl = tie_remaining
            tie_sft = []
        
        print(f"  Tie_only: {len(tie_sft)}条给SFT，{len(tie_rl)}条给RL，{len(tie_test)}条给Test")
    else:
        # 数据不足，优先保证Test，其余给SFT
        if len(tie_only_data) >= tie_for_test:
            tie_test = random.sample(tie_only_data, tie_for_test)
            tie_sft = [d for d in tie_only_data if d not in tie_test]
            tie_rl = []
        else:
            tie_test = tie_only_data
            tie_sft = []
            tie_rl = []
        print(f"  Tie_only: {len(tie_sft)}条给SFT，{len(tie_rl)}条给RL，{len(tie_test)}条给Test（数据不足{tie_for_test + tie_for_rl}条）")
    
    # 5. No_winner：全部给SFT
    print(f"  No_winner: 全部{len(no_winner_data)}条给SFT")
    
    # 步骤3: 构建RL和Test数据
    print("\n【步骤3】构建RL和Test数据")
    
    # 合并所有剩余数据作为RL候选（不包括tie，tie已经单独处理）
    rl_candidates = both_sides_rl + cand1_rl + cand2_rl + tie_rl
    print(f"  RL候选数据: {len(rl_candidates)} 条（both_sides剩余{len(both_sides_rl)} + cand_1剩余{len(cand1_rl)} + cand_2剩余{len(cand2_rl)} + tie {len(tie_rl)}）")
    
    # 从RL候选中抽取195条，加上tie的5条，总共200条Test
    test_num_from_rl = 195
    if len(rl_candidates) >= test_num_from_rl:
        test_data_from_rl = random.sample(rl_candidates, test_num_from_rl)
        rl_data = [d for d in rl_candidates if d not in test_data_from_rl]
        test_data = test_data_from_rl + tie_test
        print(f"  Test数据: {len(test_data)}条（从RL抽取{len(test_data_from_rl)}条 + tie {len(tie_test)}条）")
        print(f"  RL数据: {len(rl_data)}条")
    else:
        test_data = rl_candidates + tie_test
        rl_data = []
        print(f"  ⚠️ RL候选不足{test_num_from_rl}条，Test数据: {len(test_data)}条")
        print(f"  RL数据: 0条")
    
    # 步骤4: 写入最终数据
    print("\n【步骤4】写入最终数据")
    
    # 写入SFT数据
    sft_count = 0
    print(f"  ✍️ 写入SFT数据...")
    with open(sft_final_file, 'w', encoding='utf-8') as f:
        # 1. 双边数据（采样后的）
        for data in both_sides_sft:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
            sft_count += 1
        
        # 2. cand_1_only采样
        for data in cand1_sft:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
            sft_count += 1
        
        # 3. cand_2_only采样
        for data in cand2_sft:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
            sft_count += 1
        
        # 4. tie_only（剩余部分）
        for data in tie_sft:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
            sft_count += 1
        
        # 5. no_winner全部
        for data in no_winner_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
            sft_count += 1
    
    print(f"    ✓ SFT: {sft_count} 条")
    
    # 写入RL数据
    print(f"  ✍️ 写入RL数据...")
    with open(rl_final_file, 'w', encoding='utf-8') as f:
        for data in rl_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    print(f"    ✓ RL: {len(rl_data)} 条")
    
    # 写入Test数据
    print(f"  ✍️ 写入Test数据...")
    with open(test_final_file, 'w', encoding='utf-8') as f:
        for data in test_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    print(f"    ✓ Test: {len(test_data)} 条")
    
    # 统计信息
    stats['sft'] = {
        'both_sides': len(both_sides_sft),
        'cand_1_only': len(cand1_sft),
        'cand_2_only': len(cand2_sft),
        'tie_only': len(tie_sft),
        'no_winner': len(no_winner_data),
        'total': sft_count
    }
    stats['rl'] = {
        'both_sides_remaining': len(both_sides_rl),
        'cand_1_only_remaining': len(cand1_rl),
        'cand_2_only_remaining': len(cand2_rl),
        'tie_only': len(tie_rl),
        'total': len(rl_data)
    }
    stats['test'] = {
        'from_rl': len(test_data) - len(tie_test) if len(test_data) >= len(tie_test) else 0,
        'from_tie': len(tie_test),
        'total': len(test_data)
    }
    stats['grand_total'] = sft_count + len(rl_data) + len(test_data)
    stats['composition'] = {
        'both_sides_total': len(both_sides_data),
        'both_sides_sft': len(both_sides_sft),
        'both_sides_rl': len(both_sides_rl),
        'cand1_only_total': len(cand1_only_data),
        'cand1_only_sft': len(cand1_sft),
        'cand1_only_rl': len(cand1_rl),
        'cand2_only_total': len(cand2_only_data),
        'cand2_only_sft': len(cand2_sft),
        'cand2_only_rl': len(cand2_rl),
        'tie_only_total': len(tie_only_data),
        'tie_only_sft': len(tie_sft),
        'tie_only_rl': len(tie_rl),
        'tie_only_test': len(tie_test),
        'no_winner': len(no_winner_data)
    }
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, "final_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 输出最终统计
    print("\n" + "=" * 70)
    print("📊 最终数据集统计")
    print("=" * 70)
    
    print(f"\n✅ SFT数据 (principle>=3): {sft_count} 条")
    print(f"  ├─ 双边数据（both_sides）: {len(both_sides_sft)} 条")
    print(f"  ├─ Cand_1_only采样: {len(cand1_sft)} 条")
    print(f"  ├─ Cand_2_only采样: {len(cand2_sft)} 条")
    print(f"  ├─ Tie_only: {len(tie_sft)} 条")
    print(f"  └─ No_winner: {len(no_winner_data)} 条")
    
    print(f"\n✅ RL数据: {len(rl_data)} 条")
    print(f"  ├─ Both_sides剩余: {len(both_sides_rl)} 条")
    print(f"  ├─ Cand_1_only剩余: {len(cand1_rl)} 条")
    print(f"  ├─ Cand_2_only剩余: {len(cand2_rl)} 条")
    print(f"  └─ Tie_only: {len(tie_rl)} 条")
    
    print(f"\n✅ Test数据: {len(test_data)} 条")
    print(f"  ├─ 从RL抽取: {stats['test']['from_rl']} 条")
    print(f"  └─ 从Tie抽取: {stats['test']['from_tie']} 条")
    
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"总计: {stats['grand_total']} 条")
    print("=" * 70)
    
    print(f"\n📈 Principle数量分布（全部数据）:")
    for key in sorted(stats['principle_distribution'].keys(), key=lambda x: int(x.split('个')[0])):
        count = stats['principle_distribution'][key]
        percentage = count / stats['total'] * 100
        print(f"  {key}: {count} 条 ({percentage:.1f}%)")
    
    prin3_plus = sum(count for key, count in stats['principle_distribution'].items() 
                     if int(key.split('个')[0]) >= 3)
    print(f"  → Principle>=3总计: {prin3_plus} 条 ({prin3_plus/stats['total']*100:.1f}%)")
    
    print(f"\n🎯 Winner模式分布（全部数据）:")
    for key, count in stats['winner_patterns'].items():
        percentage = count / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {key}: {count} 条 ({percentage:.1f}%)")
    
    print(f"\n📁 输出文件:")
    print(f"  - {sft_final_file}")
    print(f"  - {rl_final_file}")
    print(f"  - {test_final_file}")
    print(f"  - {stats_file}")
    
    print("\n" + "=" * 70)
    if sft_count >= 40000:
        print(f"🎉 完美！SFT数据达到 {sft_count} 条，已超过40000的目标！")
    else:
        print(f"✨ SFT数据: {sft_count} 条")
    print("=" * 70)
    
    return stats

def main():
    input_file = "/path/to/data/example"
    output_dir = "/path/to/data/example"
    num_workers = 256 
    # input_file = "/path/to/data/example"
    # output_dir = "/path/to/data/example"
    # num_workers = 256 
    input_file = "/path/to/data/example"
    output_dir = "/path/to/data/example"
    num_workers = 256 
    print("=" * 70)
    print("🚀 构建最终数据集（SFT、RL、Test）")
    print("=" * 70)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"并行进程数: {num_workers}")
    print("=" * 70)
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 调用筛选函数
    # - both_sides: 20000条给SFT，剩余给RL
    # - cand1_only和cand2_only: 各10000条给SFT，剩余给RL
    # - tie_only: 50条给RL，5条给Test，剩余给SFT
    stats = filter_high_quality_data(
        input_file, output_dir, num_workers, 
        cand1_only_limit=10000, 
        cand2_only_limit=10000,
        both_sides_limit=20000,
        tie_for_rl=50,
        tie_for_test=5
    )

if __name__ == "__main__":
    main()

