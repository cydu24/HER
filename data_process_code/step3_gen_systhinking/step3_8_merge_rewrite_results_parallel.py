#!/usr/bin/env python3
"""
并行合并 sys_thinking 改写结果，并分析失败原因
"""

import json
import os
import glob
from tqdm import tqdm
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import re

# 路径配置
OUTPUT_DIR = "/path/to/data/example"
SFT_DATA_PATH = "/path/to/data/example"
OUTPUT_PATH = "/path/to/data/example"
FAILED_PATH = "/path/to/data/example"
FAILED_ANALYSIS_PATH = "/path/to/data/example"

NUM_WORKERS = 128


def process_single_file(filepath):
    """处理单个输出文件"""
    results = {}
    fails = []
    fail_reasons = Counter()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                raw = d.get('raw_record', {})
                trace_id = raw.get('original_trace_id', '')
                character = raw.get('character_name', '')
                vulcan_trace_id = d.get('trace_id', '')
                
                vo = d.get('vulcan_output', {})
                candidates = vo.get('model_req', {}).get('candidates', [])
                
                if not candidates:
                    fail_reasons['no_candidates'] += 1
                    fails.append({
                        'trace_id': vulcan_trace_id,
                        'reason': 'no_candidates',
                        'raw_record': raw
                    })
                    continue
                
                parts = candidates[0].get('content', {}).get('parts', [])
                if len(parts) < 2:
                    fail_reasons['no_output_parts'] += 1
                    fails.append({
                        'trace_id': vulcan_trace_id,
                        'reason': 'no_output_parts',
                        'raw_record': raw
                    })
                    continue
                
                text = parts[1].get('text', '')
                
                # 尝试提取 JSON
                json_str = None
                if '```json' in text:
                    start = text.find('```json') + 7
                    end = text.find('```', start)
                    if end > start:
                        json_str = text[start:end].strip()
                elif '```' in text:
                    # 可能是 ```\n[...]\n```
                    match = re.search(r'```\s*\n?\s*(\[[\s\S]*?\])\s*\n?\s*```', text)
                    if match:
                        json_str = match.group(1)
                
                if not json_str and text.strip().startswith('['):
                    # 尝试找到完整的 JSON 数组
                    bracket_count = 0
                    start_idx = text.find('[')
                    for i, c in enumerate(text[start_idx:], start_idx):
                        if c == '[':
                            bracket_count += 1
                        elif c == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_str = text[start_idx:i+1]
                                break
                
                if not json_str:
                    fail_reasons['no_json_format'] += 1
                    fails.append({
                        'trace_id': vulcan_trace_id,
                        'reason': 'no_json_format',
                        'text_preview': text[:500],
                        'raw_record': raw
                    })
                    continue
                
                try:
                    items = json.loads(json_str)
                except json.JSONDecodeError as e:
                    fail_reasons['json_decode_error'] += 1
                    fails.append({
                        'trace_id': vulcan_trace_id,
                        'reason': 'json_decode_error',
                        'error': str(e),
                        'json_preview': json_str[:500],
                        'raw_record': raw
                    })
                    continue
                
                if not items or not isinstance(items, list):
                    fail_reasons['empty_or_invalid_json'] += 1
                    continue
                
                if 'dialogue_index' not in items[0]:
                    fail_reasons['missing_dialogue_index'] += 1
                    fails.append({
                        'trace_id': vulcan_trace_id,
                        'reason': 'missing_dialogue_index',
                        'items_preview': str(items)[:500],
                        'raw_record': raw
                    })
                    continue
                
                # 成功解析
                for item in items:
                    d_idx = item.get('dialogue_index')
                    revised = item.get('revised_sys_thinking', '')
                    if revised and trace_id and character:
                        key = (trace_id, character, d_idx)
                        results[key] = revised
                        
            except Exception as e:
                fail_reasons['unexpected_error'] += 1
                
    return results, fails, dict(fail_reasons)


def load_rewrite_results_parallel():
    """并行加载所有改写结果"""
    output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.jsonl"))
    print(f"找到 {len(output_files)} 个输出文件，使用 {NUM_WORKERS} 个进程并行处理")
    
    results = {}
    all_fails = []
    total_fail_reasons = Counter()
    
    with Pool(NUM_WORKERS) as pool:
        for file_results, file_fails, file_reasons in tqdm(
            pool.imap_unordered(process_single_file, output_files),
            total=len(output_files),
            desc="并行加载改写结果"
        ):
            results.update(file_results)
            all_fails.extend(file_fails)
            total_fail_reasons.update(file_reasons)
    
    print(f"\n成功解析条目数: {len(results)}")
    print(f"失败样本数: {len(all_fails)}")
    print("\n失败原因统计:")
    for reason, count in total_fail_reasons.most_common():
        print(f"  {reason}: {count}")
    
    return results, all_fails, dict(total_fail_reasons)


def analyze_failures(fails, fail_reasons):
    """分析失败原因"""
    analysis = {
        'total_fails': len(fails),
        'fail_reasons': fail_reasons,
        'samples_by_reason': defaultdict(list)
    }
    
    # 按原因分类样本
    for fail in fails[:1000]:  # 只保留前1000个样本
        reason = fail.get('reason', 'unknown')
        analysis['samples_by_reason'][reason].append({
            'trace_id': fail.get('trace_id', ''),
            'error': fail.get('error', ''),
            'text_preview': fail.get('text_preview', '')[:200],
            'json_preview': fail.get('json_preview', '')[:200]
        })
    
    # 检查是否可以重试
    retriable = 0
    for fail in fails:
        reason = fail.get('reason', '')
        # JSON 解析错误和格式问题可能通过重试解决
        if reason in ['json_decode_error', 'no_json_format']:
            retriable += 1
    
    analysis['retriable_count'] = retriable
    analysis['retriable_ratio'] = retriable / len(fails) if fails else 0
    
    return analysis


def merge_to_sft_data(results):
    """合并到 SFT 数据"""
    updated_count = 0
    not_found_count = 0
    total_turns = 0
    
    with open(SFT_DATA_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="合并数据"):
            sample = json.loads(line)
            trace_id = sample.get('trace_id_book_chapter', '') or sample.get('trace_id', '')
            
            training_samples = sample.get('training_samples', {})
            
            for character, turns in training_samples.items():
                for turn in turns:
                    if turn.get('role') != 'assistant':
                        continue
                    
                    total_turns += 1
                    origin_id = turn.get('origin_id')
                    
                    if origin_id is None:
                        continue
                    
                    if isinstance(origin_id, list):
                        origin_id = origin_id[0] if origin_id else None
                    
                    if origin_id is None:
                        continue
                    
                    key = (trace_id, character, origin_id)
                    if key in results:
                        turn['sys_thinking_revised'] = results[key]
                        updated_count += 1
                    else:
                        not_found_count += 1
            
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return total_turns, updated_count, not_found_count


def save_failed_for_retry(fails):
    """保存失败样本供重试"""
    # 只保存可重试的
    retriable_fails = [
        f for f in fails 
        if f.get('reason') in ['json_decode_error', 'no_json_format']
    ]
    
    with open(FAILED_PATH, 'w', encoding='utf-8') as f:
        for fail in retriable_fails:
            f.write(json.dumps(fail, ensure_ascii=False) + '\n')
    
    print(f"\n保存了 {len(retriable_fails)} 个可重试的失败样本到 {FAILED_PATH}")
    return len(retriable_fails)


def main():
    print("=" * 60)
    print("并行合并 sys_thinking 改写结果")
    print("=" * 60)
    
    # 1. 并行加载改写结果
    results, fails, fail_reasons = load_rewrite_results_parallel()
    
    # 2. 分析失败原因
    print("\n" + "=" * 60)
    print("分析失败原因")
    print("=" * 60)
    analysis = analyze_failures(fails, fail_reasons)
    
    # 保存分析结果
    with open(FAILED_ANALYSIS_PATH, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"失败分析已保存到 {FAILED_ANALYSIS_PATH}")
    
    # 3. 保存失败样本供重试
    retriable_count = save_failed_for_retry(fails)
    
    # 4. 合并到 SFT 数据
    print("\n" + "=" * 60)
    print("合并到 SFT 数据")
    print("=" * 60)
    total_turns, updated_count, not_found_count = merge_to_sft_data(results)
    
    # 5. 打印总结
    print("\n" + "=" * 60)
    print("合并完成总结")
    print("=" * 60)
    print(f"总 assistant turns: {total_turns}")
    print(f"已更新: {updated_count} ({updated_count/total_turns*100:.1f}%)")
    print(f"未找到: {not_found_count} ({not_found_count/total_turns*100:.1f}%)")
    print(f"\n可重试失败样本: {retriable_count}")
    print(f"重试后预计提升: +{retriable_count * 4} turns (假设每个样本~4轮)")
    print(f"\n输出文件: {OUTPUT_PATH}")
    
    # 6. 是否建议重试
    print("\n" + "=" * 60)
    print("重试建议")
    print("=" * 60)
    if retriable_count > 1000:
        print(f"⚠️ 建议重新推理 {retriable_count} 个失败样本")
        print(f"   主要失败原因: {list(fail_reasons.keys())[:3]}")
    else:
        print("✅ 失败数量较少，可以接受")


if __name__ == "__main__":
    main()

