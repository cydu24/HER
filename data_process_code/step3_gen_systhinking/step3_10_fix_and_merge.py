#!/usr/bin/env python3
"""
修复并重新合并：
1. 修复 JSON 解析（处理未转义引号）
2. 重新加载 sys_thinking（正确解析空 raw_record）
3. 合并到 SFT 数据
"""

import json
import os
import glob
import re
from tqdm import tqdm
from collections import defaultdict, Counter
from multiprocessing import Pool

# 路径配置
OUTPUT_DIR = "/path/to/data/example"
ALL_SUCCESS_PATH = "/path/to/data/example"
SFT_DATA_PATH = "/path/to/data/example"
OUTPUT_PATH = "/path/to/data/example"

NUM_WORKERS = 128


def fix_json_quotes(s):
    """修复 JSON 字符串中未转义的引号"""
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '"' and (i == 0 or s[i-1] != '\\'):
            if not in_string:
                in_string = True
                result.append(c)
            else:
                # 检查这是否是字符串结束
                rest = s[i+1:].lstrip()
                if rest and rest[0] in ',}]:':
                    in_string = False
                    result.append(c)
                else:
                    # 字符串内部的引号，需要转义
                    result.append('\\"')
        else:
            result.append(c)
        i += 1
    return ''.join(result)


def extract_json_robust(text):
    """更鲁棒的 JSON 提取"""
    # 1. 标准 markdown 代码块
    if '```json' in text:
        match = re.search(r'```json\s*([\s\S]*?)```', text)
        if match:
            return match.group(1).strip()
    
    # 2. 普通代码块
    if '```' in text:
        match = re.search(r'```\s*(\[[\s\S]*?\])\s*```', text)
        if match:
            return match.group(1).strip()
    
    # 3. 直接找 JSON 数组
    if '[' in text:
        bracket_count = 0
        start_idx = text.find('[')
        for i, c in enumerate(text[start_idx:], start_idx):
            if c == '[':
                bracket_count += 1
            elif c == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    return text[start_idx:i+1]
    
    return None


def parse_trace_id(trace_id):
    """从 trace_id 解析 original_trace_id, character_name, assistant_index"""
    # 格式: sys_gen_BookName_Chapter_Plot_Conv/CharName/AssistantIdx
    if not trace_id.startswith('sys_gen_'):
        return None, None, None
    
    parts = trace_id.split('/')
    if len(parts) < 3:
        return None, None, None
    
    prefix = parts[0].replace('sys_gen_', '')
    char_name = parts[1]
    try:
        assistant_idx = int(parts[2])
    except:
        assistant_idx = None
    
    return prefix, char_name, assistant_idx


def process_single_file(filepath):
    """处理单个输出文件"""
    results = {}
    fail_count = 0
    fix_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                raw = d.get('raw_record', {})
                trace_id = raw.get('original_trace_id', '')
                character = raw.get('character_name', '')
                
                vo = d.get('vulcan_output', {})
                candidates = vo.get('model_req', {}).get('candidates', [])
                if not candidates:
                    fail_count += 1
                    continue
                
                parts = candidates[0].get('content', {}).get('parts', [])
                if len(parts) < 2:
                    fail_count += 1
                    continue
                
                text = parts[1].get('text', '')
                
                # 提取 JSON
                json_str = extract_json_robust(text)
                if not json_str:
                    fail_count += 1
                    continue
                
                # 尝试解析
                try:
                    items = json.loads(json_str)
                except json.JSONDecodeError:
                    # 尝试修复引号
                    fixed_str = fix_json_quotes(json_str)
                    try:
                        items = json.loads(fixed_str)
                        fix_count += 1
                    except:
                        fail_count += 1
                        continue
                
                if not items or not isinstance(items, list):
                    fail_count += 1
                    continue
                
                if 'dialogue_index' not in items[0]:
                    fail_count += 1
                    continue
                
                # 成功解析
                for item in items:
                    d_idx = item.get('dialogue_index')
                    revised = item.get('revised_sys_thinking', '')
                    if revised and trace_id and character:
                        key = (trace_id, character, d_idx)
                        results[key] = revised
                        
            except Exception as e:
                fail_count += 1
                
    return results, fail_count, fix_count


def load_original_sys_thinking():
    """从 all_success_final_v3.jsonl 加载原始 sys_thinking"""
    print("加载原始 sys_thinking...")
    
    # key: (trace_id, character, assistant_index) -> sys_thinking
    sys_thinking_map = {}
    
    with open(ALL_SUCCESS_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载 all_success"):
            d = json.loads(line)
            raw = d.get('raw_record', {})
            trace_id_full = d.get('trace_id', '')
            
            # 尝试从 raw_record 获取
            trace_id = raw.get('original_trace_id', '')
            char = raw.get('character_name', '')
            assistant_idx = raw.get('assistant_index')
            
            # 如果 raw_record 为空，从 trace_id 解析
            if not trace_id or not char:
                trace_id, char, assistant_idx = parse_trace_id(trace_id_full)
            
            if not trace_id or not char or assistant_idx is None:
                continue
            
            # 获取 sys_thinking
            sys_thinking = d.get('model_thinking', '') or d.get('model_response', '')
            
            if sys_thinking:
                key = (trace_id, char, assistant_idx)
                sys_thinking_map[key] = sys_thinking
    
    print(f"加载完成: {len(sys_thinking_map)} 条原始 sys_thinking")
    return sys_thinking_map


def load_rewrite_results_parallel():
    """并行加载改写结果"""
    output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.jsonl"))
    print(f"找到 {len(output_files)} 个输出文件")
    
    results = {}
    total_fail = 0
    total_fix = 0
    
    with Pool(NUM_WORKERS) as pool:
        for file_results, fail_count, fix_count in tqdm(
            pool.imap_unordered(process_single_file, output_files),
            total=len(output_files),
            desc="并行加载改写结果"
        ):
            results.update(file_results)
            total_fail += fail_count
            total_fix += fix_count
    
    print(f"改写结果: {len(results)} 条")
    print(f"JSON 修复: {total_fix} 条")
    print(f"仍失败: {total_fail} 条")
    
    return results


def merge_to_sft_data(rewrite_results, original_sys_thinking):
    """合并到 SFT 数据"""
    updated_revised = 0
    updated_original = 0
    not_found = 0
    total_turns = 0
    
    with open(SFT_DATA_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="合并数据"):
            sample = json.loads(line)
            trace_id = sample.get('trace_id_book_chapter', '') or sample.get('trace_id', '')
            
            training_samples = sample.get('training_samples', {})
            
            for character, turns in training_samples.items():
                assistant_idx = 0
                for turn in turns:
                    if turn.get('role') != 'assistant':
                        continue
                    
                    assistant_idx += 1
                    total_turns += 1
                    origin_id = turn.get('origin_id')
                    
                    if origin_id is None:
                        continue
                    
                    if isinstance(origin_id, list):
                        origin_id = origin_id[0] if origin_id else None
                    
                    if origin_id is None:
                        continue
                    
                    # 查找改写结果
                    key = (trace_id, character, origin_id)
                    if key in rewrite_results:
                        turn['sys_thinking_revised'] = rewrite_results[key]
                        updated_revised += 1
                    else:
                        # 查找原始 sys_thinking (用 assistant_idx)
                        key2 = (trace_id, character, assistant_idx)
                        if key2 in original_sys_thinking:
                            turn['sys_thinking_original'] = original_sys_thinking[key2]
                            updated_original += 1
                        else:
                            not_found += 1
            
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return total_turns, updated_revised, updated_original, not_found


def main():
    print("=" * 60)
    print("修复并重新合并 sys_thinking")
    print("=" * 60)
    
    # 1. 加载原始 sys_thinking
    original_sys_thinking = load_original_sys_thinking()
    
    # 2. 加载改写结果（带 JSON 修复）
    print()
    rewrite_results = load_rewrite_results_parallel()
    
    # 3. 合并
    print()
    total, revised, original, not_found = merge_to_sft_data(rewrite_results, original_sys_thinking)
    
    # 4. 总结
    print()
    print("=" * 60)
    print("合并完成")
    print("=" * 60)
    print(f"总 assistant turns: {total}")
    print(f"已更新 (revised): {revised} ({revised/total*100:.1f}%)")
    print(f"已更新 (original): {original} ({original/total*100:.1f}%)")
    print(f"总更新: {revised + original} ({(revised + original)/total*100:.1f}%)")
    print(f"未找到: {not_found} ({not_found/total*100:.1f}%)")
    print(f"\n输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

