#!/usr/bin/env python3
"""
Step 3.6: 将改写结果合并回 SFT 数据（主方法）

将生成的 sys_thinking 和 role_thinking 合并到 training_samples 中
"""
import json
import os
from typing import Dict, List, Any
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict


def load_rewrite_results(results_path: str) -> Dict[str, Dict]:
    """
    加载改写结果，建立索引
    
    Returns:
        Dict[original_trace_id/char_name -> Dict[assistant_index -> result]]
    """
    results = defaultdict(dict)
    
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                original_trace_id = data.get('original_trace_id', '')
                char_name = data.get('character_name', '')
                asst_index = data.get('assistant_index', 0)
                
                key = f"{original_trace_id}/{char_name}"
                results[key][asst_index] = {
                    'sys_thinking': data.get('sys_thinking', ''),
                    'role_thinking': data.get('role_thinking', ''),
                    'original_think': data.get('original_think', '')
                }
            except:
                continue
    
    return results


def merge_to_training_samples(sft_data: Dict, rewrite_map: Dict) -> Dict:
    """
    将改写结果合并到 training_samples 中
    
    在每个 assistant 回复前插入 sys_thinking
    """
    updated_data = deepcopy(sft_data)
    trace_id = sft_data.get('trace_id_book_chapter', '')
    
    total_merged = 0
    
    for char_name, samples in updated_data.get('training_samples', {}).items():
        key = f"{trace_id}/{char_name}"
        char_results = rewrite_map.get(key, {})
        
        if not char_results:
            continue
        
        # 找到所有 assistant 位置并记录索引
        new_samples = []
        assistant_count = 0
        
        for sample in samples:
            role = sample.get('role', '')
            
            if role == 'assistant':
                assistant_count += 1
                
                # 检查是否有对应的改写结果
                if assistant_count in char_results:
                    result = char_results[assistant_count]
                    
                    # 插入 sys_thinking
                    if result.get('sys_thinking'):
                        sys_thinking_sample = {
                            'role': 'assistant',
                            'type': 'sys_thinking',
                            'content': result['sys_thinking'],
                            'generated': True
                        }
                        new_samples.append(sys_thinking_sample)
                        total_merged += 1
                    
                    # 更新 assistant 的 role_thinking（如果有改进）
                    if result.get('role_thinking'):
                        # 可以选择替换或保留原有的 role_thinking
                        sample['enhanced_role_thinking'] = result['role_thinking']
            
            new_samples.append(sample)
        
        updated_data['training_samples'][char_name] = new_samples
    
    return updated_data, total_merged


def process_file(sft_input: str, results_input: str, output_path: str):
    """
    处理文件
    """
    print(f"SFT 输入: {sft_input}")
    print(f"改写结果: {results_input}")
    print(f"输出: {output_path}")
    
    # 加载改写结果
    print("加载改写结果...")
    rewrite_map = load_rewrite_results(results_input)
    print(f"加载了 {len(rewrite_map)} 个 trace_id 的结果")
    
    # 处理 SFT 数据
    total = 0
    total_merged = 0
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(sft_input, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="合并数据"):
            try:
                data = json.loads(line)
                total += 1
                
                updated_data, merged_count = merge_to_training_samples(data, rewrite_map)
                total_merged += merged_count
                
                fout.write(json.dumps(updated_data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    print(f"\n=== 完成 ===")
    print(f"总记录数: {total}")
    print(f"合并的 sys_thinking 数: {total_merged}")
    print(f"输出: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 3.6: 合并到 SFT 数据')
    parser.add_argument('--sft_input', '-s', type=str,
                        default='/path/to/data/example',
                        help='SFT 数据输入')
    parser.add_argument('--results_input', '-r', type=str,
                        default='/path/to/data/example',
                        help='改写结果输入')
    parser.add_argument('--output', '-o', type=str,
                        default='/path/to/data/example',
                        help='输出文件')
    
    args = parser.parse_args()
    
    process_file(args.sft_input, args.results_input, args.output)


if __name__ == '__main__':
    main()

