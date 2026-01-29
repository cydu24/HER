#!/usr/bin/env python3
"""
Step 3.3: 从 Vulcan 输出中抽取 model thinking 和 response

输入: Vulcan 输出文件 (包含 model_request_output)
输出: 抽取后的数据 (thinking, response)

分隔逻辑:
1. 优先用 </thinker> 分割
2. 然后用 </think> 分割
3. 最后用 "角色名: <role_thinking>" 模式找到分割点
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm


# 所有可能的 thinking 结尾标签（按优先级排序）
THINKING_END_TAGS = [
    '</thinker>',
    '</think>',
    '</thinking>',
    '</thought>',
]


def extract_thinking_and_response(text: str) -> Tuple[str, str, str]:
    """
    从模型输出中抽取 thinking 和 response
    
    Returns:
        (thinking, response, method) - method 表示使用的分割方法
    """
    text = text.strip()
    
    # 方法1: 按各种结尾标签分割
    for tag in THINKING_END_TAGS:
        if tag in text:
            parts = text.split(tag, 1)
            thinking = parts[0].strip()
            response = parts[1].strip() if len(parts) > 1 else ''
            # 移除可能的开头标签 (如 <think>, <thinker> 等)
            tag_name = tag[2:-1]  # 提取标签名，如 'think'
            for open_tag in [f'<{tag_name}>', f'<{tag_name} ']:
                if thinking.startswith(open_tag):
                    thinking = thinking[len(open_tag):].strip()
                    break
            method = tag_name  # 使用标签名作为方法名
            return thinking, response, method
    
    # 方法2: 角色名: <role_thinking> 模式
    # 匹配 "Name: <role_thinking>" 或 "Name Name: <role_thinking>"
    name_pattern = r'\n([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*): (<role_thinking>|<role_action>|[^<\n])'
    match = re.search(name_pattern, text)
    if match:
        split_pos = match.start()
        thinking = text[:split_pos].strip()
        response = text[split_pos:].strip()
        return thinking, response, 'name_pattern'
    
    # 方法3: 直接找 <role_thinking> 开头
    role_thinking_match = re.search(r'(<role_thinking>)', text)
    if role_thinking_match:
        split_pos = role_thinking_match.start()
        thinking = text[:split_pos].strip()
        response = text[split_pos:].strip()
        return thinking, response, 'role_thinking_direct'
    
    # 无法分割，全部作为 response
    return '', text, 'no_split'


def process_vulcan_output(input_file: str, output_file: str) -> Dict:
    """处理 Vulcan 输出文件"""
    
    stats = {
        'total': 0,
        'success': 0,
        'no_output': 0,
        'no_text': 0,
        'methods': {}  # 动态统计各种方法
    }
    
    results = []
    
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc=f'Processing {Path(input_file).name}'):
            stats['total'] += 1
            data = json.loads(line)
            
            # 获取模型输出
            mro = data.get('model_request_output', {})
            candidates = mro.get('candidates', [])
            
            if not candidates:
                stats['no_output'] += 1
                continue
            
            parts = candidates[0].get('content', {}).get('parts', [])
            if not parts:
                stats['no_text'] += 1
                continue
            
            text = parts[0].get('text', '').strip()
            if not text:
                stats['no_text'] += 1
                continue
            
            # 抽取 thinking 和 response
            thinking, response, method = extract_thinking_and_response(text)
            stats['methods'][method] = stats['methods'].get(method, 0) + 1
            
            if response:
                stats['success'] += 1
            
            # 保存结果
            result = {
                'trace_id': data.get('trace_id', ''),
                'raw_record': data.get('raw_record', {}),
                'model_thinking': thinking,
                'model_response': response,
                'extraction_method': method,
                'raw_text': text,  # 保留原始输出用于调试
            }
            results.append(result)
    
    # 写入输出文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract thinking from Vulcan output')
    parser.add_argument('--input_dir', type=str, 
                        default='/path/to/data/example',
                        help='Input directory containing Vulcan output files')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/data/example',
                        help='Output directory')
    parser.add_argument('--file_pattern', type=str, default='*.jsonl',
                        help='File pattern to match')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 找到所有输入文件
    input_files = list(input_dir.glob(args.file_pattern))
    print(f'Found {len(input_files)} input files')
    
    if not input_files:
        print(f'No files found in {input_dir}')
        return
    
    total_stats = {
        'total': 0,
        'success': 0,
        'no_output': 0,
        'no_text': 0,
        'methods': {}  # 动态统计各种方法
    }
    
    for input_file in input_files:
        output_file = output_dir / input_file.name
        stats = process_vulcan_output(str(input_file), str(output_file))
        
        # 累加统计
        for key in ['total', 'success', 'no_output', 'no_text']:
            total_stats[key] += stats[key]
        for method, count in stats['methods'].items():
            total_stats['methods'][method] = total_stats['methods'].get(method, 0) + count
        
        print(f'  {input_file.name}: {stats["success"]}/{stats["total"]} success')
    
    # 打印总统计
    print('\n=== 总统计 ===')
    print(f'总数: {total_stats["total"]}')
    print(f'成功: {total_stats["success"]} ({100*total_stats["success"]/total_stats["total"]:.1f}%)')
    print(f'无输出: {total_stats["no_output"]}')
    print(f'无文本: {total_stats["no_text"]}')
    print('\n分割方法:')
    for method, count in total_stats['methods'].items():
        if count > 0:
            print(f'  {method}: {count} ({100*count/total_stats["total"]:.1f}%)')


if __name__ == '__main__':
    main()
