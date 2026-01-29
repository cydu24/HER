#!/usr/bin/env python3
"""
合并和抽取 Vulcan 推理结果
"""
import json
import os
import re
import glob
from pathlib import Path
from collections import Counter
import argparse


def extract_json_from_response(response_text: str) -> dict:
    """从模型响应中提取 JSON"""
    if not response_text:
        return None
    
    # 尝试直接解析
    try:
        return json.loads(response_text)
    except:
        pass
    
    # 尝试找到 ```json ... ``` 块
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # 尝试找到 ``` ... ``` 块 (没有 json 标记)
    code_match = re.search(r'```\s*([\s\S]*?)\s*```', response_text)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except:
            pass
    
    # 尝试找到最后一个完整的 JSON 对象 (从最后一个 } 往前找匹配的 {)
    # 这是因为模型经常在 JSON 之前输出思考过程
    try:
        # 找到最后一个 }
        last_brace = response_text.rfind('}')
        if last_brace != -1:
            # 从后往前找匹配的 {
            brace_count = 0
            for i in range(last_brace, -1, -1):
                if response_text[i] == '}':
                    brace_count += 1
                elif response_text[i] == '{':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response_text[i:last_brace+1]
                        return json.loads(json_str)
    except:
        pass
    
    # 尝试找到 { ... } 块 (第一个匹配)
    brace_match = re.search(r'\{[\s\S]*\}', response_text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except:
            pass
    
    return None


def process_result_file(file_path: str) -> list:
    """处理单个结果文件"""
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                # 提取必要字段
                trace_id = data.get('trace_id', '')
                raw_record = data.get('raw_record', {})
                input_data = raw_record.get('input_data', {})
                
                # 提取模型输出
                model_output = None
                if 'model_request_output' in data:
                    try:
                        model_output = data['model_request_output']['candidates'][0]['content']['parts'][0]['text']
                    except:
                        pass
                
                if not model_output:
                    if 'vulcan_output' in data and 'model_req' in data['vulcan_output']:
                        try:
                            model_output = data['vulcan_output']['model_req']['candidates'][0]['content']['parts'][0]['text']
                        except:
                            pass
                
                # 解析 JSON
                enhanced_data = extract_json_from_response(model_output) if model_output else None
                
                # 检查状态
                status = data.get('_vulcan_doc_status', 'unknown')
                
                result = {
                    'trace_id': trace_id,
                    'input_data': input_data,
                    'model_output_raw': model_output,
                    'enhanced_data': enhanced_data,
                    'status': status,
                    'parse_success': enhanced_data is not None
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing line {line_num} in {file_path}: {e}")
                continue
    
    return results


def merge_and_extract(input_dir: str, output_dir: str, lang: str = 'en'):
    """合并和抽取结果"""
    # 找到所有结果文件
    result_files = sorted(glob.glob(os.path.join(input_dir, '*.jsonl')))
    print(f"找到 {len(result_files)} 个结果文件")
    
    # 统计
    stats = {
        'total': 0,
        'succeeded': 0,
        'failed': 0,
        'parse_success': 0,
        'parse_failed': 0
    }
    
    # 处理所有文件
    all_results = []
    for file_path in result_files:
        print(f"处理: {os.path.basename(file_path)}")
        results = process_result_file(file_path)
        all_results.extend(results)
        
        for r in results:
            stats['total'] += 1
            if r['status'] == 'succeeded':
                stats['succeeded'] += 1
            else:
                stats['failed'] += 1
            if r['parse_success']:
                stats['parse_success'] += 1
            else:
                stats['parse_failed'] += 1
    
    print(f"\n总计处理 {stats['total']} 条记录")
    print(f"  - 推理成功: {stats['succeeded']} ({stats['succeeded']*100/stats['total']:.1f}%)")
    print(f"  - 推理失败: {stats['failed']} ({stats['failed']*100/stats['total']:.1f}%)")
    print(f"  - JSON解析成功: {stats['parse_success']} ({stats['parse_success']*100/stats['total']:.1f}%)")
    print(f"  - JSON解析失败: {stats['parse_failed']} ({stats['parse_failed']*100/stats['total']:.1f}%)")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整结果
    full_output_path = os.path.join(output_dir, f'enhanced_results_{lang}_full.jsonl')
    with open(full_output_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\n完整结果已保存: {full_output_path}")
    
    # 保存成功解析的结果
    success_results = [r for r in all_results if r['parse_success']]
    success_output_path = os.path.join(output_dir, f'enhanced_results_{lang}_success.jsonl')
    with open(success_output_path, 'w', encoding='utf-8') as f:
        for r in success_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"成功解析结果已保存: {success_output_path}")
    
    # 保存只含增强对话的简化结果
    simplified_results = []
    for r in success_results:
        if r['enhanced_data'] and 'enhanced_dialogues' in r['enhanced_data']:
            simplified = {
                'trace_id': r['trace_id'],
                'book_name': r['input_data'].get('book_name', ''),
                'chapter': r['input_data'].get('chapter', ''),
                'scenario': r['input_data'].get('scenario', ''),
                'key_characters': r['input_data'].get('key_characters', []),
                'original_dialogues': r['input_data'].get('dialogues', []),
                'enhanced_dialogues': r['enhanced_data']['enhanced_dialogues'],
                'statistics': r['enhanced_data'].get('statistics', {})
            }
            simplified_results.append(simplified)
    
    simplified_output_path = os.path.join(output_dir, f'enhanced_dialogues_{lang}.jsonl')
    with open(simplified_output_path, 'w', encoding='utf-8') as f:
        for r in simplified_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"简化对话结果已保存: {simplified_output_path}")
    
    # 分析模式分布
    pattern_counter = Counter()
    for r in success_results:
        if r['enhanced_data'] and 'statistics' in r['enhanced_data']:
            patterns = r['enhanced_data']['statistics'].get('patterns_used', [])
            for p in patterns:
                pattern_counter[p] += 1
    
    print(f"\n模式分布 (Top 20):")
    for pattern, count in pattern_counter.most_common(20):
        print(f"  {count:5d} - {pattern}")
    
    # 抽取测试样本
    test_samples = simplified_results[:1000] if len(simplified_results) > 1000 else simplified_results
    test_output_path = os.path.join(output_dir, f'enhanced_dialogues_{lang}_test_1000.jsonl')
    with open(test_output_path, 'w', encoding='utf-8') as f:
        for r in test_samples:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\n测试样本已保存: {test_output_path} ({len(test_samples)} 条)")
    
    return stats, simplified_results


def main():
    parser = argparse.ArgumentParser(description='合并和抽取 Vulcan 推理结果')
    parser.add_argument('--lang', '-l', type=str, default='en', choices=['en', 'zh'],
                        help='语言版本 (en/zh)')
    parser.add_argument('--input_dir', '-i', type=str, default=None,
                        help='输入目录 (推理结果目录)')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                        help='输出目录')
    
    args = parser.parse_args()
    
    base_dir = '/path/to/data/example'
    
    if args.input_dir is None:
        args.input_dir = os.path.join(base_dir, f'vulcan_data/{args.lang}/role_thinking_enhance_{args.lang}_full')
    
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, f'enhanced_output/{args.lang}')
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"语言: {args.lang}")
    print()
    
    merge_and_extract(args.input_dir, args.output_dir, args.lang)


if __name__ == '__main__':
    main()

