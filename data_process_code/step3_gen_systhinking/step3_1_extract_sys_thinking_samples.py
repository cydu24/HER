#!/usr/bin/env python3
"""
Step 3.1: 提取 sys_thinking 推理样本（主方法）

论文方法：让模型自发生成 think，而不是显式要求生成 system_thinking

输入: step2 的 sft_data_enhanced.jsonl
输出: 每个 assistant 回复前的推理样本，用于让模型续写（包含自发 think）
"""
import json
import os
import re
from typing import Dict, List, Any
from tqdm import tqdm
from copy import deepcopy


def extract_sys_thinking_samples(data: Dict) -> List[Dict]:
    """
    为每个角色的每个 assistant 回复提取推理样本
    
    与消融实验不同，这里不添加显式的 sys_thinking 占位符，
    而是让模型自发生成 think + response
    """
    samples = []
    
    trace_id_base = data.get('trace_id_book_chapter', '')
    book_name = data.get('book_name', '')
    chapter = data.get('chapter', '')
    
    # 获取场景信息
    scenario = ''
    if data.get('conversation'):
        scenario = data['conversation'][0].get('scenario', '')
    
    for char_name, training_samples in data.get('training_samples', {}).items():
        # 构建 trace_id
        role_trace_id = f"{trace_id_base}/{char_name}"
        
        # 找到所有 assistant 回复的位置
        assistant_indices = []
        for i, sample in enumerate(training_samples):
            if sample.get('role') == 'assistant':
                assistant_indices.append(i)
        
        # 为每个 assistant 回复生成推理样本
        for idx, asst_idx in enumerate(assistant_indices):
            # 生成唯一 trace_id
            inference_trace_id = f"{role_trace_id}/{idx+1:04d}"
            
            # 提取上下文（assistant 之前的所有内容）
            context_before = training_samples[:asst_idx]
            
            # 获取当前 assistant 回复（作为 chat_after / ground truth）
            current_assistant = training_samples[asst_idx]
            
            # 提取后续内容（用于参考）
            context_after = []
            for s in training_samples[asst_idx+1:]:
                context_after.append(s)
                if len(context_after) >= 2:
                    break
            
            sample = {
                'trace_id': inference_trace_id,
                'original_trace_id': trace_id_base,
                'character_name': char_name,
                'book_name': book_name,
                'chapter': chapter,
                'scenario': scenario,
                'assistant_index': idx + 1,
                'context_before': context_before,
                'current_assistant': current_assistant,  # ground truth response
                'context_after': context_after
            }
            
            samples.append(sample)
    
    return samples


def remove_requirements_section(content: str) -> str:
    """
    从 system prompt 中移除 ===Requirements=== 及其后面的内容
    因为我们会在 Step 3.2 中单独添加 Requirements
    """
    # 查找 ===Requirements=== 或类似标记的位置
    patterns = [
        r'===Requirements===.*$',
        r'## Instructions for roleplaying.*$',
        r'## Requirements.*$',
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    
    return content.strip()


def build_chat_history(context_before: List[Dict], ai_name: str) -> str:
    """
    构建对话历史字符串
    
    格式说明：
    - system prompt 保留 "System:" 前缀
    - user 和 assistant 的 content 已经包含角色名（如 "Josef K: ..."），直接使用
    - 不添加额外的 "User:"、"Assistant:"、"Role Thinking:" 标签
    
    注意：会移除 system prompt 中的 Requirements 部分，
    因为 Step 3.2 会单独添加新的 Requirements
    """
    chat_history = ""
    
    for ctx in context_before:
        role = ctx.get('role', '')
        content = ctx.get('content', '')
        
        if role == 'system':
            # 移除 Requirements 部分
            content = remove_requirements_section(content)
            chat_history += f"System: {content}\n\n"
        elif role == 'user':
            # user content 已经包含角色名和对话，直接使用
            chat_history += f"{content}\n\n"
        elif role == 'assistant':
            # assistant content 已经包含角色名和对话，直接使用
            chat_history += f"{content}\n\n"
    
    return chat_history.strip()


def convert_to_serving_format(samples: List[Dict]) -> List[Dict]:
    """
    转换为 serving 格式
    
    Prompt 格式参考：让模型自发生成 think + response
    对话格式：直接使用角色名（如 "Josef K: ..."），不使用 User/Assistant 标签
    """
    serving_data = []
    
    for sample in samples:
        # 提取 ai_name (角色名)
        ai_name = sample['character_name']
        
        # 构建对话历史
        chat_history = build_chat_history(sample['context_before'], ai_name)
        
        # 获取 ground truth response
        ground_truth = sample['current_assistant'].get('content', '')
        
        # chat_after 就是 ground truth（已经包含角色名）
        chat_after = ground_truth
        
        serving_sample = {
            'trace_id': sample['trace_id'],
            'original_trace_id': sample['original_trace_id'],
            'character_name': ai_name,
            'book_name': sample['book_name'],
            'chapter': sample['chapter'],
            'scenario': sample['scenario'],
            'assistant_index': sample['assistant_index'],
            'chat_history': chat_history,  # 扁平化格式（兼容旧方法）
            'context_before': sample['context_before'],  # 结构化格式（多轮对话）
            'chat_after': chat_after.strip(),
            'ai_name': ai_name,
            'ground_truth_response': ground_truth,
        }
        
        serving_data.append(serving_sample)
    
    return serving_data


def process_file(input_path: str, output_inference: str, output_serving: str):
    """处理文件"""
    print(f"输入: {input_path}")
    print(f"推理输出: {output_inference}")
    print(f"Serving输出: {output_serving}")
    
    all_samples = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="提取样本"):
            try:
                data = json.loads(line)
                samples = extract_sys_thinking_samples(data)
                all_samples.extend(samples)
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    print(f"\n提取了 {len(all_samples)} 个推理样本")
    
    # 保存推理数据
    output_dir = os.path.dirname(output_inference)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_inference, 'w', encoding='utf-8') as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 转换为 serving 格式
    serving_data = convert_to_serving_format(all_samples)
    
    with open(output_serving, 'w', encoding='utf-8') as f:
        for item in serving_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"推理数据: {output_inference} ({len(all_samples)} 条)")
    print(f"Serving数据: {output_serving} ({len(serving_data)} 条)")
    
    return len(all_samples)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 3.1: 提取 sys_thinking 推理样本（主方法）')
    parser.add_argument('--input', '-i', type=str,
                        default='/path/to/data/example',
                        help='输入文件 (step2 输出)')
    parser.add_argument('--output_inference', type=str,
                        default='/path/to/data/example',
                        help='推理样本输出')
    parser.add_argument('--output_serving', type=str,
                        default='/path/to/data/example',
                        help='Serving格式输出')
    
    args = parser.parse_args()
    
    process_file(args.input, args.output_inference, args.output_serving)


if __name__ == '__main__':
    main()

