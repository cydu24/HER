#!/usr/bin/env python3
"""
Step 3.2: 构建 Vulcan 推理数据（主方法）

论文方法：让模型自发生成 think + response
不在 prompt 中要求生成 system_thinking，而是让模型自然续写

使用多轮对话格式，让模型更沉浸在角色扮演中：
- system role: 角色设定 + 场景 + 输出格式要求
- user/assistant: 真实的多轮对话历史
"""
import json
import os
import re
import argparse
from typing import Dict, List
from tqdm import tqdm


# 输出格式要求（添加到 system prompt 末尾）
REQUIREMENTS_TEMPLATE = """===Requirements===
Your output should include **thought**, **speech**, and **action**. Use <role_thinking>your thought</role_thinking> for thoughts, which others can't see. Use <role_action>your action</role_action> for actions, which others can see. These three elements (thought, speech and action) can appear multiple times and be freely interleaved.
(all thinking is invisible to others)

**Important: Only generate the NEXT SINGLE turn of dialogue. Do not generate multiple turns or continue the conversation beyond one response.**

Start your response with "{ai_name}: " followed by your role-play response."""


def remove_requirements_section(content: str) -> str:
    """从 system prompt 中移除旧的 Requirements 部分"""
    patterns = [
        r'===Requirements===.*$',
        r'## Instructions for roleplaying.*$',
        r'## Requirements.*$',
    ]
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    return content.strip()


def construct_vulcan_item_multiturn(sample: Dict) -> Dict:
    """
    构建多轮对话格式的 Vulcan 推理项
    
    让模型更沉浸在角色扮演中，而不是"完成任务"
    """
    trace_id = sample.get('trace_id', '')
    # 兼容两种字段名
    ai_name = sample.get('ai_name', '') or sample.get('character_name', '')
    context_before = sample.get('context_before', [])
    
    # 构建多轮对话数据
    data = []
    
    for ctx in context_before:
        role = ctx.get('role', '')
        content = ctx.get('content', '')
        
        if role == 'system':
            # 清理旧的 requirements，添加新的
            content = remove_requirements_section(content)
            requirements = REQUIREMENTS_TEMPLATE.format(ai_name=ai_name)
            system_content = f"{content}\n\n{requirements}"
            data.append({
                "role": "system",
                "text": system_content,
                "name": "system"
            })
        elif role == 'user':
            data.append({
                "role": "user",
                "text": content,
                "name": "user"
            })
        elif role == 'assistant':
            # assistant -> model (model 的角色名)
            data.append({
                "role": "assistant",
                "text": content,
                "name": "assistant"
            })
    
    # 添加空的 ai 回复让模型续写
    data.append({
        "role": "ai",
        "text": "",
        "name": "ai"
    })
    
    vulcan_item = {
        "trace_id": f"sys_gen_{trace_id}",
        "data": data,
        "model_control": None,
        "follow_system": True,
        "train_start_index": -1,
        "need_valid": True,
        "raw_record": sample
    }
    
    return vulcan_item


# 保留旧方法用于对比（可选）
def construct_vulcan_item_flat(sample: Dict) -> Dict:
    """
    [旧方法] 扁平化格式：所有内容拼成一个 user message
    """
    trace_id = sample.get('trace_id', '')
    chat_history = sample.get('chat_history', '')
    ai_name = sample.get('ai_name', '')
    
    requirements = REQUIREMENTS_TEMPLATE.format(ai_name=ai_name)
    user_prompt = f"{chat_history}\n\n{requirements}"
    
    vulcan_item = {
        "trace_id": f"sys_gen_{trace_id}",
        "data": [
            {"role": "user", "text": user_prompt, "name": "user"},
            {"role": "ai", "text": "", "name": "ai"}
        ],
        "model_control": None,
        "follow_system": True,
        "train_start_index": -1,
        "need_valid": True,
        "raw_record": sample
    }
    
    return vulcan_item


def construct_vulcan_item(sample: Dict, use_multiturn: bool = True) -> Dict:
    """
    构建 Vulcan 推理项
    
    Args:
        sample: 输入样本
        use_multiturn: 是否使用多轮对话格式（推荐 True）
    """
    if use_multiturn and 'context_before' in sample:
        return construct_vulcan_item_multiturn(sample)
    else:
        return construct_vulcan_item_flat(sample)


def process_file(input_path: str, output_dir: str, num_parts: int = 6, test_samples: int = 1000):
    """
    处理文件，生成 Vulcan 格式数据
    """
    print(f"输入: {input_path}")
    print(f"输出目录: {output_dir}")
    
    # 读取数据
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取数据"):
            try:
                data.append(json.loads(line))
            except:
                continue
    
    print(f"共读取 {len(data)} 条数据")
    
    # 转换为 Vulcan 格式
    vulcan_data = []
    for sample in tqdm(data, desc="转换格式"):
        vulcan_item = construct_vulcan_item(sample)
        vulcan_data.append(vulcan_item)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整数据
    full_output = os.path.join(output_dir, 'sys_gen_full.jsonl')
    with open(full_output, 'w', encoding='utf-8') as f:
        for item in vulcan_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"完整数据: {full_output} ({len(vulcan_data)} 条)")
    
    # 分片保存
    part_size = len(vulcan_data) // num_parts + 1
    for i in range(num_parts):
        start = i * part_size
        end = min((i + 1) * part_size, len(vulcan_data))
        part_data = vulcan_data[start:end]
        
        if part_data:
            part_output = os.path.join(output_dir, f'sys_gen_part{i}.jsonl')
            with open(part_output, 'w', encoding='utf-8') as f:
                for item in part_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"分片 {i}: {part_output} ({len(part_data)} 条)")
    
    # 保存测试集
    test_output = os.path.join(output_dir, f'sys_gen_test_{test_samples}.jsonl')
    test_data = vulcan_data[:test_samples]
    with open(test_output, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"测试集: {test_output} ({len(test_data)} 条)")
    
    return len(vulcan_data)


def main():
    parser = argparse.ArgumentParser(description='Step 3.2: 构建 Vulcan 推理数据（主方法）')
    parser.add_argument('--input', '-i', type=str,
                        default='/path/to/data/example',
                        help='输入文件')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='/path/to/data/example',
                        help='输出目录')
    parser.add_argument('--num_parts', type=int, default=6,
                        help='分片数量')
    parser.add_argument('--test_samples', type=int, default=1000,
                        help='测试集样本数')
    
    args = parser.parse_args()
    
    process_file(args.input, args.output_dir, args.num_parts, args.test_samples)


if __name__ == '__main__':
    main()

