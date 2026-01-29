#!/usr/bin/env python3
"""
Step 4.0: 修复 system prompt 拼接

当前问题: training_samples 中的 system prompt 只用了 description 和 experience
需要修复: 使用 character_datasets 中的完整字段拼接 system prompt

参考: utils.py get_character_prompt 函数
"""

import json
import os
from tqdm import tqdm

# 路径配置
INPUT_PATH = "/path/to/data/example"
OUTPUT_PATH = "/path/to/data/example"


def build_system_prompt(char_name, char_data, book_name):
    """
    根据 character_datasets 构建完整的 system prompt
    参考 utils.py get_character_prompt (fixed_template=True 模式)
    """
    character_profile = char_data.get('character_profile', '')
    background = char_data.get('background', '')
    scenario = char_data.get('scenario', '')
    motivation = char_data.get('motivation', '')
    description = char_data.get('description', '')
    experience = char_data.get('experience', '')
    other_profiles = char_data.get('other_character_profiles', {})
    output_format = char_data.get('output_format', '')
    
    # 构建 profile 部分
    profile_parts = []
    if description:
        profile_parts.append(f"description: {description}")
    if character_profile:
        profile_parts.append(character_profile)
    if experience:
        profile_parts.append(f"experience: {experience}")
    
    profile_text = '\n'.join(profile_parts) if profile_parts else ''
    
    # 构建其他角色信息
    other_chars_text = ''
    if other_profiles and isinstance(other_profiles, dict):
        other_parts = []
        for other_name, other_profile in other_profiles.items():
            if other_name != char_name:
                other_parts.append(f"**{other_name}**: {other_profile}")
        if other_parts:
            other_chars_text = '\n\n'.join(other_parts)
    
    # 拼接完整 system prompt
    prompt_parts = [f"You are {char_name} from {book_name}."]
    
    # Profile
    if profile_text:
        prompt_parts.append(f"\n\n==={char_name}'s Profile===\n{profile_text}")
    
    # Background (Plot Summary)
    if background:
        prompt_parts.append(f"\n\n===Background===\n{background}")
    
    # Current Scenario
    if scenario:
        prompt_parts.append(f"\n\n===Current Scenario===\n{scenario}")
    
    # Other Characters
    if other_chars_text:
        prompt_parts.append(f"\n\n===Information about other Characters===\n{other_chars_text}")
    
    # Inner Thoughts / Motivation
    if motivation:
        prompt_parts.append(f"\n\n===Your Inner Thoughts===\n{motivation}")
    
    # Output Format / Requirements
    if output_format:
        prompt_parts.append(f"\n\n===Requirements===\n{output_format}")
    
    return ''.join(prompt_parts)


def fix_training_samples(sample):
    """修复一个样本的 training_samples"""
    character_datasets = sample.get('character_datasets', {})
    training_samples = sample.get('training_samples', {})
    book_name = sample.get('book_name', '')
    
    for char_name, samples in training_samples.items():
        # 获取 character_datasets 中的数据
        char_data = character_datasets.get(char_name, {})
        
        if not char_data:
            continue
        
        # 找到 system role 并替换
        for s in samples:
            if s.get('role') == 'system':
                new_prompt = build_system_prompt(char_name, char_data, book_name)
                s['content'] = new_prompt
                break
    
    return sample


def main():
    print("=" * 60)
    print("Step 4.0: 修复 system prompt 拼接")
    print("=" * 60)
    
    total = 0
    fixed = 0
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="处理中"):
            sample = json.loads(line)
            total += 1
            
            # 修复
            fixed_sample = fix_training_samples(sample)
            f_out.write(json.dumps(fixed_sample, ensure_ascii=False) + '\n')
            fixed += 1
    
    print()
    print("=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"处理: {total} 条")
    print(f"输出: {OUTPUT_PATH}")
    
    # 显示修复后的示例
    print("\n=== 修复后的 system prompt 示例 ===")
    with open(OUTPUT_PATH, 'r') as f:
        d = json.loads(f.readline())
        ts = d.get('training_samples', {})
        for char_name, samples in ts.items():
            for s in samples:
                if s.get('role') == 'system':
                    content = s.get('content', '')
                    print(f"\n【{char_name}】({len(content)} 字符)")
                    print(content[:1500] + "..." if len(content) > 1500 else content)
                    break
            break


if __name__ == "__main__":
    main()

