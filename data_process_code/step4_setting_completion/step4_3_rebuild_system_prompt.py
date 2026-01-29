#!/usr/bin/env python3
"""
Step 4.3: 用增强后的字段重新拼接 system prompt

使用 character_datasets 中的 xxx_enriched 字段拼接 training_samples 中的 system prompt
如果 xxx_enriched 不存在，则使用原始字段
"""

import json
import os
from tqdm import tqdm

# 路径配置
INPUT_PATH = "/path/to/data/example"
OUTPUT_PATH = "/path/to/data/example"


def get_field(char_data, field_name, use_enriched=True):
    """获取字段值，优先使用增强后的版本"""
    if use_enriched:
        enriched_key = f'{field_name}_enriched'
        if enriched_key in char_data and char_data[enriched_key]:
            return char_data[enriched_key]
    return char_data.get(field_name, '')


def build_system_prompt(char_name, char_data, book_name, use_enriched=True):
    """
    根据 character_datasets 构建完整的 system prompt
    优先使用 xxx_enriched 字段
    """
    # 获取字段（优先增强版本）
    character_profile = get_field(char_data, 'character_profile', use_enriched)
    background = get_field(char_data, 'background', use_enriched)
    scenario = get_field(char_data, 'scenario', use_enriched)
    motivation = get_field(char_data, 'motivation', use_enriched)
    description = get_field(char_data, 'description', use_enriched)
    experience = get_field(char_data, 'experience', use_enriched)
    other_profiles = get_field(char_data, 'other_character_profiles', use_enriched)
    output_format = char_data.get('output_format', '')  # 这个不需要增强
    
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


def rebuild_training_samples(sample, use_enriched=True):
    """用增强后的字段重建 training_samples"""
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
                new_prompt = build_system_prompt(char_name, char_data, book_name, use_enriched)
                s['content'] = new_prompt
                break
    
    return sample


def main():
    print("=" * 60)
    print("Step 4.3: 用增强后的字段重新拼接 system prompt")
    print("=" * 60)
    
    total = 0
    enriched_count = 0
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="处理中"):
            sample = json.loads(line)
            total += 1
            
            # 检查是否有增强字段
            has_enriched = False
            for char_data in sample.get('character_datasets', {}).values():
                if 'character_profile_enriched' in char_data:
                    has_enriched = True
                    break
            
            if has_enriched:
                enriched_count += 1
            
            # 重建 system prompt
            rebuilt_sample = rebuild_training_samples(sample, use_enriched=True)
            f_out.write(json.dumps(rebuilt_sample, ensure_ascii=False) + '\n')
    
    print()
    print("=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"总样本数: {total}")
    print(f"使用增强字段: {enriched_count}")
    print(f"使用原始字段: {total - enriched_count}")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

