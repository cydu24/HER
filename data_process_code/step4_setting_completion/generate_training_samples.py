#!/usr/bin/env python3
"""
生成 training_samples - 使用 enriched 字段

输入: HER-dataset.jsonl (不含 training_samples)
输出: HER-dataset-with-training.jsonl (含 training_samples)

使用的 enriched 字段:
- character_profile_enriched
- background_enriched  
- motivation_enriched
- description_enriched
- experience_enriched
- scenario_enriched
"""

import json
import re
import random
from typing import Dict, List, Any
from tqdm import tqdm

INPUT_FILE = "/path/to/project/data_process/step4_setting_completion/sft_data_final_v7_patched.jsonl"
OUTPUT_FILE = "/path/to/project/data_process/step4_setting_completion/HER-dataset-with-training.jsonl"


def get_character_prompt_enriched(
    book_name: str,
    character: str,
    character_profile: str,
    background: str,
    scenario: str,
    motivation: str,
    description: str = "",
    experience: str = "",
    other_character_profiles: Dict[str, str] = None,
    model_type: str = 'her'
) -> str:
    """
    生成使用 enriched 字段的 System Prompt
    
    Args:
        book_name: 书名
        character: 角色名
        character_profile: 角色档案 (character_profile_enriched)
        background: 背景 (background_enriched)
        scenario: 场景 (scenario_enriched)
        motivation: 动机 (motivation_enriched)
        description: 简短描述 (description_enriched)
        experience: 经历 (experience_enriched)
        other_character_profiles: 其他角色档案
        model_type: 模型类型 (her/m2/coser)
    """
    # 根据模型类型选择输出格式
    if model_type == 'coser':
        output_format = """
Your output should include **thought**, **speech**, and **action**. 
Use [your thought] for thoughts, which others can't see. 
Use (your action) for actions, which others can see.
(all thinking is invisible to others)
"""
    elif model_type == 'm2':
        output_format = """
Your output should include **speech** and **action**. 
Use *your action* to describe actions (visible to others). 
Respond directly as the character - no meta-commentary or instructions.

Example output format:
*looks up from the book with a slight smile* Good morning! I was just thinking about you. *sets the book aside*

Now respond in character:
"""
    else:  # her (默认) - 与评测代码保持一致
        output_format = """
=Requirements=
Your output should follow this two-part structure in strict order:
1. System Thinking: A single block at the very beginning, wrapped in <system_thinking> and </system_thinking>. This is the third-person analysis of how to portray the role.
2. Role-play Response: The response include the role's thought, speech and action. Use <role_thinking> your thought </role_thinking> and <role_action> your action</role_action> as needed. The role's thought should be invisible to others.
"""

    # 构建其他角色信息
    other_chars_str = ""
    if other_character_profiles:
        others = []
        for other_char, profile in other_character_profiles.items():
            if other_char != character and profile:
                others.append(f"**{other_char}**: {profile}")
        if others:
            other_chars_str = "\n".join(others)

    # 随机选择模板风格
    style = random.choice(['natural', 'structured'])
    
    if style == 'natural':
        # 自然流畅风格
        prompt = f"You are {character}"
        if book_name:
            prompt += f" from {book_name}"
        prompt += ".\n\n"
        
        # 简短描述
        if description:
            prompt += f"{description}\n\n"
        
        # 角色档案
        prompt += f"**Character Profile**\n{character_profile}\n\n"
        
        # 经历（如果有）
        if experience:
            prompt += f"**Recent Experience**\n{experience}\n\n"
        
        # 背景（50% 概率包含）
        if background and random.random() < 0.5:
            prompt += f"**Background**\n{background}\n\n"
        
        # 当前场景
        prompt += f"**Current Scenario**\n{scenario}\n\n"
        
        # 动机
        if motivation:
            prompt += f"**Your Inner State**\n{motivation}\n\n"
        
        # 其他角色
        if other_chars_str:
            prompt += f"**Other Characters**\n{other_chars_str}\n\n"
        
        # 输出格式要求
        prompt += f"**Response Format**{output_format}"
        
    else:
        # 结构化风格
        decorators = random.choice([
            ("===", "==="),
            ("##", ""),
            ("**", "**"),
        ])
        d1, d2 = decorators
        
        prompt = f"You are roleplaying as {character}.\n\n"
        
        if description:
            prompt += f"{description}\n\n"
        
        prompt += f"{d1} Character Profile {d2}\n{character_profile}\n\n"
        
        if experience:
            prompt += f"{d1} Recent Experience {d2}\n{experience}\n\n"
        
        if background and random.random() < 0.5:
            prompt += f"{d1} Background {d2}\n{background}\n\n"
        
        prompt += f"{d1} Current Scenario {d2}\n{scenario}\n\n"
        
        if motivation:
            prompt += f"{d1} Your Inner State {d2}\n{motivation}\n\n"
        
        if other_chars_str:
            prompt += f"{d1} Other Characters {d2}\n{other_chars_str}\n\n"
        
        prompt += f"{d1} Response Format {d2}{output_format}"
    
    return prompt


def remove_role_thinking(content: str) -> str:
    """移除 <role_thinking> 标签（其他角色看不到）"""
    content = re.sub(r'<role_thinking>.*?</role_thinking>', '', content, flags=re.DOTALL)
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    return content.strip()


def generate_training_samples(sample: Dict[str, Any], model_type: str = 'her') -> Dict[str, Any]:
    """
    为样本生成 training_samples
    """
    character_datasets = sample.get('character_datasets', {})
    conversations = sample.get('conversation', [])
    
    if not conversations:
        return sample
    
    all_training_samples = {}  # {角色名: [messages]}
    
    for conv in conversations:
        dialogues = conv.get('dialogues', [])
        
        # 优先使用 enriched scenario
        scenario = conv.get('scenario_enriched') or conv.get('scenario', '')
        
        if not dialogues:
            continue
        
        # 收集所有角色
        all_characters = set()
        for dlg in dialogues:
            char = dlg.get('character', '')
            if char and char.lower() not in ['environment', 'env']:
                all_characters.add(char)
        
        # 为每个角色生成 training_samples
        for target_character in all_characters:
            if target_character not in character_datasets:
                continue
            
            char_data = character_datasets[target_character]
            book_name = char_data.get('book_name', '')
            
            # 使用 enriched 字段（优先）或原始字段
            character_profile = char_data.get('character_profile_enriched') or char_data.get('character_profile', '')
            background = char_data.get('background_enriched') or char_data.get('background', '')
            motivation = char_data.get('motivation_enriched') or char_data.get('motivation', '')
            description = char_data.get('description_enriched') or char_data.get('description', '')
            experience = char_data.get('experience_enriched') or char_data.get('experience', '')
            
            # 获取其他角色档案
            other_profiles = {}
            for other_char in all_characters:
                if other_char != target_character and other_char in character_datasets:
                    other_data = character_datasets[other_char]
                    other_profile = other_data.get('description_enriched') or other_data.get('character_profile_enriched') or other_data.get('character_profile', '')
                    if other_profile:
                        other_profiles[other_char] = other_profile
            
            # 生成 system prompt
            system_prompt = get_character_prompt_enriched(
                book_name=book_name,
                character=target_character,
                character_profile=character_profile,
                background=background,
                scenario=scenario,
                motivation=motivation,
                description=description,
                experience=experience,
                other_character_profiles=other_profiles,
                model_type=model_type
            )
            
            # 构建对话 messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # 找到该角色的所有回复位置
            user_buffer = []
            
            for i, dlg in enumerate(dialogues):
                char = dlg.get('character', '')
                # 优先使用 enhanced_standard_format
                content = dlg.get('enhanced_standard_format') or dlg.get('standard_format', '')
                
                if not content:
                    continue
                
                if char == target_character:
                    # 先 flush user buffer
                    if user_buffer:
                        messages.append({
                            "role": "user",
                            "content": "\n\n".join(user_buffer)
                        })
                        user_buffer = []
                    
                    # 角色回复
                    role_content = content
                    if not content.startswith(f"{char}:"):
                        role_content = f"{char}: {content}"
                    
                    # 添加 sys_thinking
                    sys_thinking = dlg.get('sys_thinking', '')
                    if sys_thinking and model_type == 'her':
                        role_content = f"<system_thinking>{sys_thinking}</system_thinking>{role_content}"
                    
                    messages.append({
                        "role": "assistant",
                        "content": role_content
                    })
                    
                elif char.lower() not in ['environment', 'env']:
                    # 其他角色 -> user
                    content_clean = remove_role_thinking(content)
                    if not content_clean.startswith(f"{char}:"):
                        content_clean = f"{char}: {content_clean}"
                    user_buffer.append(content_clean)
                else:
                    # 环境描述
                    user_buffer.append(content)
            
            # 处理最后的 user buffer
            if user_buffer:
                messages.append({
                    "role": "user",
                    "content": "\n\n".join(user_buffer)
                })
            
            # 保存
            if target_character not in all_training_samples:
                all_training_samples[target_character] = messages
            else:
                # 如果已存在，合并对话
                all_training_samples[target_character].extend(messages[1:])  # 跳过 system
    
    sample['training_samples'] = all_training_samples
    return sample


def main():
    print("=" * 60)
    print("生成 training_samples (使用 enriched 字段)")
    print("=" * 60)
    print(f"输入: {INPUT_FILE}")
    print(f"输出: {OUTPUT_FILE}")
    print()
    
    total = 0
    total_characters = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="处理中"):
            try:
                sample = json.loads(line.strip())
                
                # 生成 training_samples
                sample = generate_training_samples(sample, model_type='her')
                
                # 统计
                total += 1
                ts = sample.get('training_samples', {})
                total_characters += len(ts)
                
                fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"错误: {e}")
                continue
    
    print()
    print("=" * 60)
    print(f"完成！")
    print(f"处理样本数: {total}")
    print(f"生成角色数: {total_characters}")
    print(f"输出文件: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()

