#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 full_final_cleaned 数据转换为 SFT 训练格式

主要功能：
1. training_samples 现在是真正的训练数据（每个角色一条），包含 ShareGPT 格式的对话
2. 字段从 training_samples 中移除，放入 character_datasets
3. dialogues 新增字段: system, think, without_think, origin_id
4. 每个角色只能看到自己的 think，看不到别人的 think
5. 参考 CoSER 评估代码的 system prompt 构建方式

输入: full_final_cleaned/*.json (CoSER 原始格式)
输出: sft_data_full.jsonl (SFT 训练格式)
"""

import json
import os
import glob
import re
from tqdm import tqdm
from typing import Dict, List, Optional


# ==================== 配置 ====================
INPUT_DIR = "/path/to/data/example"
OUTPUT_FILE = "/path/to/data/example"


# ==================== 工具函数 ====================

def extract_book_name(filename: str) -> str:
    """从文件名提取书名"""
    return os.path.basename(filename).replace('.json', '')


def convert_to_standard_format(message: str) -> str:
    """
    将原始格式转换为标准的 role_thinking/role_action 格式
    
    转换规则:
    - [内心想法] → <role_thinking>内心想法</role_thinking>
    - (动作) → <role_action>动作</role_action>
    - 所有标签之间、标签和台词之间都无空格：</role_thinking><role_action>动作</role_action>台词
    
    Returns:
        转换后的标准格式内容
    """
    result = message
    
    # 转换 [...] 为 <role_thinking>...</role_thinking>
    result = re.sub(r'\[([^\]]+)\]', r'<role_thinking>\1</role_thinking>', result)
    
    # 转换 (...) 为 <role_action>...</role_action>
    # 注意：需要排除空括号和嵌套情况
    result = re.sub(r'\(([^()]+)\)', r'<role_action>\1</role_action>', result)
    
    # 去掉所有标签后面的空格（标签之间、标签和台词之间都无空格）
    # </role_thinking> <role_action> → </role_thinking><role_action>
    # </role_action> <role_thinking> → </role_action><role_thinking>
    # </role_action> Hello → </role_action>Hello
    # </role_thinking> Hello → </role_thinking>Hello
    result = re.sub(r'</role_thinking>\s+', r'</role_thinking>', result)
    result = re.sub(r'</role_action>\s+', r'</role_action>', result)
    
    return result


def remove_inner_thoughts(message: str) -> str:
    """
    从消息中移除内心想法，返回 without_think 内容（用于给其他角色看）
    
    支持格式:
    - [内心想法] (CoSER 原生格式)
    - <think>内心想法</think>
    - <role_thinking>内心想法</role_thinking>
    
    注意：保留动作内容，只移除思考内容
    
    Returns:
        without_think: 去除所有think标记后的内容
    """
    without_think = message
    
    # 移除 [...] 格式的内心想法
    without_think = re.sub(r'\[[^\]]+\]', '', without_think)
    
    # 移除 <think>...</think> 格式
    without_think = re.sub(r'<think>.*?</think>', '', without_think, flags=re.DOTALL)
    
    # 移除 <role_thinking>...</role_thinking> 格式
    without_think = re.sub(r'<role_thinking>.*?</role_thinking>', '', without_think, flags=re.DOTALL)
    
    # 清理多余空白，但保留基本格式
    without_think = re.sub(r' +', ' ', without_think)  # 多个空格合并
    without_think = re.sub(r'\n\s*\n', '\n\n', without_think)  # 多个空行合并
    without_think = without_think.strip()
    
    return without_think


def remove_thoughts_and_convert_actions(message: str) -> str:
    """
    移除思考内容，并将动作转换为标准格式（用于 user 消息中其他角色的发言）
    
    转换规则:
    - [内心想法] → 移除
    - (动作) → <role_action>动作</role_action>
    
    Returns:
        处理后的内容
    """
    result = message
    
    # 移除 [...] 格式的内心想法
    result = re.sub(r'\[[^\]]+\]', '', result)
    
    # 转换 (...) 为 <role_action>...</role_action>
    result = re.sub(r'\(([^()]+)\)', r'<role_action>\1</role_action>', result)
    
    # 清理多余空白
    result = re.sub(r' +', ' ', result)
    result = re.sub(r'\n\s*\n', '\n\n', result)
    result = result.strip()
    
    return result


def get_speaking_characters(dialogues: List[Dict]) -> List[str]:
    """从对话中提取所有说话的角色"""
    characters = []
    for d in dialogues:
        char = d.get('character', '')
        if char and char not in characters:
            characters.append(char)
    return characters


def get_major_characters_from_key_characters(key_characters: List[Dict]) -> List[str]:
    """从 key_characters 中提取主要角色名"""
    return [kc.get('name', '') for kc in key_characters if kc.get('name')]


def build_character_motivation_map(conv_key_characters: List[Dict]) -> Dict[str, str]:
    """构建角色名到motivation的映射"""
    return {
        kc.get('name', ''): kc.get('motivation', '')
        for kc in conv_key_characters
        if kc.get('name')
    }


def build_character_experience_map(plot_key_characters: List[Dict]) -> Dict[str, str]:
    """构建角色名到experience的映射"""
    return {
        kc.get('name', ''): kc.get('experience', '')
        for kc in plot_key_characters
        if kc.get('name')
    }


def build_character_description_map(plot_key_characters: List[Dict]) -> Dict[str, str]:
    """构建角色名到description的映射"""
    return {
        kc.get('name', ''): kc.get('description', '')
        for kc in plot_key_characters
        if kc.get('name')
    }


def build_character_info(
    book_name: str,
    character_datasets: Dict,
    speaking_characters: List[str],
    conv_key_characters: List[Dict],
    plot_key_characters: List[Dict],
    scenario: str,
    summary: str,
    topic: str
) -> Dict[str, Dict]:
    """
    为每个说话角色构建角色信息（用于 character_datasets）
    
    Returns:
        Dict[character_name, {
            'book_name': str,
            'character': str,
            'character_profile': str,
            'background': str,
            'scenario': str,
            'topic': str,
            'motivation': str,
            'experience': str,
            'description': str,
            'other_character_profiles': Dict[str, str]
        }]
    """
    motivation_map = build_character_motivation_map(conv_key_characters)
    experience_map = build_character_experience_map(plot_key_characters)
    description_map = build_character_description_map(plot_key_characters)
    
    character_info = {}
    
    for char in speaking_characters:
        if char == 'Environment':
            continue
        
        # 获取角色 profile
        char_profile = ''
        if char in character_datasets:
            char_profile = character_datasets[char].get('profile', '')
        if not char_profile and char in description_map:
            char_profile = description_map[char]
        
        # 获取其他角色的 profiles
        other_profiles = {}
        for other_char in speaking_characters:
            if other_char != char and other_char != 'Environment':
                if other_char in character_datasets:
                    other_profiles[other_char] = character_datasets[other_char].get('profile', '')
                elif other_char in description_map:
                    other_profiles[other_char] = description_map[other_char]
        
        character_info[char] = {
            'book_name': book_name,
            'character': char,
            'character_profile': char_profile,
            'background': summary,
            'scenario': scenario,
            'topic': topic,
            'motivation': motivation_map.get(char, ''),
            'experience': experience_map.get(char, ''),
            'description': description_map.get(char, ''),
            'other_character_profiles': other_profiles,
            'output_format': get_output_format()  # 单独存储输出格式要求
        }
    
    return character_info


def get_output_format() -> str:
    """
    获取输出格式要求（单独字段）
    
    使用 <role_thinking> 和 <role_action> 格式 (HER/云端API格式)
    """
    output_format = """Your output should follow this two-part structure in strict order:
1. System Thinking: A single block at the very beginning, wrapped in <system_thinking> and </system_thinking>. This is the third-person analysis of how to portray the role.
2. Role-play Response: The response include the role's thought, speech and action. Use <role_thinking> your thought </role_thinking> for thoughts (invisible to others) and <role_action> your action </role_action> for actions (visible to others). These three elements (thought, speech and action) can appear multiple times and be freely interleaved.
(all thinking is invisible to others)"""
    return output_format


def get_character_prompt(
    book_name: str,
    character: str,
    character_profile: str,
    background: str,
    scenario: str,
    motivation: str = '',
    other_character_profiles: Dict[str, str] = None,
    include_requirements: bool = True
) -> str:
    """
    构建角色扮演的 system prompt（参考 CoSER 评估代码）
    
    Args:
        include_requirements: 是否包含 Requirements 部分，默认 True
                              如果为 False，则 Requirements 需要单独存储
    """
    # 构建其他角色信息
    other_char_str = ''
    if other_character_profiles:
        for other_char, profile in other_character_profiles.items():
            if other_char != character and profile:
                other_char_str += f"\n{other_char}: {profile}\n"
    
    # 构建 system prompt
    system_prompt = f"You are {character} from {book_name}.\n\n"
    system_prompt += f"==={character}'s Profile===\n{character_profile}\n\n"
    system_prompt += f"===Current Scenario===\n{scenario}\n\n"
    
    if other_char_str:
        system_prompt += f"===Information about the other Characters===\n{other_char_str}\n\n"
    
    if motivation:
        system_prompt += f"===Your Inner Thoughts===\n{motivation}\n\n"
    
    if include_requirements:
        system_prompt += f"===Requirements===\n{get_output_format()}\n\n"
    
    return system_prompt


def build_training_samples_sharegpt(
    dialogues: List[Dict],
    character_info: Dict[str, Dict],
    trace_id: str
) -> Dict[str, List[Dict]]:
    """
    为每个角色构建 ShareGPT 格式的训练数据
    
    每个角色只能看到自己的 think，看不到别人的 think
    使用 ===Conversation Start=== 标记对话开始
    
    注意：dialogues 已经在 enrich_dialogues 中合并过连续同角色发言
    每个 dialogue 的 origin_id 是一个列表（包含被合并的原始索引）
    
    每条消息包含 origin_id 字段，记录对应的原始对话索引，方便拼接回去
    
    Returns:
        Dict[character_name, List[ShareGPT messages]]
        每条消息格式: {"role": str, "content": str, "origin_id": List[int] or None}
    """
    training_samples = {}
    
    # 获取所有说话的角色（排除 Environment）
    speaking_chars = [char for char in get_speaking_characters(dialogues) if char != 'Environment']
    
    for target_char in speaking_chars:
        if target_char not in character_info:
            continue
        
        info = character_info[target_char]
        
        # 构建 system prompt
        system_prompt = get_character_prompt(
            book_name=info['book_name'],
            character=info['character'],
            character_profile=info['character_profile'],
            background=info['background'],
            scenario=info['scenario'],
            motivation=info['motivation'],
            other_character_profiles=info['other_character_profiles']
        )
        
        # 构建对话历史
        messages = [
            {"role": "system", "content": system_prompt, "origin_id": None}
        ]
        
        # 添加对话开始标记
        user_buffer = "===Conversation Start===\n\n"
        user_origin_ids = []  # 记录 user 消息对应的原始对话索引
        assistant_buffer = ""
        assistant_origin_ids = []  # 记录 assistant 消息对应的原始对话索引
        
        # 遍历对话（已合并过连续同角色发言）
        for dialogue in dialogues:
            char = dialogue.get('character', '')
            # dialogue['origin_id'] 现在是列表（合并后的原始索引）
            dialogue_origin_ids = dialogue.get('origin_id', [])
            
            if char == target_char:
                # 目标角色发言
                # 先 flush user_buffer（如果有实质内容）
                if user_buffer.strip() and user_buffer != "===Conversation Start===\n\n":
                    messages.append({
                        "role": "user", 
                        "content": user_buffer.strip(),
                        "origin_id": user_origin_ids.copy()
                    })
                    user_buffer = ""
                    user_origin_ids = []
                
                # 使用 standard_format（已在 enrich_dialogues 中转换）
                converted_message = dialogue.get('standard_format', dialogue.get('message', ''))
                
                # 目标角色的回复直接作为一个 assistant 消息
                # （因为 enrich_dialogues 已经合并过了）
                messages.append({
                    "role": "assistant", 
                    "content": f"{char}: {converted_message}",
                    "origin_id": dialogue_origin_ids  # 直接使用列表
                })
            else:
                # 其他角色发言：使用 without_think（移除思考的内容）
                other_message = dialogue.get('without_think', dialogue.get('message', ''))
                user_buffer += f"{char}: {other_message}\n\n"
                user_origin_ids.extend(dialogue_origin_ids)  # 扩展索引列表
        
        # 处理结尾：flush 剩余的 user_buffer（如果有）
        if user_buffer.strip() and user_buffer != "===Conversation Start===\n\n":
            # 检查最后一条消息是否不是 user
            if messages and messages[-1]["role"] != "user":
                messages.append({
                    "role": "user", 
                    "content": user_buffer.strip(),
                    "origin_id": user_origin_ids.copy()
                })
        
        training_samples[target_char] = messages
    
    return training_samples


def enrich_dialogues(dialogues: List[Dict]) -> List[Dict]:
    """
    为 dialogues 添加新字段，并合并连续的同一角色多句话
    
    处理流程：
    1. 合并连续的同一角色多句话
    2. 为每条消息添加：
       - origin_id: 原始对话索引列表（合并后可能包含多个索引）
    - standard_format: 转换后的标准格式 (role_thinking/role_action)
    - without_think: 移除思考并转换动作格式的内容（给别的角色看）
    
    Returns:
        enriched dialogues list (已合并连续发言)
    """
    if not dialogues:
        return []
    
    # 第一步：合并连续的同一角色多句话
    merged = []
    current_char = None
    current_messages = []
    current_origin_ids = []
    
    for i, dialogue in enumerate(dialogues):
        char = dialogue.get('character', '')
        message = dialogue.get('message', '')
        
        if char == current_char:
            # 同一角色连续发言，追加
            current_messages.append(message)
            current_origin_ids.append(i)
        else:
            # 角色切换，保存之前的合并结果
            if current_char is not None and current_messages:
                merged.append({
                    'character': current_char,
                    'message': '\n\n'.join(current_messages),  # 用双换行合并
                    'origin_id': current_origin_ids.copy()
                })
            # 开始新角色
            current_char = char
            current_messages = [message]
            current_origin_ids = [i]
    
    # 处理最后一组
    if current_char is not None and current_messages:
        merged.append({
            'character': current_char,
            'message': '\n\n'.join(current_messages),
            'origin_id': current_origin_ids.copy()
        })
    
    # 第二步：为合并后的对话添加其他字段
    enriched = []
    for dialogue in merged:
        original_message = dialogue.get('message', '')
        # 转换为标准格式
        standard_format = convert_to_standard_format(original_message)
        # 移除思考，保留动作（给其他角色看）
        without_think = remove_thoughts_and_convert_actions(original_message)
        
        enriched_dialogue = {
            'character': dialogue['character'],
            'message': original_message,
            'origin_id': dialogue['origin_id'],  # 原始索引列表
            'standard_format': standard_format,
            'without_think': without_think
        }
        enriched.append(enriched_dialogue)
    
    return enriched


def process_single_book(filepath: str, character_datasets: Dict) -> List[Dict]:
    """
    处理单本书的数据
    
    Returns:
        List of training samples (each sample = one conversation)
    """
    book_name = extract_book_name(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plots = data.get('plots', [])
    samples = []
    
    for plot_idx, plot in enumerate(plots):
        text = plot.get('text', '')
        summary = plot.get('summary', '')
        prominence = plot.get('prominence', 0)
        chapter = plot.get('chapter', '')
        plot_key_characters = plot.get('key_characters', [])
        state = plot.get('state', '')
        i_chunk = plot.get('i_chunk', 0)
        i_p = plot.get('i_p', 0)
        
        conversations = plot.get('conversation', [])
        
        for conv_idx, conv in enumerate(conversations):
            scenario = conv.get('scenario', '')
            topic = conv.get('topic', '')
            conv_key_characters = conv.get('key_characters', [])
            dialogues = conv.get('dialogues', [])
            i_c = conv.get('i_c', 0)
            
            # 跳过空对话
            if not dialogues:
                continue
            
            # 获取说话角色
            speaking_characters = get_speaking_characters(dialogues)
            major_characters = get_major_characters_from_key_characters(conv_key_characters)
            
            # 构建角色信息（放入 character_datasets）
            character_info = build_character_info(
                book_name=book_name,
                character_datasets=character_datasets,
                speaking_characters=speaking_characters,
                conv_key_characters=conv_key_characters,
                plot_key_characters=plot_key_characters,
                scenario=scenario,
                summary=summary,
                topic=topic
            )
            
            # 构建唯一标识
            trace_id = f"{book_name}_{chapter}_{plot_idx}_{conv_idx}"
            
            # 丰富 dialogues（添加 system, think, without_think, origin_id）
            enriched_dialogues = enrich_dialogues(dialogues)
            
            # 构建 ShareGPT 格式的训练样本
            training_samples = build_training_samples_sharegpt(
                dialogues=enriched_dialogues,
                character_info=character_info,
                trace_id=trace_id
            )
            
            # 添加 Environment 如果存在
            speaking_characters_w_env = speaking_characters.copy()
            if 'Environment' in [d.get('character') for d in dialogues]:
                if 'Environment' not in speaking_characters_w_env:
                    speaking_characters_w_env.append('Environment')
            
            sample = {
                # === 原始 plot 字段 ===
                'text': text,
                'summary': summary,
                'prominence': prominence,
                'key_characters': plot_key_characters,
                'chapter': chapter,
                'state': state,
                'i_chunk': i_chunk,
                'i_p': i_p,
                
                # === conversation 字段（更新后） ===
                'conversation': [{
                    'scenario': scenario,
                    'topic': topic,
                    'key_characters': conv_key_characters,
                    'i_c': i_c,
                    'speaking_characters_w_env': speaking_characters_w_env,
                    'major_characters': major_characters,
                    'dialogues': enriched_dialogues  # 使用丰富后的 dialogues
                }],
                
                # === 新增字段 ===
                'character_datasets': character_info,  # 角色信息（原 training_samples 的内容）
                'training_samples': training_samples,  # ShareGPT 格式的训练数据
                'trace_id_book_chapter': trace_id,
                'book_name': book_name,
                'plot_index': plot_idx,
                'conv_index': conv_idx
            }
            
            samples.append(sample)
    
    return samples


def main():
    """主函数"""
    print("=" * 60)
    print("full_final_cleaned → SFT Format 转换工具")
    print("=" * 60)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出文件: {OUTPUT_FILE}")
    print()
    
    # 获取所有 JSON 文件
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    print(f"找到 {len(json_files)} 个文件")
    
    all_samples = []
    total_dialogues = 0
    total_training_samples = 0
    books_with_no_profile = []
    
    # 第一步：收集所有样本
    for filepath in tqdm(json_files, desc="处理书籍"):
        try:
            # 读取 character_datasets
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            character_datasets = data.get('character_datasets', {})
            
            # 检查 character_datasets 是否为空
            if not character_datasets:
                books_with_no_profile.append(extract_book_name(filepath))
            
            # 处理单本书
            samples = process_single_book(filepath, character_datasets)
            all_samples.extend(samples)
            
            for sample in samples:
                total_dialogues += len(sample['conversation'][0]['dialogues'])
                total_training_samples += len(sample['training_samples'])
                
        except Exception as e:
            print(f"\n处理文件出错 {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    # 第二步：按 trace_id_book_chapter 升序排序
    print(f"\n排序中... (共 {len(all_samples)} 个样本)")
    all_samples.sort(key=lambda x: x['trace_id_book_chapter'], reverse=False)
    
    # 第三步：写入输出文件
    print(f"写入文件...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for sample in all_samples:
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print()
    print("=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"总书籍数: {len(json_files)}")
    print(f"总样本数 (conversations): {len(all_samples)}")
    print(f"总对话轮次: {total_dialogues}")
    print(f"总训练样本 (每角色一条): {total_training_samples}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"排序方式: trace_id_book_chapter 升序")
    
    if books_with_no_profile:
        print(f"\n⚠️ 以下书籍没有 character_datasets:")
        for b in books_with_no_profile[:10]:
            print(f"  - {b}")
        if len(books_with_no_profile) > 10:
            print(f"  ... 共 {len(books_with_no_profile)} 本")


if __name__ == '__main__':
    main()

