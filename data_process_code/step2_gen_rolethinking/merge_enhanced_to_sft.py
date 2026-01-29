#!/usr/bin/env python3
"""
将增强后的 Role Thinking 结果合并回原始 SFT 数据

输入:
  - sft_data_full.jsonl: 原始 SFT 数据
  - enhanced_dialogues_en.jsonl: 增强后的对话数据

输出:
  - sft_data_enhanced.jsonl: 合并后的数据，包含 enhanced_standard_format 字段
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path


def load_enhanced_data(enhanced_path: str) -> dict:
    """
    加载增强数据，建立 trace_id -> {conv_index -> {origin_id -> enhanced_dialogue}} 的索引
    """
    enhanced_index = {}
    
    with open(enhanced_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            trace_id = data['trace_id']
            
            # 移除前缀 "role_thinking_enhance_en_" 或 "role_thinking_enhance_zh_"
            if trace_id.startswith('role_thinking_enhance_en_'):
                trace_id = trace_id[len('role_thinking_enhance_en_'):]
            elif trace_id.startswith('role_thinking_enhance_zh_'):
                trace_id = trace_id[len('role_thinking_enhance_zh_'):]
            
            # 建立 origin_id -> enhanced_dialogue 的映射
            dialogue_map = {}
            for enh in data.get('enhanced_dialogues', []):
                # origin_id 是一个列表，因为可能是合并的
                origin_ids = enh.get('origin_id', [])
                key = tuple(origin_ids)  # 用 tuple 作为 key
                dialogue_map[key] = enh
            
            enhanced_index[trace_id] = {
                'dialogue_map': dialogue_map,
                'statistics': data.get('statistics', {})
            }
    
    return enhanced_index


def rebuild_training_samples(data: dict) -> dict:
    """
    用 enhanced_standard_format 重新构建 training_samples
    """
    conversations = data.get('conversation', [])
    trace_id = data.get('trace_id_book_chapter', '')
    
    # 构建角色信息
    character_info = {}
    for char in data.get('key_characters', []):
        char_name = char.get('name', '')
        if char_name:
            character_info[char_name] = char
    
    # 收集所有对话中出现的角色
    all_characters = set()
    for conv in conversations:
        for dialogue in conv.get('dialogues', []):
            char = dialogue.get('character', '')
            if char and char != 'Environment':
                all_characters.add(char)
    
    training_samples = {}
    
    for target_char in all_characters:
        messages = []
        
        # 构建 system prompt
        if target_char in character_info:
            info = character_info[target_char]
            system_prompt = f"You are {target_char} from {trace_id.replace('_', ' ')}.\n\n"
            system_prompt += f"==={target_char}'s Profile===\n"
            for key, value in info.items():
                if key not in ['name'] and value:
                    system_prompt += f"{key}: {value}\n"
        else:
            system_prompt = f"You are {target_char}."
        
        messages.append({
            "role": "system", 
            "content": system_prompt, 
            "origin_id": None
        })
        
        # 处理对话
        user_buffer = "===Conversation Start===\n\n"
        user_origin_ids = []
        
        for conv in conversations:
            for dialogue in conv.get('dialogues', []):
                char = dialogue.get('character', '')
                dialogue_origin_ids = dialogue.get('origin_id', [])
                if not isinstance(dialogue_origin_ids, list):
                    dialogue_origin_ids = [dialogue_origin_ids]
                
                if char == target_char:
                    # 目标角色发言 - 先 flush user_buffer
                    if user_buffer.strip() and user_buffer != "===Conversation Start===\n\n":
                        messages.append({
                            "role": "user", 
                            "content": user_buffer.strip(),
                            "origin_id": user_origin_ids.copy()
                        })
                        user_buffer = ""
                        user_origin_ids = []
                    
                    # ★ 关键：优先使用 enhanced_standard_format
                    converted_message = dialogue.get('enhanced_standard_format', 
                                                     dialogue.get('standard_format', 
                                                                  dialogue.get('message', '')))
                    
                    messages.append({
                        "role": "assistant", 
                        "content": f"{char}: {converted_message}",
                        "origin_id": dialogue_origin_ids
                    })
                else:
                    # 其他角色发言：使用 without_think
                    other_message = dialogue.get('without_think', dialogue.get('message', ''))
                    user_buffer += f"{char}: {other_message}\n\n"
                    user_origin_ids.extend(dialogue_origin_ids)
        
        # flush 剩余的 user_buffer
        if user_buffer.strip() and user_buffer != "===Conversation Start===\n\n":
            if messages and messages[-1]["role"] != "user":
                messages.append({
                    "role": "user", 
                    "content": user_buffer.strip(),
                    "origin_id": user_origin_ids.copy()
                })
        
        training_samples[target_char] = messages
    
    return training_samples


def merge_data(sft_path: str, enhanced_index: dict, output_path: str):
    """
    合并原始 SFT 数据和增强数据，同时重建 training_samples
    """
    total_samples = 0
    matched_samples = 0
    total_dialogues = 0
    enhanced_dialogues = 0
    
    with open(sft_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            data = json.loads(line)
            total_samples += 1
            
            # 获取 trace_id
            trace_id_base = data.get('trace_id_book_chapter', '')
            
            # 遍历所有 conversation
            sample_matched = False
            for conv_idx, conv in enumerate(data.get('conversation', [])):
                # 构建完整的 trace_id (包含 conv_index)
                trace_id = f"{trace_id_base}"
                
                if trace_id in enhanced_index:
                    sample_matched = True
                    dialogue_map = enhanced_index[trace_id]['dialogue_map']
                    
                    # 遍历 dialogues，添加增强字段
                    for dialogue in conv.get('dialogues', []):
                        total_dialogues += 1
                        origin_id = dialogue.get('origin_id', [])
                        key = tuple(origin_id)
                        
                        if key in dialogue_map:
                            enh = dialogue_map[key]
                            dialogue['enhanced_standard_format'] = enh.get('enhanced_role_think', '')
                            dialogue['enhanced_reason'] = enh.get('enhanced_reason', '')
                            dialogue['enhanced_pattern'] = enh.get('pattern', '')
                            enhanced_dialogues += 1
                        else:
                            # 没有找到增强数据，使用原始数据
                            dialogue['enhanced_standard_format'] = dialogue.get('standard_format', '')
                            dialogue['enhanced_reason'] = ''
                            dialogue['enhanced_pattern'] = ''
            
            if sample_matched:
                matched_samples += 1
            
            # ★ 关键：重建 training_samples，使用 enhanced_standard_format
            data['training_samples'] = rebuild_training_samples(data)
            
            # 写入输出
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    return {
        'total_samples': total_samples,
        'matched_samples': matched_samples,
        'total_dialogues': total_dialogues,
        'enhanced_dialogues': enhanced_dialogues
    }


def main():
    parser = argparse.ArgumentParser(description='合并增强数据到 SFT 数据')
    parser.add_argument('--sft', type=str,
                        default='/path/to/data/example',
                        help='原始 SFT 数据路径')
    parser.add_argument('--enhanced', type=str,
                        default='/path/to/data/example',
                        help='增强数据路径')
    parser.add_argument('--output', type=str,
                        default='/path/to/data/example',
                        help='输出路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("合并增强数据到 SFT 数据")
    print("=" * 60)
    print(f"SFT 数据: {args.sft}")
    print(f"增强数据: {args.enhanced}")
    print(f"输出路径: {args.output}")
    print()
    
    # 加载增强数据
    print("加载增强数据...")
    enhanced_index = load_enhanced_data(args.enhanced)
    print(f"  加载了 {len(enhanced_index)} 个增强样本")
    
    # 合并数据
    print("\n合并数据...")
    stats = merge_data(args.sft, enhanced_index, args.output)
    
    # 输出统计
    print("\n" + "=" * 60)
    print("合并完成！")
    print("=" * 60)
    print(f"总样本数: {stats['total_samples']}")
    print(f"匹配样本数: {stats['matched_samples']} ({stats['matched_samples']*100/stats['total_samples']:.1f}%)")
    print(f"总对话数: {stats['total_dialogues']}")
    if stats['total_dialogues'] > 0:
        print(f"增强对话数: {stats['enhanced_dialogues']} ({stats['enhanced_dialogues']*100/stats['total_dialogues']:.1f}%)")
    else:
        print(f"增强对话数: {stats['enhanced_dialogues']}")
    print(f"\n输出文件: {args.output}")
    
    # 输出文件大小
    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"文件大小: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()

