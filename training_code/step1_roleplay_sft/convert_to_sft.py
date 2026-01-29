#!/usr/bin/env python3
"""
Convert data to SFT training format / 将数据转换为 SFT 训练格式

Input / 输入: sft_data_final_synced.jsonl
Output / 输出: sft_train_data.jsonl

Features / 特点:
1. Each sample contains training_samples for multiple characters
   每个样本包含多个角色的 training_samples
2. Extract each character's dialogue as independent SFT sample
   提取每个角色的对话作为独立的 SFT 样本
3. Include complete System Prompt / 包含完整的 System Prompt
4. Generate unique trace_id using trace_id_book_chapter + character name
   使用 trace_id_book_chapter + 角色名生成唯一 trace_id
"""

import json
import os
import argparse
from tqdm import tqdm

def convert_to_sft(input_file, output_file):
    """Convert to SFT format / 转换为 SFT 格式"""
    print(f"Starting conversion / 开始转换: {input_file} -> {output_file}")
    
    total_samples = 0
    sft_samples_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="处理中"):
            try:
                data = json.loads(line)
                total_samples += 1
                
                trace_id_base = data.get('trace_id_book_chapter', '')
                training_samples = data.get('training_samples', {})
                
                # Iterate through each character / 遍历每个角色
                for char_name, messages in training_samples.items():
                    if not messages:
                        continue
                    
                    # Verify system prompt exists / 验证是否有 system prompt
                    if not (messages and messages[0]['role'] == 'system'):
                        # Skip if first message is not system (abnormal data)
                        # 如果第一条不是 system，可能是异常数据，跳过
                        continue
                        
                    # Construct SFT sample / 构造 SFT 样本
                    # trace_id: base_id + character name (handle special chars)
                    # trace_id: base_id + 角色名 (处理特殊字符)
                    safe_char_name = "".join(c if c.isalnum() else "_" for c in char_name)
                    trace_id = f"{trace_id_base}_{safe_char_name}"
                    
                    # Find index of last assistant message / 找到最后一个 assistant 消息的索引
                    last_assistant_idx = -1
                    for i, msg in enumerate(messages):
                        if msg.get('role') == 'assistant':
                            last_assistant_idx = i
                    
                    # Filter messages (ensure correct format) / 过滤 messages (确保格式正确)
                    clean_messages = []
                    for i, msg in enumerate(messages):
                        role = msg.get('role', '')
                        content = msg.get('content', '')
                        
                        # Skip empty content (except Conversation Start marker)
                        # 跳过空内容 (除了 Conversation Start 标记)
                        if not content and content != "===Conversation Start===":
                            continue
                            
                        # Map role (safety check even though source is already formatted)
                        # 映射 role (虽然源数据已经是 system/user/assistant，但为了保险)
                        if role not in ['system', 'user', 'assistant']:
                            # 处理可能存在的其他 role 标记
                            if role == 'user_question': role = 'user'
                            elif role == 'role_answer': role = 'assistant'
                            else: role = 'user' # fallback
                        
                        # Only add system_thinking to last assistant message
                        # 只有最后一个 assistant 消息才加 system_thinking
                        if role == 'assistant' and i == last_assistant_idx:
                            sys_thinking = msg.get('sys_thinking_revised', '')
                            if sys_thinking:
                                content = f"<system_thinking>{sys_thinking}</system_thinking>{content}"
                        
                        clean_messages.append({
                            "role": role,
                            "content": content
                        })
                    
                    # Skip if only system prompt without dialogue
                    # 只有 system prompt 没有对话的不算
                    if len(clean_messages) <= 1:
                        continue
                        
                    sft_sample = {
                        "trace_id": trace_id,
                        "messages": clean_messages
                    }
                    
                    f_out.write(json.dumps(sft_sample, ensure_ascii=False) + '\n')
                    sft_samples_count += 1
                    
            except json.JSONDecodeError:
                print(f"JSON解析错误")
                continue
            except Exception as e:
                print(f"处理错误: {e}")
                continue
                
    print(f"\nConversion complete! / 转换完成!")
    print(f"Source samples / 源文件样本数: {total_samples}")
    print(f"Generated SFT samples / 生成 SFT 样本数: {sft_samples_count}")
    print(f"Output file / 输出文件: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert data to SFT format / 将数据转换为 SFT 格式')
    parser.add_argument('--input', type=str, 
                        default='/path/to/project/data_process/step4_setting_completion/sft_data_final_synced.jsonl',
                        help='Input file path / 输入文件路径')
    parser.add_argument('--output', type=str,
                        default='/path/to/project/code/step1_roleplay_sft/sft_train_data.jsonl',
                        help='Output file path / 输出文件路径')
    
    args = parser.parse_args()
    
    # Ensure output directory exists / 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    convert_to_sft(args.input, args.output)

if __name__ == "__main__":
    main()

