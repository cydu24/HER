#!/usr/bin/env python3
"""
同步 conversation[0]['dialogues'] 中的 enhanced_standard_format 到 training_samples

逻辑：
1. 从 dialogues 中取 enhanced_standard_format
2. 根据 origin_id 匹配 training_samples 中的消息
3. assistant (自己的发言)：保留完整内容（包含 <role_thinking>）
4. user (别人的发言)：去掉 <role_thinking>...</role_thinking>（看不到别人的思考）

输入: sft_data_final_patched.jsonl
输出: sft_data_final_synced.jsonl
"""

import json
import re
from tqdm import tqdm


INPUT_PATH = "/path/to/project/data_process/step4_setting_completion/sft_data_final_patched.jsonl"
OUTPUT_PATH = "/path/to/project/data_process/step4_setting_completion/sft_data_final_synced.jsonl"


def remove_role_thinking(text: str) -> str:
    """
    去掉 <role_thinking>...</role_thinking>
    可能在开头、中间、结尾，可能有多个
    """
    if not text:
        return text
    # 非贪婪匹配，去掉所有 <role_thinking>...</role_thinking>
    result = re.sub(r'<role_thinking>.*?</role_thinking>', '', text, flags=re.DOTALL)
    # 清理多余空格
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def sync_training_samples():
    print("=" * 60)
    print("同步 dialogues 到 training_samples")
    print("=" * 60)
    print(f"输入: {INPUT_PATH}")
    print(f"输出: {OUTPUT_PATH}")
    print()
    
    total = 0
    synced_samples = 0
    synced_assistant = 0
    synced_user = 0
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="处理中"):
            data = json.loads(line)
            total += 1
            
            # 获取 dialogues (在 conversation[0] 里)
            conversations = data.get('conversation', [])
            if not conversations:
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                continue
                
            dialogues = conversations[0].get('dialogues', [])
            if not dialogues:
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                continue
            
            # 建立 origin_id -> (character, enhanced_standard_format) 的映射
            dialogue_map = {}
            for dlg in dialogues:
                origin_id = dlg.get('origin_id', [])
                enhanced = dlg.get('enhanced_standard_format', '')
                character = dlg.get('character', '')
                if origin_id and enhanced:
                    key = tuple(origin_id)
                    dialogue_map[key] = {
                        'character': character,
                        'enhanced': enhanced
                    }
            
            # 更新 training_samples
            training_samples = data.get('training_samples', {})
            sample_updated = False
            
            for current_char, messages in training_samples.items():
                for msg in messages:
                    origin_id = msg.get('origin_id', [])
                    if not origin_id:
                        continue
                    
                    key = tuple(origin_id)
                    if key not in dialogue_map:
                        continue
                    
                    dlg_info = dialogue_map[key]
                    speaking_char = dlg_info['character']
                    enhanced = dlg_info['enhanced']
                    
                    if msg.get('role') == 'assistant':
                        # 自己的发言：保留完整内容（包含 <role_thinking>）
                        msg['content'] = f"{speaking_char}: {enhanced}"
                        synced_assistant += 1
                        sample_updated = True
                        
                    elif msg.get('role') == 'user':
                        # 别人的发言：去掉 <role_thinking>
                        cleaned = remove_role_thinking(enhanced)
                        msg['content'] = f"{speaking_char}: {cleaned}"
                        synced_user += 1
                        sample_updated = True
            
            if sample_updated:
                synced_samples += 1
            
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print()
    print("=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"总样本数: {total}")
    print(f"更新的样本数: {synced_samples}")
    print(f"更新的 assistant 消息数: {synced_assistant}")
    print(f"更新的 user 消息数: {synced_user}")
    print(f"输出: {OUTPUT_PATH}")


if __name__ == "__main__":
    sync_training_samples()
