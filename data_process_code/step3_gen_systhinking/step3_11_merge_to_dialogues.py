#!/usr/bin/env python3
"""
将 sys_thinking 从 training_samples 合并到 dialogues
"""

import json
from tqdm import tqdm

INPUT_PATH = "/path/to/data/example"
OUTPUT_PATH = "/path/to/data/example"


def merge_sys_thinking_to_dialogues(sample):
    """将 sys_thinking 从 training_samples 合并到 dialogues"""
    
    # 获取 dialogues
    conversation = sample.get('conversation', [])
    if not conversation:
        return sample
    
    dialogues = conversation[0].get('dialogues', [])
    if not dialogues:
        return sample
    
    training_samples = sample.get('training_samples', {})
    
    # 构建 (character, dialogue_index) -> sys_thinking 映射
    sys_thinking_map = {}
    for character, turns in training_samples.items():
        for turn in turns:
            if turn.get('role') != 'assistant':
                continue
            
            origin_id = turn.get('origin_id')
            if origin_id is None:
                continue
            
            if isinstance(origin_id, list):
                origin_id = origin_id[0] if origin_id else None
            
            if origin_id is None:
                continue
            
            # 获取 sys_thinking (优先 revised，其次 original)
            sys_thinking = turn.get('sys_thinking_revised') or turn.get('sys_thinking_original') or ''
            
            if sys_thinking:
                sys_thinking_map[(character, origin_id)] = sys_thinking
    
    # 合并到 dialogues
    for i, dlg in enumerate(dialogues):
        char = dlg.get('character', '')
        if char and (char, i) in sys_thinking_map:
            dlg['sys_thinking'] = sys_thinking_map[(char, i)]
    
    return sample


def main():
    print("合并 sys_thinking 到 dialogues...")
    
    updated_count = 0
    total_dialogues = 0
    with_sys_thinking = 0
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="处理中"):
            sample = json.loads(line)
            
            # 统计原始状态
            conv = sample.get('conversation', [])
            if conv:
                dialogues = conv[0].get('dialogues', [])
                total_dialogues += len(dialogues)
            
            # 合并
            merged = merge_sys_thinking_to_dialogues(sample)
            
            # 统计合并后
            if conv:
                dialogues = merged['conversation'][0].get('dialogues', [])
                for dlg in dialogues:
                    if dlg.get('sys_thinking'):
                        with_sys_thinking += 1
            
            f_out.write(json.dumps(merged, ensure_ascii=False) + '\n')
            updated_count += 1
    
    print(f"\n=== 完成 ===")
    print(f"处理样本数: {updated_count}")
    print(f"总 dialogues: {total_dialogues}")
    print(f"有 sys_thinking: {with_sys_thinking} ({with_sys_thinking/total_dialogues*100:.1f}%)")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

