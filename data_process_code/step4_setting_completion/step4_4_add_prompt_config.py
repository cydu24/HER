#!/usr/bin/env python3
"""
Step 4.4: 将 system prompt 配置字段添加到数据文件中

在每个样本中添加 prompt_config 字段，方便下次直接拼接
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

# 添加父目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))
from system_prompt_config import OUTPUT_FORMAT, ENRICHED_FIELDS, PROFILE_FIELDS

# 路径配置
INPUT_PATH = "/path/to/project/data_process/step4_setting_completion/sft_data_final_v6_full_prompt.jsonl"
OUTPUT_PATH = "/path/to/project/data_process/step4_setting_completion/sft_data_final_v6_with_config.jsonl"


# 要存入 jsonl 的配置
PROMPT_CONFIG = {
    "output_format": OUTPUT_FORMAT,
    
    "enriched_fields": ENRICHED_FIELDS,
    
    "profile_fields": [
        {"field": f, "template": t} for f, t in PROFILE_FIELDS
    ],
    
    "system_prompt_sections": [
        {"name": "profile", "header": "==={character}'s Profile==="},
        {"name": "background", "header": "===Background==="},
        {"name": "scenario", "header": "===Current Scenario==="},
        {"name": "other_characters", "header": "===Information about other Characters==="},
        {"name": "motivation", "header": "===Your Inner Thoughts==="},
        {"name": "requirements", "header": "===Requirements==="},
    ],
    
    "version": "v6",
    "description": "System prompt 合成配置，包含输出格式、增强字段列表、各部分模板等"
}


def main():
    print("=" * 60)
    print("Step 4.4: 将配置字段添加到数据文件")
    print("=" * 60)
    print(f"输入: {INPUT_PATH}")
    print(f"输出: {OUTPUT_PATH}")
    print()
    
    total = 0
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="处理中"):
            sample = json.loads(line)
            total += 1
            
            # 添加 prompt_config 字段
            sample['prompt_config'] = PROMPT_CONFIG
            
            # 确保每个角色的 character_datasets 都有 output_format
            for char_name, char_data in sample.get('character_datasets', {}).items():
                if 'output_format' not in char_data or not char_data['output_format']:
                    char_data['output_format'] = OUTPUT_FORMAT
            
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print()
    print("=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"总样本数: {total}")
    print(f"输出文件: {OUTPUT_PATH}")
    
    # 验证
    print("\n验证添加的字段:")
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        first = json.loads(f.readline())
        print(f"  prompt_config keys: {list(first.get('prompt_config', {}).keys())}")
        print(f"  output_format 长度: {len(first['prompt_config'].get('output_format', ''))}")


if __name__ == "__main__":
    main()

