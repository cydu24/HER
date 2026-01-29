#!/usr/bin/env python3
"""
Step 4.5: 合并补丁数据到主文件

将 merged_fix_v3.jsonl 和 merged_tx_format.jsonl 中的增强数据合并到主数据文件:
1. fix_v3 的 'fixed' 字段 → 覆盖 dialogue 的 'enhanced_standard_format'
2. tx_format 的 'tx_format' 字段 → 新增 dialogue 的 'tx_format' 字段

输入:
  - sft_data_final_v6_with_config.jsonl: 主数据文件
  - merged_fix_v3.jsonl: 人称修正补丁
  - merged_tx_format.jsonl: TX格式转换补丁

输出:
  - sft_data_final_v7_patched.jsonl: 合并后的数据
"""

import json
from collections import defaultdict
from tqdm import tqdm

# 路径配置
MAIN_DATA_PATH = "/path/to/project/data_process/step4_setting_completion/sft_data_final_v6_with_config.jsonl"
FIX_V3_PATH = "/path/to/project/data_process/step2_gen_rolethinking/merged_fix_v3.jsonl"
TX_FORMAT_PATH = "/path/to/project/data_process/step2_gen_rolethinking/merged_tx_format.jsonl"
OUTPUT_PATH = "/path/to/project/data_process/step4_setting_completion/sft_data_final_v7_patched.jsonl"


def load_patch_data(patch_path: str, prefix: str, field_name: str) -> dict:
    """
    加载补丁数据，建立索引
    
    Args:
        patch_path: 补丁文件路径
        prefix: trace_id 前缀（如 'fix_v3_' 或 'tx_v3_'）
        field_name: 要提取的字段名（如 'fixed' 或 'tx_format'）
    
    Returns:
        dict: {(trace_id, origin_id_tuple): field_value}
    """
    index = {}
    
    with open(patch_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # 提取原始 trace_id
            trace_id = data.get('trace_id', '')
            if trace_id.startswith(prefix):
                trace_id = trace_id[len(prefix):]
            
            # 从 raw_record 获取原始 trace_id（更准确）
            raw_record = data.get('raw_record', {})
            original_trace_id = raw_record.get('original_trace_id', trace_id)
            
            # 遍历 dialogues
            model_response = data.get('model_response', {})
            for dlg in model_response.get('dialogues', []):
                origin_id = dlg.get('origin_id', [])
                key = (original_trace_id, tuple(origin_id))
                
                value = dlg.get(field_name, '')
                if value:
                    index[key] = value
    
    return index


def main():
    print("=" * 70)
    print("Step 4.5: 合并补丁数据 (fix_v3 + tx_format)")
    print("=" * 70)
    print(f"主数据: {MAIN_DATA_PATH}")
    print(f"Fix_v3: {FIX_V3_PATH}")
    print(f"TX格式: {TX_FORMAT_PATH}")
    print(f"输出:   {OUTPUT_PATH}")
    print()
    
    # 1. 加载补丁数据
    print("加载补丁数据...")
    fix_v3_index = load_patch_data(FIX_V3_PATH, 'fix_v3_', 'fixed')
    print(f"  fix_v3 索引: {len(fix_v3_index)} 条")
    
    tx_format_index = load_patch_data(TX_FORMAT_PATH, 'tx_v3_', 'tx_format')
    print(f"  tx_format 索引: {len(tx_format_index)} 条")
    
    # 2. 处理主数据
    print("\n处理主数据...")
    total_samples = 0
    total_dialogues = 0
    fix_v3_matched = 0
    tx_format_matched = 0
    
    with open(MAIN_DATA_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="合并中"):
            sample = json.loads(line)
            total_samples += 1
            
            trace_id = sample.get('trace_id_book_chapter', '')
            
            # 遍历所有 conversation 的 dialogues
            for conv in sample.get('conversation', []):
                for dlg in conv.get('dialogues', []):
                    total_dialogues += 1
                    origin_id = dlg.get('origin_id', [])
                    key = (trace_id, tuple(origin_id))
                    
                    # 匹配 fix_v3 补丁 → 覆盖 enhanced_standard_format
                    if key in fix_v3_index:
                        dlg['enhanced_standard_format'] = fix_v3_index[key]
                        fix_v3_matched += 1
                    
                    # 匹配 tx_format 补丁 → 新增 tx_format 字段
                    if key in tx_format_index:
                        dlg['tx_format'] = tx_format_index[key]
                        tx_format_matched += 1
            
            # 同时更新 training_samples 中的内容
            for char_name, messages in sample.get('training_samples', {}).items():
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        origin_id = msg.get('origin_id', [])
                        key = (trace_id, tuple(origin_id))
                        
                        # 更新 assistant 消息的 content
                        if key in fix_v3_index:
                            # content 格式: "{character}: {enhanced_standard_format}"
                            old_content = msg.get('content', '')
                            if ': ' in old_content:
                                char_prefix = old_content.split(': ', 1)[0]
                                msg['content'] = f"{char_prefix}: {fix_v3_index[key]}"
                        
                        # 添加 tx_format 字段
                        if key in tx_format_index:
                            msg['tx_format'] = tx_format_index[key]
            
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 3. 输出统计
    print()
    print("=" * 70)
    print("合并完成!")
    print("=" * 70)
    print(f"总样本数:        {total_samples}")
    print(f"总对话数:        {total_dialogues}")
    print(f"fix_v3 匹配:     {fix_v3_matched} ({fix_v3_matched*100/total_dialogues:.1f}%)")
    print(f"tx_format 匹配:  {tx_format_matched} ({tx_format_matched*100/total_dialogues:.1f}%)")
    print(f"\n输出文件: {OUTPUT_PATH}")
    
    # 4. 验证
    print("\n验证结果:")
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        first = json.loads(f.readline())
        conv = first.get('conversation', [{}])[0]
        dlgs = conv.get('dialogues', [])
        if dlgs:
            print(f"  第一条 dialogue 字段: {list(dlgs[0].keys())}")
            if 'tx_format' in dlgs[0]:
                print(f"  tx_format 示例: {dlgs[0]['tx_format'][:100]}...")


if __name__ == "__main__":
    main()

