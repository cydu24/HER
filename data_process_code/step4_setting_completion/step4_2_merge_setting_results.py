#!/usr/bin/env python3
"""
Step 4.2: 合并 Setting Completion 推理结果

将增强后的字段合并回 character_datasets，作为新字段保存
方便后续做对比实验，不覆盖原始数据
"""

import json
import os
import glob
import re
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool

# 路径配置
INPUT_SFT_PATH = "/path/to/data/example"
OUTPUT_DIR = "/path/to/data/example"
OUTPUT_SFT_PATH = "/path/to/data/example"

NUM_WORKERS = 64


def extract_json_from_text(text):
    """从文本中提取 JSON 对象"""
    text = text.strip()

    # 尝试从 markdown 代码块提取
    if '```json' in text:
        match = re.search(r'```json\s*([\s\S]*?)```', text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except:
                pass

    # 尝试从普通代码块提取
    if '```' in text:
        match = re.search(r'```\s*([\s\S]*?)```', text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except:
                pass

    # 尝试直接解析
    if text.startswith('{'):
        try:
            return json.loads(text)
        except:
            pass

    return None


def process_single_file(filepath):
    """处理单个输出文件"""
    results = {}
    fail_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                trace_id = d.get('trace_id', '').replace('setting_', '')

                if not trace_id:
                    fail_count += 1
                    continue

                # 解析 trace_id 获取 i_c
                parts = trace_id.rsplit('_', 1)
                if len(parts) != 2:
                    fail_count += 1
                    continue

                original_trace_id = parts[0]
                i_c = int(parts[1])

                # 获取模型输出
                vo = d.get('vulcan_output', {})
                candidates = vo.get('model_req', {}).get('candidates', [])
                if not candidates:
                    fail_count += 1
                    continue

                parts_content = candidates[0].get('content', {}).get('parts', [])
                if len(parts_content) < 2:
                    fail_count += 1
                    continue

                text = parts_content[1].get('text', '')

                # 提取 JSON (增强结果)
                enriched_data = extract_json_from_text(text)
                if enriched_data:
                    # 按 original_trace_id 和 i_c 组织数据
                    if original_trace_id not in results:
                        results[original_trace_id] = {}
                    results[original_trace_id][i_c] = enriched_data
                else:
                    fail_count += 1

            except Exception as e:
                fail_count += 1

    return results, fail_count


def load_results_parallel():
    """并行加载推理结果"""
    output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.jsonl"))
    print(f"找到 {len(output_files)} 个输出文件")

    if not output_files:
        print("警告: 没有找到输出文件!")
        return {}

    all_results = {}
    total_fail = 0

    with Pool(NUM_WORKERS) as pool:
        for file_results, fail_count in tqdm(
            pool.imap_unordered(process_single_file, output_files),
            total=len(output_files),
            desc="并行加载结果"
        ):
            # 合并结果
            for trace_id, conversations in file_results.items():
                if trace_id not in all_results:
                    all_results[trace_id] = {}
                all_results[trace_id].update(conversations)
            total_fail += fail_count

    # 统计
    total_samples = len(all_results)
    total_conversations = sum(len(convs) for convs in all_results.values())

    print(f"成功解析样本: {total_samples}")
    print(f"成功解析对话: {total_conversations}")
    print(f"解析失败: {total_fail} 条")

    return all_results


def merge_enriched_to_character_datasets(original_char_data, enriched_char_data):
    """
    将增强后的字段合并到 character_datasets
    保留原始字段，增强字段用 _enriched 后缀
    """
    merged = original_char_data.copy()

    # 需要合并的字段
    fields_to_merge = [
        'character_profile',
        'background',
        'motivation',
        'description',
        'experience'
    ]

    for field in fields_to_merge:
        enriched_key = f'{field}_enriched'
        if enriched_key in enriched_char_data:
            merged[enriched_key] = enriched_char_data[enriched_key]

    # 添加 reasoning（如果存在）
    if 'reasoning' in enriched_char_data:
        merged['setting_enrichment_reasoning'] = enriched_char_data['reasoning']

    return merged


def merge_to_sft_data(enriched_results):
    """
    将增强后的字段合并到 SFT 数据
    """

    total_samples = 0
    updated_samples = 0
    updated_conversations = 0
    updated_characters = 0
    not_found = 0

    with open(INPUT_SFT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_SFT_PATH, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="合并数据"):
            sample = json.loads(line)
            total_samples += 1

            trace_id = sample.get('trace_id_book_chapter', '')

            # 检查是否有增强结果
            if trace_id not in enriched_results:
                not_found += 1
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                continue

            # 获取该样本的所有对话增强结果
            conversations_enriched = enriched_results[trace_id]

            # 处理每个对话
            conversations = sample.get('conversation', [])
            sample_updated = False

            for conv in conversations:
                i_c = conv.get('i_c', 0)

                if i_c not in conversations_enriched:
                    continue

                enriched_data = conversations_enriched[i_c]

                # 1. 更新 scenario（如果存在）
                if 'scenario_enriched' in enriched_data:
                    conv['scenario_enriched'] = enriched_data['scenario_enriched']
                    if 'scenario_reasoning' in enriched_data:
                        conv['scenario_enrichment_reasoning'] = enriched_data['scenario_reasoning']

                # 2. 更新 character_datasets
                character_datasets = sample.get('character_datasets', {})
                enriched_characters = enriched_data.get('characters', {})

                for char_name, enriched_char_data in enriched_characters.items():
                    if char_name in character_datasets:
                        # 合并增强字段
                        merged_data = merge_enriched_to_character_datasets(
                            character_datasets[char_name],
                            enriched_char_data
                        )
                        character_datasets[char_name] = merged_data
                        updated_characters += 1
                        sample_updated = True

                if sample_updated:
                    updated_conversations += 1

            if sample_updated:
                updated_samples += 1

            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    return total_samples, updated_samples, updated_conversations, updated_characters, not_found


def show_sample_comparison():
    """显示增强前后的对比示例"""
    print("\n" + "=" * 80)
    print("增强前后对比示例")
    print("=" * 80)

    with open(OUTPUT_SFT_PATH, 'r') as f:
        for line in f:
            sample = json.loads(line)
            character_datasets = sample.get('character_datasets', {})

            # 找到第一个有 enriched 字段的角色
            for char_name, char_data in character_datasets.items():
                if 'character_profile_enriched' in char_data:
                    print(f"\n角色: {char_name}")
                    print("-" * 80)

                    # 对比各个字段
                    fields = ['character_profile', 'background', 'motivation', 'description', 'experience']

                    for field in fields:
                        original = char_data.get(field, '')
                        enriched = char_data.get(f'{field}_enriched', '')

                        if enriched:
                            print(f"\n【{field}】")
                            print(f"原始 ({len(original)} 字符): {original[:200]}...")
                            print(f"增强 ({len(enriched)} 字符): {enriched[:200]}...")

                    # 显示 reasoning
                    if 'setting_enrichment_reasoning' in char_data:
                        print(f"\n【reasoning】")
                        reasoning = char_data['setting_enrichment_reasoning']
                        print(reasoning[:500] + "...")

                    return  # 只显示第一个示例


def main():
    print("=" * 80)
    print("Step 4.2: 合并 Setting Completion 结果")
    print("=" * 80)
    print(f"输入: {INPUT_SFT_PATH}")
    print(f"输出: {OUTPUT_SFT_PATH}")
    print(f"结果目录: {OUTPUT_DIR}")
    print()

    # 加载推理结果
    enriched_results = load_results_parallel()

    if not enriched_results:
        print("没有找到可合并的结果，退出")
        return

    # 合并
    print()
    total, updated_samples, updated_convs, updated_chars, not_found = merge_to_sft_data(enriched_results)

    print()
    print("=" * 80)
    print("合并完成!")
    print("=" * 80)
    print(f"总样本数: {total}")
    print(f"已更新样本: {updated_samples} ({updated_samples/total*100:.2f}%)")
    print(f"已更新对话: {updated_convs}")
    print(f"已更新角色字段: {updated_chars}")
    print(f"未找到增强结果: {not_found}")
    print(f"\n输出文件: {OUTPUT_SFT_PATH}")

    # 显示对比示例
    show_sample_comparison()


if __name__ == "__main__":
    main()

