#!/usr/bin/env python3
"""
构造完整的 Vulcan 推理格式数据
用于批量调用 model API 进行 Role Thinking 增强

支持中文和英文两种 Prompt 版本
"""

import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '.')
from role_thinking_enhance_prompt_v2 import extract_input_data_from_sample, build_prompt

# 路径配置
INPUT_FILE = '/path/to/data/example'
OUTPUT_DIR = '/path/to/data/example'


def construct_vulcan_item(sample: dict, sample_idx: int, lang: str = "zh") -> dict:
    """
    构造单条 Vulcan 推理格式数据
    
    Args:
        sample: 原始样本数据
        sample_idx: 样本索引
        lang: 语言 "zh" 或 "en"
    """
    # 提取完整输入数据（包含所有元数据）
    input_data = extract_input_data_from_sample(sample)
    
    if not input_data["dialogues"]:
        return None
    
    # 构建 prompt（根据语言选择）
    prompt = build_prompt(input_data, lang=lang)
    
    trace_id = input_data["trace_id"]
    
    # Vulcan 推理格式
    vulcan_item = {
        "trace_id": f"role_thinking_enhance_{lang}_{trace_id}",
        "data": [
            {
                "role": "user",
                "text": prompt,
                "name": "user"
            },
            {
                "role": "ai",
                "text": "",
                "name": "ai"
            }
        ],
        "model_control": None,
        "follow_system": True,
        "train_start_index": -1,
        "need_valid": True,
        "raw_record": {
            # 保留完整的输入数据，方便后期对齐
            "input_data": input_data,
            "sample_idx": sample_idx,
            "lang": lang
        }
    }
    
    return vulcan_item


def main(max_samples: int = None, split_size: int = 5000, lang: str = "zh"):
    """
    主函数
    
    Args:
        max_samples: 最大样本数（None 表示全部）
        split_size: 每个文件的样本数
        lang: 语言 "zh" 中文 或 "en" 英文
    """
    lang_name = "中文" if lang == "zh" else "英文"
    output_dir = os.path.join(OUTPUT_DIR, lang)
    
    print("=" * 60)
    print(f"构造 Vulcan 推理格式数据 ({lang_name})")
    print("=" * 60)
    print(f"输入: {INPUT_FILE}")
    print(f"输出目录: {output_dir}")
    print(f"语言: {lang_name}")
    if max_samples:
        print(f"最大样本数: {max_samples}")
    print(f"每文件样本数: {split_size}")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print("读取输入数据...")
    samples = []
    with open(INPUT_FILE, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line))
    
    print(f"读取 {len(samples)} 条样本")
    print()
    
    # 构造 Vulcan 数据
    print(f"构造 Vulcan 格式数据 ({lang_name})...")
    vulcan_data = []
    total_dialogues = 0
    skipped = 0
    
    for i, sample in enumerate(tqdm(samples, desc="处理样本")):
        item = construct_vulcan_item(sample, i, lang=lang)
        if item:
            vulcan_data.append(item)
            total_dialogues += len(item["raw_record"]["input_data"]["dialogues"])
        else:
            skipped += 1
    
    print(f"\n生成 {len(vulcan_data)} 条 Vulcan 数据")
    print(f"跳过 {skipped} 条（无对话）")
    print(f"总对话数: {total_dialogues}")
    print()
    
    # 分割保存
    print("保存数据...")
    file_count = 0
    for start in range(0, len(vulcan_data), split_size):
        end = min(start + split_size, len(vulcan_data))
        chunk = vulcan_data[start:end]
        
        output_file = os.path.join(output_dir, f"role_thinking_enhance_{lang}_part{file_count}.jsonl")
        with open(output_file, 'w') as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  {output_file} ({len(chunk)} 条)")
        file_count += 1
    
    # 保存全量文件
    full_output_file = os.path.join(output_dir, f"role_thinking_enhance_{lang}_full.jsonl")
    with open(full_output_file, 'w') as f:
        for item in vulcan_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  {full_output_file} (全量 {len(vulcan_data)} 条)")
    
    print()
    print("=" * 60)
    print(f"✅ {lang_name}版完成!")
    print("=" * 60)
    print(f"总样本数: {len(vulcan_data)}")
    print(f"总对话数: {total_dialogues}")
    print(f"文件数: {file_count + 1} (含全量文件)")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    return len(vulcan_data), total_dialogues


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("-s", "--split_size", type=int, default=5000, help="每文件样本数")
    parser.add_argument("-l", "--lang", type=str, default="both", choices=["zh", "en", "both"], help="语言: zh/en/both")
    args = parser.parse_args()
    
    if args.lang == "both":
        # 生成中英文两个版本
        print("\n" + "=" * 60)
        print("生成中英文两个版本")
        print("=" * 60 + "\n")
        
        main(args.max_samples, args.split_size, lang="zh")
        print("\n")
        main(args.max_samples, args.split_size, lang="en")
    else:
        main(args.max_samples, args.split_size, lang=args.lang)

