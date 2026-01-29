#!/usr/bin/env python3
"""
清理 full_final 目录中的空数据
1. 移除空character_datasets的书籍
2. 移除plots中text为空的条目
"""

import json
import os
import glob
import shutil
from pathlib import Path

# 源目录和目标目录
SRC_DIR = "/path/to/data/example"
DST_DIR = "/path/to/data/example"

def is_character_datasets_empty(cd):
    """检查character_datasets是否为空"""
    if cd is None:
        return True
    if isinstance(cd, dict) and len(cd) == 0:
        return True
    if isinstance(cd, list) and len(cd) == 0:
        return True
    return False

def clean_plots(plots):
    """清理plots，移除text为空的条目"""
    cleaned = []
    removed_count = 0
    for plot in plots:
        text = plot.get('text', '')
        if text and len(text.strip()) > 0:
            cleaned.append(plot)
        else:
            removed_count += 1
    return cleaned, removed_count

def main():
    # 创建目标目录
    os.makedirs(DST_DIR, exist_ok=True)
    
    json_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.json")))
    
    stats = {
        'total_files': len(json_files),
        'removed_empty_cd': 0,
        'removed_empty_cd_books': [],
        'kept_files': 0,
        'total_plots_removed': 0,
        'files_with_plots_removed': 0
    }
    
    print(f"开始处理 {len(json_files)} 个文件...")
    print(f"源目录: {SRC_DIR}")
    print(f"目标目录: {DST_DIR}")
    print("-" * 50)
    
    for f in json_files:
        filename = os.path.basename(f)
        
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 检查character_datasets是否为空
        cd = data.get('character_datasets')
        if is_character_datasets_empty(cd):
            stats['removed_empty_cd'] += 1
            stats['removed_empty_cd_books'].append(filename)
            print(f"❌ 跳过 (空character_datasets): {filename}")
            continue
        
        # 清理plots
        if 'plots' in data:
            cleaned_plots, removed_count = clean_plots(data['plots'])
            if removed_count > 0:
                stats['total_plots_removed'] += removed_count
                stats['files_with_plots_removed'] += 1
                print(f"⚠️  {filename}: 移除了 {removed_count} 个空text的plots")
            data['plots'] = cleaned_plots
        
        # 保存到目标目录
        dst_path = os.path.join(DST_DIR, filename)
        with open(dst_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        
        stats['kept_files'] += 1
    
    print("-" * 50)
    print("\n📊 处理完成统计:")
    print(f"  原始文件数: {stats['total_files']}")
    print(f"  保留文件数: {stats['kept_files']}")
    print(f"  因空character_datasets移除的书籍: {stats['removed_empty_cd']}")
    print(f"  移除的空text plots总数: {stats['total_plots_removed']}")
    print(f"  有plots被移除的文件数: {stats['files_with_plots_removed']}")
    
    if stats['removed_empty_cd_books']:
        print(f"\n❌ 被移除的书籍列表:")
        for book in stats['removed_empty_cd_books']:
            print(f"    - {book}")
    
    print(f"\n✅ 清理后的数据保存在: {DST_DIR}")

if __name__ == "__main__":
    main()

