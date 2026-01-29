#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规范化文件名

处理规则：
1. 空格 → 下划线 _
2. 多个连续空格/下划线 → 单个下划线
3. 特殊Unicode字符:
   - 零宽空格 (U+200B) → 移除
   - 右单引号 (U+2019 ') → 普通单引号 '
   - 长破折号 (U+2014 —) → 短横线 -
   - 带重音字母保留 (é, Á, å 等)
4. 移除感叹号 ! 和问号 ?
5. 分号 ; → 逗号 ,
"""

import os
import re
import glob
import json
from typing import Dict, Tuple

# 配置
INPUT_DIR = "/path/to/data/example"
DRY_RUN = False  # 设为 True 只预览，不执行重命名


def get_file_richness(filepath: str) -> int:
    """获取文件的丰富度（plots数量），用于比较重复文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data.get('plots', []))
    except:
        return 0


def normalize_filename(filename: str) -> str:
    """规范化文件名"""
    name = filename
    
    # 1. 特殊Unicode字符替换
    replacements = {
        '\u200B': '',      # 零宽空格 → 移除
        '\u2019': "'",     # 右单引号 → 普通单引号
        '\u2018': "'",     # 左单引号 → 普通单引号
        '\u201C': '"',     # 左双引号 → 普通双引号
        '\u201D': '"',     # 右双引号 → 普通双引号
        '\u2014': '-',     # 长破折号 → 短横线
        '\u2013': '-',     # 中破折号 → 短横线
        '\u2026': '...',   # 省略号 → 三个点
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # 2. 移除感叹号和问号
    name = name.replace('!', '')
    name = name.replace('?', '')
    
    # 3. 分号替换为逗号
    name = name.replace(';', ',')
    
    # 4. 空格替换为下划线
    name = name.replace(' ', '_')
    
    # 5. 多个连续下划线替换为单个
    name = re.sub(r'_+', '_', name)
    
    # 6. 移除文件名开头和结尾的下划线（保留.json扩展名）
    if name.endswith('.json'):
        base = name[:-5]
        base = base.strip('_')
        name = base + '.json'
    
    return name


def main():
    print("=" * 60)
    print("文件名规范化工具")
    print("=" * 60)
    print(f"目录: {INPUT_DIR}")
    print(f"模式: {'预览 (DRY RUN)' if DRY_RUN else '执行重命名'}")
    print()
    
    # 获取所有JSON文件
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    print(f"找到 {len(json_files)} 个文件")
    print()
    
    # 分析并重命名
    changes: Dict[str, str] = {}
    merge_conflicts = []  # 需要合并的冲突（保留更丰富的）
    no_change = 0
    
    for filepath in json_files:
        old_name = os.path.basename(filepath)
        new_name = normalize_filename(old_name)
        
        if old_name != new_name:
            # 检查是否有冲突
            new_path = os.path.join(INPUT_DIR, new_name)
            
            if new_name in changes.values():
                # 找到之前映射到这个目标的文件
                prev_old = [k for k, v in changes.items() if v == new_name][0]
                prev_path = os.path.join(INPUT_DIR, prev_old)
                
                # 比较两个文件的丰富度
                prev_richness = get_file_richness(prev_path)
                curr_richness = get_file_richness(filepath)
                
                if curr_richness > prev_richness:
                    # 当前文件更丰富，删除之前的映射，添加当前的
                    merge_conflicts.append((prev_old, old_name, new_name, prev_richness, curr_richness, "保留当前"))
                    del changes[prev_old]
                    changes[old_name] = new_name
                else:
                    # 之前的文件更丰富，保留之前的
                    merge_conflicts.append((old_name, prev_old, new_name, curr_richness, prev_richness, "保留之前"))
                    
            elif os.path.exists(new_path) and new_path != filepath:
                # 目标文件已存在
                existing_richness = get_file_richness(new_path)
                curr_richness = get_file_richness(filepath)
                
                if curr_richness > existing_richness:
                    # 当前文件更丰富，标记删除已存在的
                    merge_conflicts.append((new_name, old_name, new_name, existing_richness, curr_richness, "替换已存在"))
                    changes[old_name] = new_name
                else:
                    # 已存在的更丰富，删除当前文件
                    merge_conflicts.append((old_name, new_name, new_name, curr_richness, existing_richness, "删除当前"))
            else:
                changes[old_name] = new_name
        else:
            no_change += 1
    
    # 显示统计
    print(f"无需改变: {no_change} 个文件")
    print(f"需要重命名: {len(changes)} 个文件")
    print(f"合并冲突: {len(merge_conflicts)} 个")
    print()
    
    # 显示合并冲突
    if merge_conflicts:
        print("⚠️ 合并冲突处理:")
        for item in merge_conflicts:
            loser, winner, target, loser_plots, winner_plots, action = item
            print(f"  目标: {target}")
            print(f"  删除: {loser} ({loser_plots} plots)")
            print(f"  保留: {winner} ({winner_plots} plots)")
            print(f"  操作: {action}")
            print()
    
    # 显示改变预览
    print("📝 重命名预览 (前20个):")
    for i, (old, new) in enumerate(list(changes.items())[:20]):
        print(f"  {old}")
        print(f"  → {new}")
        print()
    
    if len(changes) > 20:
        print(f"  ... 共 {len(changes)} 个文件需要重命名")
    
    # 执行重命名
    if not DRY_RUN:
        print()
        print("=" * 60)
        print("执行重命名...")
        print("=" * 60)
        
        success = 0
        failed = 0
        deleted = 0
        
        # 1. 首先处理合并冲突（删除较差的文件）
        for item in merge_conflicts:
            loser, winner, target, loser_plots, winner_plots, action = item
            loser_path = os.path.join(INPUT_DIR, loser)
            
            if os.path.exists(loser_path):
                try:
                    os.remove(loser_path)
                    print(f"🗑️ 删除: {loser} ({loser_plots} plots)")
                    deleted += 1
                except Exception as e:
                    print(f"❌ 删除失败: {loser} → {e}")
        
        # 2. 然后执行重命名
        for old_name, new_name in changes.items():
            old_path = os.path.join(INPUT_DIR, old_name)
            new_path = os.path.join(INPUT_DIR, new_name)
            
            if not os.path.exists(old_path):
                continue  # 文件可能已被删除
            
            try:
                # 如果目标已存在（且当前文件更好），先删除目标
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(old_path, new_path)
                success += 1
            except Exception as e:
                print(f"❌ 重命名失败: {old_name} → {e}")
                failed += 1
        
        print()
        print(f"✅ 成功重命名: {success} 个文件")
        print(f"🗑️ 删除重复: {deleted} 个文件")
        if failed:
            print(f"❌ 失败: {failed} 个文件")
    
    else:
        print()
        print("💡 这是预览模式。要执行重命名，请将 DRY_RUN 设为 False")


if __name__ == '__main__':
    main()

