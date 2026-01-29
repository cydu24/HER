#!/usr/bin/env python3
"""
分析增强后数据的 Pattern 多样性
生成统计报告和 Markdown 表格
"""
import json
import re
import math
from collections import Counter
from datetime import datetime
import argparse


def extract_pattern(content: str) -> str:
    """从增强内容中提取 pattern"""
    if not content:
        return 'empty'
    
    segments = []
    for match in re.finditer(r'<role_thinking>.*?</role_thinking>', content, re.DOTALL):
        segments.append(('think', match.start(), match.end()))
    for match in re.finditer(r'<role_action>.*?</role_action>', content, re.DOTALL):
        segments.append(('act', match.start(), match.end()))
    
    segments.sort(key=lambda x: x[1])
    
    pattern = []
    last_end = 0
    
    for tag_type, start, end in segments:
        text_before = content[last_end:start].strip()
        text_before = re.sub(r'<role_thinking>.*?</role_thinking>', '', text_before, flags=re.DOTALL)
        text_before = re.sub(r'<role_action>.*?</role_action>', '', text_before, flags=re.DOTALL)
        if text_before and len(text_before) > 0:
            pattern.append('speech')
        pattern.append(tag_type)
        last_end = end
    
    text_after = content[last_end:].strip()
    if text_after and len(text_after) > 0:
        pattern.append('speech')
    
    return '→'.join(pattern) if pattern else 'empty'


def calculate_entropy(counter: Counter, total: int) -> float:
    """计算香农熵"""
    entropy = 0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def analyze_data(input_path: str):
    """分析数据并返回统计结果"""
    original_patterns = Counter()
    enhanced_patterns = Counter()
    patterns_per_sample = []
    
    total_samples = 0
    total_dialogues = 0
    
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            total_samples += 1
            
            sample_patterns = set()
            
            # 原始对话
            for d in data.get('original_dialogues', []):
                content = d.get('standard_format', '')
                pattern = extract_pattern(content)
                original_patterns[pattern] += 1
            
            # 增强后对话
            for d in data.get('enhanced_dialogues', []):
                content = d.get('enhanced_role_think', '')
                pattern = extract_pattern(content)
                enhanced_patterns[pattern] += 1
                sample_patterns.add(pattern)
                total_dialogues += 1
            
            patterns_per_sample.append(len(sample_patterns))
    
    return {
        'total_samples': total_samples,
        'total_dialogues': total_dialogues,
        'original_patterns': original_patterns,
        'enhanced_patterns': enhanced_patterns,
        'patterns_per_sample': patterns_per_sample
    }


def generate_markdown_report(stats: dict, output_path: str):
    """生成 Markdown 格式的报告"""
    total_samples = stats['total_samples']
    total_dialogues = stats['total_dialogues']
    original_patterns = stats['original_patterns']
    enhanced_patterns = stats['enhanced_patterns']
    patterns_per_sample = stats['patterns_per_sample']
    
    orig_total = sum(original_patterns.values())
    orig_entropy = calculate_entropy(original_patterns, orig_total)
    enh_entropy = calculate_entropy(enhanced_patterns, total_dialogues)
    
    avg_patterns = sum(patterns_per_sample) / len(patterns_per_sample)
    
    # 分组
    high_freq = [(p, c, c/total_dialogues*100) for p, c in enhanced_patterns.most_common() if c/total_dialogues*100 > 1]
    medium_freq = [(p, c, c/total_dialogues*100) for p, c in enhanced_patterns.most_common() if 0.1 <= c/total_dialogues*100 <= 1]
    low_freq = [(p, c, c/total_dialogues*100) for p, c in enhanced_patterns.most_common() if c/total_dialogues*100 < 0.1]
    
    md = f"""# Pattern 多样性分析报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 概览

| 指标 | 数值 |
|------|------|
| 总样本数 | {total_samples:,} |
| 总对话数 | {total_dialogues:,} |
| 平均每样本对话数 | {total_dialogues/total_samples:.1f} |
| Pattern 种类数 | {len(enhanced_patterns)} |
| 每样本平均 Pattern 种类 | {avg_patterns:.2f} |

## 📈 多样性指标

| 指标 | 原始 | 增强后 | 变化 |
|------|------|--------|------|
| Pattern 种类数 | {len(original_patterns)} | {len(enhanced_patterns)} | +{len(enhanced_patterns)-len(original_patterns)} |
| 香农熵 | {orig_entropy:.4f} | {enh_entropy:.4f} | +{enh_entropy-orig_entropy:.4f} ({(enh_entropy-orig_entropy)/orig_entropy*100:+.1f}%) |

## 🔴 高频 Pattern (>1%)

共 {len(high_freq)} 种，占总量 {sum(p[2] for p in high_freq):.1f}%

| 排名 | Pattern | 数量 | 占比 |
|------|---------|------|------|
"""
    
    for i, (pattern, count, pct) in enumerate(high_freq, 1):
        md += f"| {i} | `{pattern}` | {count:,} | {pct:.2f}% |\n"
    
    md += f"""
## 🟡 中频 Pattern (0.1%-1%)

共 {len(medium_freq)} 种，占总量 {sum(p[2] for p in medium_freq):.1f}%

| 排名 | Pattern | 数量 | 占比 |
|------|---------|------|------|
"""
    
    for i, (pattern, count, pct) in enumerate(medium_freq, 1):
        md += f"| {i} | `{pattern}` | {count:,} | {pct:.2f}% |\n"
    
    md += f"""
## 🟢 低频 Pattern (<0.1%)

共 {len(low_freq)} 种，占总量 {sum(p[2] for p in low_freq):.1f}%

<details>
<summary>点击展开前 50 种</summary>

| 排名 | Pattern | 数量 | 占比 |
|------|---------|------|------|
"""
    
    for i, (pattern, count, pct) in enumerate(low_freq[:50], 1):
        md += f"| {i} | `{pattern}` | {count:,} | {pct:.3f}% |\n"
    
    if len(low_freq) > 50:
        md += f"\n*... 还有 {len(low_freq) - 50} 种低频 pattern*\n"
    
    md += """
</details>

## 🎯 关键 Pattern 变化

"""
    
    # 关键 pattern 对比
    key_patterns = ['think→act→speech', 'act→think→speech', 'think→speech', 'act→speech', 'speech']
    md += "| Pattern | 原始占比 | 增强后占比 | 变化 |\n"
    md += "|---------|----------|------------|------|\n"
    
    for pattern in key_patterns:
        orig_pct = original_patterns.get(pattern, 0) / orig_total * 100 if orig_total > 0 else 0
        enh_pct = enhanced_patterns.get(pattern, 0) / total_dialogues * 100 if total_dialogues > 0 else 0
        change = enh_pct - orig_pct
        emoji = "✅" if change < 0 else ("⚠️" if change > 5 else "➖")
        md += f"| `{pattern}` | {orig_pct:.2f}% | {enh_pct:.2f}% | {change:+.2f}% {emoji} |\n"
    
    md += f"""
## 📊 每样本 Pattern 多样性分布

| Pattern 种类数 | 样本数 | 占比 |
|----------------|--------|------|
"""
    
    pattern_count_dist = Counter(patterns_per_sample)
    for n in sorted(pattern_count_dist.keys()):
        count = pattern_count_dist[n]
        pct = count / total_samples * 100
        md += f"| {n} 种 | {count:,} | {pct:.2f}% |\n"
    
    md += f"""
## 📋 总结

1. **Pattern 多样性**: 共 {len(enhanced_patterns)} 种不同 pattern
2. **香农熵**: {enh_entropy:.4f} (原始: {orig_entropy:.4f}，提升 {(enh_entropy-orig_entropy)/orig_entropy*100:.1f}%)
3. **主要 pattern `think→act→speech`**: 占比 {enhanced_patterns.get('think→act→speech', 0)/total_dialogues*100:.2f}%
4. **平均每样本使用**: {avg_patterns:.2f} 种不同 pattern
5. **高频 pattern (>1%)**: {len(high_freq)} 种，占 {sum(p[2] for p in high_freq):.1f}%

---

*此报告由 `analyze_pattern_diversity.py` 自动生成*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    
    return md


def main():
    parser = argparse.ArgumentParser(description='分析 Pattern 多样性')
    parser.add_argument('--input', '-i', type=str, 
                        default='/path/to/data/example',
                        help='输入文件路径')
    parser.add_argument('--output', '-o', type=str,
                        default='/path/to/data/example',
                        help='输出 Markdown 文件路径')
    
    args = parser.parse_args()
    
    print(f"分析文件: {args.input}")
    stats = analyze_data(args.input)
    
    print(f"\n生成报告: {args.output}")
    md = generate_markdown_report(stats, args.output)
    
    print("\n✅ 完成！")
    print(f"   - 总样本: {stats['total_samples']:,}")
    print(f"   - 总对话: {stats['total_dialogues']:,}")
    print(f"   - Pattern 种类: {len(stats['enhanced_patterns'])}")


if __name__ == '__main__':
    main()

