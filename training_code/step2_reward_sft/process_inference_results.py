#!/usr/bin/env python3
"""
处理推理结果，从推理输出中提取和构造数据
输入：推理结果文件
输出：处理后的数据文件
"""

import json
import os
from typing import Dict, List, Any
from pathlib import Path

principle_template_en = """##prompt info 
### Role
You are the world's best data annotator, specializing in distinguishing differences between different responses in the same role-playing scenario.

### Background
You will receive content information including **character background settings**, **dialogue context**, **response candidate 1**, **response candidate 2**, and a set of **evaluation dimensions and principles**. You must compare the two candidate responses strictly following the provided principles.

The format of **dialogue history** is as follows, where "(system)" after the character name represents system output, "(assistant)" represents model output, and "(user)" represents user output:
```markdown
id:1, character_name1(system): (background settings...)
id:2, character_name2(assistant): (dialogue content...)
id:3, character_name3(user): (dialogue content...)
```

The following are rule definitions for dialogue content, which will help you understand and evaluate the dialogue.

### Key Concepts: Dimensions vs Principles

The evaluation schema contains two hierarchical levels:
- Dimensions: broad categories used to organize principles .
- Dimensions are only structural labels and not used directly to compare responses.
- Principles: concrete judgment rules under each dimension.
- Principles are the actual units for comparing cand_1 and cand_2.
- Principles are selected by case, depending on the dialogue context.

Each principle includes:
- definition: the rule you must apply
- level: "sentence" (judged from the current utterance) or "session" (requires full dialogue history)
You must:
- Select only principles relevant to the current dialogue
- Always evaluate negative principles
- Use positive principles only when they meaningfully distinguish the two responses
- You can reference the existing principles and add new principles according to the dialogue context, but you must explain why you add this principle.
- For each chosen principle, compare cand_1 and cand_2 and decide the winner

### Evaluation Process

1. Carefully read the entire dialogue history, sentence by sentence, and understand the full context.
2. Evaluate all negative principles:
- If one response violates any negative principle → the other wins immediately.
- If both responses violate negative principles → output "tie" (both invalid).
- If neither violates negative principles → proceed.
3. Select relevant positive principles only:
- Choose only principles that matter for the current dialogue turn.
- Explain why each principle is relevant.
- If the difference between two responses on a principle is insignificant, do not select that principle.
4. For each selected principle:
- Analyze cand_1 and cand_2 separately using the principle definition.
- Provide evidence from their responses.
- Decide a winner: "cand_1", "cand_2", or "tie".
5. Consider:
- Number of principles each response wins
- The weight and importance of principles
- The degree of difference (slight / clear / significant)
6. Make the final decision: "cand_1", "cand_2", or "tie".


### Principles
Below are the principles from which you need to select the dimensions and principles you think are needed.
<principles>
{principles}
</principles>

### Output Format
- The output must strictly follow the JSON structure below.
- All output text must be in English.
- All double quotes inside text must be replaced with single quotes to avoid parsing errors.

- Return format example
```json
{
  "result": [
    {
      "cand_1": "Response candidate 1 original text",
      "cand_2": "Response candidate 2 original text",
      "principle": {
        "Principle 1": {
          "principle_name": "Name of the selected principle",
          "dimension_name": "Name of the dimension this principle belongs to",
          "principle_level": "sentence or session",
          "main_content": "Definition of the selected principle",
          "reason_for_choosing": "Why this principle is relevant in this context and how it helps distinguish the two responses"
        },
        "Principle 2": {
          "principle_name": "Name of the selected principle",
          "dimension_name": "Name of the dimension this principle belongs to",
          "principle_level": "sentence or session",
          "main_content": "Definition of the selected principle",
          "reason_for_choosing": "Why this principle is relevant and how it highlights differences"
        }
      },
      "analysis": {
        "principle_comparisons": [
          {
            "principle_name": "Principle name",
            "principle_level": "sentence or session",
            "cand_1_performance": "Evidence-based analysis of candidate 1 under this principle",
            "cand_2_performance": "Evidence-based analysis of candidate 2 under this principle",
            "comparison_reason": "Clear comparison and explanation of the degree of difference (slight / clear / significant).",
            "winner": "cand_1" or "cand_2" or "tie"
          }
        ],
        "overall_analysis": "How all principle-level comparisons jointly inform the final judgment.",
        "principle_summary": "Summary of wins, losses, ties, weights, and degrees of difference."
      },
      "better_response": "cand_1" or "cand_2" or "tie"
    }
  ]
}
```

Note:
- The returned content must start with ```json and end with ```.


### Input

**Dialogue Context**  
{input}

**Response Candidate 1**  
{cand_1}

**Response Candidate 2**  
{cand_2}

"""


def clean_response_content(content: str) -> str:
    """清理回复内容, 只保留实际的对话内容"""
    if not content:
        return content
    
    import re
    
    # 去除<system_thinking>...</system_thinking>
    content = re.sub(r'<system_think>.*?</system_think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<system_thinking>.*?</system_thinking>', '', content, flags=re.DOTALL)
    # 去除<long_role_thinking>...</long_role_thinking>
    content = re.sub(r'<long_role_thinking>.*?</long_role_thinking>', '', content, flags=re.DOTALL)
    
    lines = content.strip().split('\n')
    content = '\n'.join(lines).strip()
    return content


def load_jsonl(file_path: str) -> List[Dict]:
    """加载jsonl文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def extract_context_from_messages(messages: List[Dict]) -> str:
    """从messages中提取上下文"""
    context_parts = []
    
    for msg_idx, msg in enumerate(messages):
        role = msg.get('role', '')
        content = msg.get('content', '')
        # 如果是最后一轮且不是user角色, 则不添加到messages中
        if msg_idx == len(messages) - 1 and role != 'user':
            continue
            
        if role == 'system':
            context_parts.append(f"System: {content}")
        elif role == 'user':
            context_parts.append(f"User: {content}")
        elif role == 'assistant':
            context_parts.append(f"Assistant: {content}")

    return "\n\n".join(context_parts)


def construct_inference_data(
    data: List[Dict], 
    choice_indices: tuple, 
    output_file: str, 
    work_dir: str,
    max_samples: int = None
) -> int:
    """
    构造推理数据集
    choice_indices: 使用哪两个候选回复的索引, 例如(0,1)表示使用第1个和第2个回复
    """
    inference_data = []
    skipped_count = 0
    
    print(f"\n处理数据...")
    print(f"总数据量: {len(data)}")
    print(f"使用候选回复: {choice_indices[0]+1} 和 {choice_indices[1]+1}")
    
    # 加载principles/path/to/data/example
    principal = json.load(open(f"/path/to/data/example", 'r', encoding='utf-8'))
    principles = json.dumps(principal, ensure_ascii=False)
    
    for item in data:
        trace_id = item.get('trace_id', '')
        
        # 直接从item中获取数据（推理结果的结构）
        messages = item.get('messages', [])
        model_response = item.get('model_response', {})
        
        # 获取候选回复
        choices = model_response.get('choices', [])
        
        # 确保有足够的候选回复
        if len(choices) <= max(choice_indices):
            skipped_count += 1
            continue
            
        # 提取上下文(对话历史)
        context = extract_context_from_messages(messages)
        
        # 提取指定索引的两个回复
        idx1, idx2 = choice_indices
        response1 = choices[idx1].get('message', {}).get('content', '')
        response2 = choices[idx2].get('message', {}).get('content', '')
        
        if not response1 or not response2:
            skipped_count += 1
            continue
        
        # 清理回复内容
        cand_1 = clean_response_content(response1)
        cand_2 = clean_response_content(response2)
        
        # 使用principle_template构造prompt
        prompt = principle_template_en.replace("{input}", context)
        prompt = prompt.replace("{cand_1}", str(cand_1))
        prompt = prompt.replace("{cand_2}", str(cand_2))
        prompt = prompt.replace("{principles}", str(principles))
        
        # 构造推理数据格式
        inference_item = {
            "trace_id": f"{trace_id}_choices_{choice_indices[0]}_{choice_indices[1]}",
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
            "model_control": item.get("model_control", None),
            "follow_system": True,
            "train_start_index": -1,
            "need_valid": True,
            "raw_record": {
                **item,  # 保留原始数据
                "candidate_1": cand_1,
                "candidate_2": cand_2,
                "metadata": {
                    "choice_indices": choice_indices,
                    "source": "model_response"
                }
            }
        }
        
        inference_data.append(inference_item)
        
        # 如果达到最大样本数, 停止
        if max_samples and len(inference_data) >= max_samples:
            break
    
    # 保存数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in inference_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(inference_data)} 条数据到 {output_file}")
    print(f"⚠️ 跳过 {skipped_count} 条数据（候选回复不足）")
    
    return len(inference_data)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="/path/to/data/example")
    parser.add_argument("--input_file", type=str, default="/path/to/data/example")
    parser.add_argument("--output_dir", type=str, default="/path/to/data/example")
    args = parser.parse_args()
    work_dir = args.work_dir
    input_file = args.input_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("📂 处理推理结果")
    print("=" * 70)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 70)
    
    # 加载推理结果
    print("\n加载推理结果...")
    all_data = load_jsonl(input_file)
    print(f"✅ 加载了 {len(all_data)} 条数据")
    
    # 构造不同候选回复对的数据集
    # 回复组合: 1-2, 3-4, 5-6, 7-8
    choice_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    all_inference_data = []
    
    for pair_idx, choice_indices in enumerate(choice_pairs):
        print(f"\n{'='*70}")
        print(f"处理候选回复对 {choice_indices[0]+1}-{choice_indices[1]+1}...")
        print(f"{'='*70}")
        
        # 为每个组合构造数据
        pair_output_file = f"{output_dir}/inference_data_choices_{choice_indices[0]}_{choice_indices[1]}.jsonl"
        
        pair_count = construct_inference_data(
            all_data, 
            choice_indices, 
            pair_output_file,
            work_dir
        )
        
        # 读取刚保存的数据并加入总集合
        pair_data = load_jsonl(pair_output_file)
        all_inference_data.extend(pair_data)
    
    # 保存包含所有组合的完整数据集
    print(f"\n{'='*70}")
    print("保存完整数据集...")
    print(f"{'='*70}")
    full_output_file = f"{output_dir}/inference_data_all_pairs.jsonl"
    with open(full_output_file, 'w', encoding='utf-8') as f:
        for item in all_inference_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ 完整数据集已保存到: {full_output_file}")
    print(f"✅ 总共生成了 {len(all_inference_data)} 条推理数据")
    
    # 打印统计信息
    print(f"\n{'='*70}")
    print("📊 数据集统计信息")
    print(f"{'='*70}")
    print(f"原始推理结果数: {len(all_data)}")
    print(f"完整数据集: {full_output_file} ({len(all_inference_data)} 条)")
    print("\n各候选回复对数据集:")
    for choice_indices in choice_pairs:
        pair_file = f"{output_dir}/inference_data_choices_{choice_indices[0]}_{choice_indices[1]}.jsonl"
        if os.path.exists(pair_file):
            pair_count = len(load_jsonl(pair_file))
            print(f"  ✅ 候选回复 {choice_indices[0]+1}-{choice_indices[1]+1}: {pair_file} ({pair_count} 条)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

