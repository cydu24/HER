#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构造RM训练数据
从model评估结果中提取数据，构造让RM自己生成principle的训练数据

输入：model_processed_enhanced_result.jsonl（包含model的评估结果）
输出：RM训练数据（不包含principle，让模型自己生成）
"""

import json
import os
import re
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any

# RM的prompt模板（不包含给定的principles，让模型自己生成）
RM_PROMPT_TEMPLATE = """##prompt info 
### Role
You are the world's best data annotator, specializing in distinguishing differences between different responses in the same role-playing scenario.

### Background
You will receive content information including **character background settings**, **dialogue context**, **dialogue continuation**, **response candidate 1**, and **response candidate 2**. You must generate relevant evaluation principles based on the dialogue context and compare the two candidate responses.

The format of **dialogue history** is as follows, where "(system)" after the character name represents system output, "(assistant)" represents model output, and "(user)" represents user output:
```markdown
id:1, npc_name1(system): (background settings...)
id:2, npc_name2(assistant): (dialogue content...)
id:3, npc_name3(user): (dialogue content...)


The user-played character's output is not entirely written by the user. Judge according to the identifier after the character name.
The following are rule definitions for dialogue content, which will help you understand and evaluate the dialogue.

### Key Concepts: Dimensions vs Principles

The evaluation schema contains two hierarchical levels:
- Dimensions: broad categories used to organize principles.
- Dimensions are only structural labels and not used directly to compare responses.
Principles: concrete judgment rules under each dimension.
- Principles are the actual units for comparing cand_1 and cand_2.
- Principles should be generated based on the dialogue context.

Each principle includes:
- definition: the rule you must apply
- level: "sentence" (judged from the current utterance) or "session" (requires full dialogue history)
You must:
- Generate principles relevant to the current dialogue context
- Always evaluate negative principles (if applicable)
- Use positive principles only when they meaningfully distinguish the two responses
- Generate 3-5 principles that best capture the differences
- For each generated principle, compare cand_1 and cand_2 and decide the winner

### Evaluation Process

1. Carefully read the entire dialogue history, sentence by sentence, and understand the full context.
2. Evaluate all negative principles:
- If one response violates any negative principle → the other wins immediately.
- If both responses violate negative principles → output "tie" (both invalid).
- If neither violates negative principles → proceed.
3. Generate relevant positive principles only:
- Create only principles that matter for the current dialogue turn.
- Explain why each principle is relevant.
- If the difference between two responses on a principle is insignificant, do not generate that principle.
4. For each generated principle:
- Analyze cand_1 and cand_2 separately using the principle definition.
- Provide evidence from their responses.
- Decide a winner: "cand_1", "cand_2", or "tie".
5. Consider:
- Number of principles each response wins
- The weight and importance of principles
- The degree of difference (slight / clear / significant)
6. Make the final decision: "cand_1", "cand_2", or "tie".


### Principle Generation Guidelines
You should generate 3-5 evaluation principles that are most relevant for comparing these specific responses in this dialogue context.

Consider two types of dimensions:
- **Negative Dimensions**: Principles that identify flaws or errors in responses
- **Positive Dimensions**: Principles that evaluate quality and excellence in responses 
Generate principles based on what matters most for this specific dialogue context and the actual differences between the two responses.

### Output Format
- The returned result must strictly follow the JSON format below. The output content should be in English and must be placed within English double quotes. If there are English double quotes in the content, they must be replaced with single quotes, otherwise it will cause parsing errors.

- Return format example:
```json
{
  "result": [
    {
      "cand_1": "Response candidate 1 original text",
      "cand_2": "Response candidate 2 original text",
      "principle": {
        "Principle 1": {
          "principle_name": "Name of the principle you generated",
          "dimension_name": "Dimension category of this principle",
          "principle_level": "sentence or session",
          "main_content": "Specific principle description and definition",
          "reason_for_choosing": "Reason for generating this typical principle under **Previous context** and **Dialogue to be Evaluated**, and consider the principle that can show the difference of these two responses"
        },
        "Principle 2": {
          "principle_name": "Name of the principle you generated",
          "dimension_name": "Dimension category of this principle",
          "principle_level": "sentence or session",
          "main_content": "Specific principle description and definition",
          "reason_for_choosing": "Reason for generating this typical principle under **Previous context** and **Dialogue to be Evaluated**, and consider the principle that can show the difference of these two responses"
        },
        "Principle N": {
          "principle_name": "Name of the principle you generated",
          "dimension_name": "Dimension category of this principle",
          "principle_level": "sentence or session",
          "main_content": "Specific principle description and definition",
          "reason_for_choosing": "Reason for generating this typical principle under **Previous context** and **Dialogue to be Evaluated**, and consider the principle that can show the difference of these two responses"
        }
      },
      "analysis": {
        "principle_comparisons": [
          {
            "principle_name": "Principle name",
            "principle_level": "sentence or session",
            "cand_1_performance": "showing specific content in response candidate 1 to support the principle, and analysis how Response 1 performs",
            "cand_2_performance": "showing specific content in response candidate 2 to support the principle, and analysis how Response 2 performs",
            "comparison_reason": "Provide a detailed comparative analysis of response candidate 1 and response candidate 2 under this principle, explaining not only which one performs better but also the degree of difference (e.g., slight, clear, or significant).",
            "winner": "cand_1" or "cand_2" or "tie"
          }
          ...
        ],
        "overall_analysis": "Overall analysis of all principles",
        "principle_summary": "Statistics of wins and losses for each principle, consider the degree of difference of these two responses, and different weights for different principles. e.g.: Cand_1 wins 3 principles (principle_names, principle_weights, degree of difference); Cand_2 wins 1 principle (principle_names, principle_weights, degree of difference); 1 ties"
      },
      "better_response": "cand_1" or "cand_2" or "tie"
    }
  ]
}
```

Note:
- The returned content must start with ```json and end with ```.
- The analysis field is a JSON object containing the principle_comparisons array, overall_analysis, and principle_summary
- The principle_comparisons array needs to include comparison results for all principles

### Input

**Dialogue Context**  
{context}

**Response Candidate 1**  
{cand_1}

**Response Candidate 2**  
{cand_2}

"""


def extract_json_from_response(response_text):
    """从response文本中提取JSON"""
    if not response_text:
        return None
    
    match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


def extract_context_from_raw_record(raw_record: Dict) -> str:
    """从raw_record中提取对话上下文"""
    messages = raw_record.get('messages', [])
    context_parts = []
    
    for msg_idx, msg in enumerate(messages):
        role = msg.get('role', '')
        content = msg.get('content', '')
        # 如果是最后一轮且不是user角色, 则不添加
        if msg_idx == len(messages) - 1 and role != 'user':
            continue
            
        if role == 'system':
            context_parts.append(f"System: {content}")
        elif role == 'user':
            context_parts.append(f"User: {content}")
        elif role == 'assistant':
            context_parts.append(f"Assistant: {content}")
    
    return "\n\n".join(context_parts)


def construct_rm_training_data(
    input_file: str,
    output_file: str,
    skip_invalid: bool = True
) -> int:
    """
    构造RM训练数据
    
    Args:
        input_file: model评估结果文件
        output_file: 输出的RM训练数据文件
        skip_invalid: 是否跳过无效数据（无法解析response的）
    
    Returns:
        成功处理的数据条数
    """
    
    print("=" * 70)
    print("🔨 构造RM训练数据")
    print("=" * 70)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("=" * 70)
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return 0
    
    # 读取model评估结果
    print("\n📖 读取model评估结果...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    
    print(f"总计 {len(lines)} 条数据")
    
    # 处理数据
    print("\n⚙️ 处理数据...")
    rm_training_data = []
    skipped_count = 0
    
    for line in tqdm(lines, desc="处理数据"):
        try:
            data = json.loads(line)
            
            # 提取model的评估结果作为标签
            response_text = data.get('response', '')
            response_json = extract_json_from_response(response_text)
            
            if not response_json or 'result' not in response_json:
                if skip_invalid:
                    skipped_count += 1
                    continue
                else:
                    # 如果不跳过，设置一个空的response
                    response_json = {"result": [{"better_response": "tie"}]}
            
            result = response_json['result'][0] if isinstance(response_json['result'], list) else response_json['result']
            
            # 提取候选回复（从raw_record中）
            raw_record = data.get('raw_record', {})
            cand_1 = raw_record.get('candidate_1', '')
            cand_2 = raw_record.get('candidate_2', '')
            
            if not cand_1 or not cand_2:
                skipped_count += 1
                continue
            
            # 提取对话上下文
            context = extract_context_from_raw_record(raw_record)
            
            if not context:
                skipped_count += 1
                continue
            
            # 构造RM训练数据的input（不包含principle）
            rm_input = RM_PROMPT_TEMPLATE.replace("{context}", context)
            rm_input = rm_input.replace("{cand_1}", cand_1)
            rm_input = rm_input.replace("{cand_2}", cand_2)
            
            # 构造RM训练数据的output（model的完整评估结果）
            rm_output = f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```"
            
            # 构造训练数据格式
            training_item = {
                "trace_id": data.get('trace_id', ''),
                "messages": [
                    {
                        "role": "user",
                        "content": rm_input
                    },
                    {
                        "role": "assistant",
                        "content": rm_output
                    }
                ],
                "metadata": {
                    "better_response": result.get('better_response', 'tie'),
                    "num_principles": len(result.get('principle', {})),
                    "source": "model_evaluation"
                }
            }
            
            rm_training_data.append(training_item)
            
        except Exception as e:
            skipped_count += 1
            continue
    
    # 保存数据
    print(f"\n💾 保存RM训练数据...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in rm_training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 统计信息
    print("\n" + "=" * 70)
    print("📊 处理统计")
    print("=" * 70)
    print(f"✅ 成功处理: {len(rm_training_data)} 条")
    print(f"⚠️ 跳过: {skipped_count} 条")
    print(f"📁 输出文件: {output_file}")
    print("=" * 70)
    
    return len(rm_training_data)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="构造RM训练数据")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="model评估结果文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出的RM训练数据文件路径"
    )
    parser.add_argument(
        "--skip_invalid",
        action="store_true",
        default=True,
        help="是否跳过无效数据"
    )
    
    args = parser.parse_args()
    
    construct_rm_training_data(
        input_file=args.input_file,
        output_file=args.output_file,
        skip_invalid=args.skip_invalid
    )


if __name__ == "__main__":
    main()


# 使用示例
"""
# 示例1: 处理SFT数据
python construct_rm_training_data.py \
    --input_file /path/to/data/example \
    --output_file /path/to/data/example

# 示例2: 处理RL数据
python construct_rm_training_data.py \
    --input_file /path/to/data/example \
    --output_file /path/to/data/example

# 示例3: 处理Test数据
python construct_rm_training_data.py \
    --input_file /path/to/data/example \
    --output_file /path/to/data/example
"""

