#!/usr/bin/env python3
"""
Step 3.7: 构造 Conversation 级别的逻辑一致性改写数据

功能：
1. 读取 sft_data_enhanced.jsonl (带有 role_thinking 增强)
2. 读取 all_success_final.jsonl (sys_thinking 生成结果)
3. 按 (trace_id, character) 聚合所有轮次（包括没有 sys_thinking 的）
4. 用 JSON 格式清晰展示每轮数据，让模型改写/补全 sys_thinking

输入:
- sft_data_enhanced.jsonl: 带有增强 role_thinking 的 SFT 数据
- all_success_final.jsonl: 生成的 sys_thinking 结果

输出:
- main/rewrite_data/rewrite_consistency.jsonl (推理格式，conversation 级别)
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse
import re


# Conversation 级别逻辑一致性改写 Prompt 模板
CONSISTENCY_REWRITE_PROMPT_TEMPLATE = """You are a professional role-playing dialogue consistency editor. Your task is to revise the **sys_thinking** (system planning) to align with the actual **enhanced_speech** output.

## What is sys_thinking?

**sys_thinking** is the model's internal planning BEFORE generating each response:
- Written from the **MODEL's perspective (third-person about the character)**, NOT the character's first-person voice
- CORRECT: "I need to play the role of {character_name}...", "My character should express nervousness...", "The scene requires me to..."
- WRONG: "I can feel him standing there..." (this is character's first-person - belongs in role_thinking, NOT sys_thinking!)
- Reference the input sys_thinking style for diversity - keep similar format/structure
- It plans HOW to respond, analyzing context and deciding the approach
- It must logically lead to the **enhanced_speech** output (the role_thinking, role_action, and speech)

**CRITICAL DISTINCTION:**
- `sys_thinking`: Model's planning voice - "I need to portray {character_name} as nervous because..."
- `role_thinking`: Character's inner voice - "I can feel him watching me..." (this is in enhanced_speech, NOT sys_thinking!)

**CRITICAL - NEVER use "user" in sys_thinking:**
- NEVER say "The user", "user is", "user wants", "user input", "the user (Name)"
- ALWAYS refer to other characters by their names: "Miles said...", "Sarah responded..."
- WRONG: "The user (Miles) provided input..." 
- CORRECT: "Miles responded with..."
- This is an immersive roleplay - there are no "users", only characters

## Visibility Rules (CRITICAL - What the model can see when generating response)

**For the current character ({character_name})'s previous turns:**
- CAN see: Own previous `<role_thinking>` (first-person inner thoughts)
- CAN see: Own previous `<role_action>` (actions)  
- CAN see: Own previous speech (dialogue)

**For other characters' turns:**
- CANNOT see: Their `<role_thinking>` (hidden inner thoughts - this is private!)
- CAN see: Their `<role_action>` (visible actions)
- CAN see: Their speech (dialogue)

**Important**: sys_thinking is planning for the NEXT response only. The model cannot see any sys_thinking from previous turns - it only sees the conversation history.

## Complete Dialogue Data (JSON format)

The JSON array starts with a **system_info** entry containing character context, followed by dialogue turns.

**Structure:**
- First entry (`"role": "system_info"`): Character's system prompt and other character profiles - USE THIS for context!
- Subsequent entries: Dialogue turns with `dialogue_index` 0, 1, 2, ...
  - `sys_thinking`: System planning BEFORE response (only for {character_name}'s turns) - this is what you need to revise
  - `enhanced_speech`: The actual response AFTER planning - this is the target to align to
  - `need_revise`: true = needs revision, false = context only

**Logical flow**: sys_thinking (planning) → leads to → enhanced_speech (output)

**You only need to revise turns where `need_revise: true`** (turns from {character_name}).

```json
{full_dialogue_json}
```

## Your Task

For each turn, revise **sys_thinking** to **ALIGN with enhanced_speech**:

**FIRST: Identify the original sys_thinking type:**

**Type A - Correct Third-Person Format** (starts with "I need to portray...", "My character is...", "Context:", "Goal:", etc.):
→ PRESERVE the exact structure, length (±10%), and format
→ Only revise CONTENT to align with enhanced_speech
→ Keep all sections (Character, Context, Goal, Action, Plan, Drafting, etc.)
→ ⚠️ CHECK CHARACTER COUNT: If original is ~2000 chars, output MUST be ~2000 chars (not 1000!)

**Type B - Wrong First-Person Format** (starts with "I feel...", "I am hungry...", character's voice):
→ REWRITE completely in third-person model perspective
→ Generate proper analysis structure: Context → Goal → Plan → Drafting
→ Do NOT follow the original format (it's wrong!)

**Type C - Empty `[EMPTY]`**:
→ GENERATE new sys_thinking from scratch
→ Use proper third-person model perspective
→ Analyze scenario, character motivation, and plan the response

**For all types, align with enhanced_speech:**
- sys_thinking must logically lead to the role_thinking, role_action, and speech in enhanced_speech

⚠️ **CRITICAL - Strict third-person perspective!**
- NEVER write "The user (playing X)..." or "The user wants..."
- **Model's voice** (planning): "I need to portray {character_name} as...", "I am playing {character_name}...", "I should show..."
- **Character analysis** (NOT first-person!): "{character_name} wants...", "{character_name} feels...", "The character needs to..."
- ❌ WRONG: "I want to see her" (sounds like character speaking)
- ✅ RIGHT: "I need to portray Jonah's desire to see her" or "Jonah wants to see her"
- Reference other characters by NAME: "Miles responded...", not "The user said..."

**For the FIRST turn of the character**: Thoroughly analyze:
- The scenario/scene setup from system_info
- The character's background and motivation
- How to begin the roleplay appropriately

## Output Format - STRICT JSON ONLY

⚠️ **CRITICAL: Output ONLY a valid JSON array. NO explanations, NO markdown headers, NO "Here's how I'm thinking..." - JUST the JSON array!**

```json
[
  {{
    "dialogue_index": 0,
    "revised_sys_thinking": "I need to portray {character_name} as... [KEEP SAME LENGTH AS ORIGINAL]",
    "revision_notes": "Aligned X with enhanced_speech Y"
  }},
  {{
    "dialogue_index": 2,
    "revised_sys_thinking": "...",
    "revision_notes": "..."
  }}
]
```

**REQUIREMENTS:**
1. Output EXACTLY {num_turns} entries in the JSON array
2. Use EXACTLY these field names: `dialogue_index`, `revised_sys_thinking`, `revision_notes`
3. `dialogue_index` must match the original values from the input
4. **For Type A (correct third-person format):**
   - PRESERVE LENGTH (±10%) and STRUCTURE exactly
   - ⚠️ If `target_chars` is specified (e.g., "~2500 chars"), your output MUST be that length!
   - Do NOT compress or summarize - expand if needed to match target length
5. **For Type B (wrong first-person) or Type C (empty):**
   - Generate proper third-person analysis (~800-1500 chars)
   - Include: Context analysis → Character goal → Action plan → Response drafting
6. In `revision_notes`, indicate: "Type A: preserved format" or "Type B: rewrote from first-person" or "Type C: generated new"
7. NO text before or after the JSON array
"""


def load_sys_thinking_aggregated(aggregated_path: str) -> dict:
    """
    加载已聚合的 sys_thinking 数据
    
    Returns:
        Dict[(original_trace_id, character_name) -> {assistant_index -> sys_thinking}]
    """
    results = {}
    
    print(f"加载聚合的 sys_thinking: {aggregated_path}")
    with open(aggregated_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载 sys_thinking"):
            try:
                data = json.loads(line)
                original_trace_id = data.get('original_trace_id', '')
                character_name = data.get('character_name', '')
                
                if not original_trace_id or not character_name:
                    continue
                
                key = (original_trace_id, character_name)
                results[key] = {}
                
                for turn in data.get('turns', []):
                    assistant_index = turn.get('assistant_index')
                    sys_thinking = turn.get('model_thinking', '')
                    
                    # 清理
                    if sys_thinking:
                        sys_thinking = re.sub(r'</?(think|thinker)>', '', sys_thinking)
                    
                    if assistant_index is not None:
                        results[key][assistant_index] = sys_thinking
                
            except Exception as e:
                continue
    
    print(f"加载了 {len(results)} 个 training sample 的 sys_thinking")
    return results


def load_and_aggregate_sft_data(sft_path: str, sys_thinking_map: dict) -> dict:
    """
    加载 SFT 数据并按 (trace_id, character, i_c) 聚合
    
    数据来源：
    - enhanced_speech: 从 sft_data_enhanced_v3.jsonl
    - sys_thinking: 从 aggregated_by_training_sample.jsonl (通过 sys_thinking_map)
    
    sys_thinking_map 结构: Dict[(original_trace_id, character_name) -> {assistant_index -> sys_thinking}]
    """
    aggregated = defaultdict(lambda: {
        'trace_id': '',
        'character_name': '',
        'book_name': '',
        'chapter': '',
        'scenario': '',
        'i_c': 0,
        'turns': [],
        'full_dialogues': [],
        'system_prompt': '',  # 角色的 system prompt
        'character_profile': '',  # 角色详细 profile（完整版）
        'background': '',  # 故事背景
        'motivation': '',  # 角色内心想法/动机
        'other_character_profiles': {},  # 其他角色 profiles
        'output_format': ''  # 输出格式要求
    })
    
    print(f"加载 SFT 数据: {sft_path}")
    
    total_turns = 0
    turns_with_sys = 0
    turns_without_sys = 0
    
    with open(sft_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载 SFT 数据"):
            try:
                sample = json.loads(line)
                trace_id = sample.get('trace_id_book_chapter', '')
                book_name = sample.get('book_name', '')
                chapter = sample.get('chapter', '')
                
                # 提取 training_samples 和 character_datasets
                training_samples = sample.get('training_samples', {})
                character_datasets = sample.get('character_datasets', {})
                
                for i_c, conversation in enumerate(sample.get('conversation', [])):
                    scenario = conversation.get('scenario', '')
                    dialogues = conversation.get('dialogues', [])
                    major_chars = conversation.get('major_characters', [])
                    
                    if not major_chars:
                        continue
                    
                    # 每个角色单独计数 assistant_index
                    char_assistant_index = defaultdict(int)
                    
                    for d_idx, dialogue in enumerate(dialogues):
                        character = dialogue.get('character', '')
                        
                        if character == 'Environment':
                            continue
                        
                        if character in major_chars:
                            char_assistant_index[character] += 1
                            assistant_index = char_assistant_index[character]
                            total_turns += 1
                            
                            # 获取 enhanced_speech（从 sft_data_enhanced_v3.jsonl）
                            enhanced_format = dialogue.get('enhanced_standard_format', '')
                            if not enhanced_format:
                                continue
                            
                            # 获取 sys_thinking（从 aggregated_by_training_sample.jsonl）
                            # sys_thinking_map: (trace_id, character) -> {assistant_index -> sys_thinking}
                            char_sys_map = sys_thinking_map.get((trace_id, character), {})
                            sys_thinking = char_sys_map.get(assistant_index, '')
                            
                            if sys_thinking:
                                turns_with_sys += 1
                            else:
                                turns_without_sys += 1
                            
                            agg_key = (trace_id, character, i_c)
                            
                            # 设置基本信息
                            if not aggregated[agg_key]['trace_id']:
                                aggregated[agg_key]['trace_id'] = trace_id
                                aggregated[agg_key]['character_name'] = character
                                aggregated[agg_key]['book_name'] = book_name
                                aggregated[agg_key]['chapter'] = chapter
                                aggregated[agg_key]['scenario'] = scenario
                                aggregated[agg_key]['i_c'] = i_c
                                
                                # 提取 system prompt（从 training_samples）
                                char_samples = training_samples.get(character, [])
                                if char_samples and char_samples[0].get('role') == 'system':
                                    aggregated[agg_key]['system_prompt'] = char_samples[0].get('content', '')
                                
                                # 提取 character_datasets 完整信息
                                char_dataset = character_datasets.get(character, {})
                                aggregated[agg_key]['character_profile'] = char_dataset.get('character_profile', '')
                                aggregated[agg_key]['background'] = char_dataset.get('background', '')
                                aggregated[agg_key]['motivation'] = char_dataset.get('motivation', '')
                                aggregated[agg_key]['other_character_profiles'] = char_dataset.get('other_character_profiles', {})
                                aggregated[agg_key]['output_format'] = char_dataset.get('output_format', '')
                                
                                # 保存完整对话（所有角色的 enhanced_speech）
                                aggregated[agg_key]['full_dialogues'] = [
                                    {
                                        'dialogue_index': idx,
                                        'character': d.get('character', ''),
                                        'enhanced_speech': d.get('enhanced_standard_format', '') or d.get('content', '')
                                    }
                                    for idx, d in enumerate(dialogues)
                                ]
                            
                            # 添加当前角色的 turn
                            aggregated[agg_key]['turns'].append({
                                'dialogue_index': d_idx,
                                'assistant_index': assistant_index,
                                'sys_thinking': sys_thinking,
                                'enhanced_speech': enhanced_format
                            })
                            
            except Exception as e:
                continue
    
    # 按 dialogue_index 排序
    for agg_key in aggregated:
        aggregated[agg_key]['turns'].sort(key=lambda x: x['dialogue_index'])
    
    print(f"聚合得到 {len(aggregated)} 个 training sample")
    print(f"总轮次: {total_turns}, 有 sys_thinking: {turns_with_sys}, 无 sys_thinking: {turns_without_sys}")
    
    return aggregated


def construct_vulcan_item(agg_data: dict, max_input_tokens: int = 30000) -> dict:
    """构造单个 Vulcan 数据项（conversation 级别）"""
    
    trace_id = agg_data['trace_id']
    character_name = agg_data['character_name']
    book_name = agg_data['book_name']
    chapter = agg_data['chapter']
    scenario = agg_data['scenario']
    i_c = agg_data['i_c']
    turns = agg_data['turns']
    full_dialogues = agg_data.get('full_dialogues', [])
    system_prompt = agg_data.get('system_prompt', '')
    character_profile = agg_data.get('character_profile', '')
    background = agg_data.get('background', '')
    motivation = agg_data.get('motivation', '')
    other_character_profiles = agg_data.get('other_character_profiles', {})
    
    if not turns:
        return None
    
    # 创建当前角色 turns 的查找表（dialogue_index -> turn data）
    current_char_turns = {t['dialogue_index']: t for t in turns}
    
    # 构建 other_profiles_dict（用于 JSON）
    other_profiles_dict = {}
    for other_char, profile in other_character_profiles.items():
        if other_char != character_name and profile:
            # 截取 profile 前 500 字符避免过长
            other_profiles_dict[other_char] = profile[:500] + "..." if len(profile) > 500 else profile
    
    # 构建完整对话 JSON（所有角色，标记 need_revise）
    full_dialogue_for_json = []
    
    # 第一个元素：system_info（完整角色设定和上下文）
    system_info = {
        'role': 'system_info',
        'character_name': character_name,
        'book': book_name,
        'chapter': chapter,
        'scenario': scenario,  # 当前场景
    }
    
    # 添加详细的角色 profile（如果有）
    if character_profile:
        # 截取避免过长
        system_info['character_profile'] = character_profile[:1500] + "..." if len(character_profile) > 1500 else character_profile
    
    # 添加背景（如果有）
    if background:
        system_info['background'] = background
    
    # 添加角色动机/内心想法（如果有）
    if motivation:
        system_info['motivation'] = motivation
    
    # 添加其他角色信息
    if other_profiles_dict:
        system_info['other_character_profiles'] = other_profiles_dict
    
    # 添加原始 system_prompt（作为参考）
    if system_prompt:
        system_info['original_system_prompt'] = system_prompt
    
    full_dialogue_for_json.append(system_info)
    
    num_need_revise = 0
    
    for d in full_dialogues:
        d_idx = d.get('dialogue_index')
        char = d.get('character', '')
        speech = d.get('enhanced_speech', '')
        
        if not speech:
            continue
        
        # 检查是否是当前角色的 turn（需要修改）
        if d_idx in current_char_turns:
            turn_data = current_char_turns[d_idx]
            sys_thinking = turn_data.get('sys_thinking', '') or ''
            sys_len = len(sys_thinking) if sys_thinking else 0
            # 顺序：sys_thinking（规划）在前，enhanced_speech（输出）在后
            turn_entry = {
                'dialogue_index': d_idx,
                'character': char,
                'sys_thinking': sys_thinking if sys_thinking else "[EMPTY - needs generation]",
                'enhanced_speech': speech,
                'need_revise': True
            }
            # 如果原始 sys_thinking 较长，加入目标字符数提示
            if sys_len > 1000:
                turn_entry['target_chars'] = f"~{sys_len} chars (MUST match original length!)"
            full_dialogue_for_json.append(turn_entry)
            num_need_revise += 1
        else:
            # 其他角色的 turn，只提供上下文（无 sys_thinking）
            full_dialogue_for_json.append({
                'dialogue_index': d_idx,
                'character': char,
                'enhanced_speech': speech,
                'need_revise': False
            })
    
    full_dialogue_json = json.dumps(full_dialogue_for_json, indent=2, ensure_ascii=False)
    num_turns = num_need_revise
    
    # 构造 prompt（system_info 已经在 JSON 里了）
    prompt = CONSISTENCY_REWRITE_PROMPT_TEMPLATE.format(
        character_name=character_name,
        full_dialogue_json=full_dialogue_json,
        num_turns=num_turns
    )
    
    # 估算 token 数
    estimated_tokens = len(prompt) / 4
    if estimated_tokens > max_input_tokens:
        return None
    
    # 生成唯一 trace_id
    safe_char_name = character_name.replace(' ', '_').replace('/', '_')
    vulcan_trace_id = f"consistency_{trace_id}/{safe_char_name}/{i_c:04d}"
    
    vulcan_item = {
        "trace_id": vulcan_trace_id,
        "data": [
            {"role": "system", "text": "You are a JSON-only assistant. Output ONLY valid JSON arrays, with NO explanations, NO markdown headers, NO 'Here is...' phrases. Start directly with [ and end with ].", "name": "system"},
            {"role": "user", "text": prompt, "name": "user"}
        ],
        "model_control": None,
        "follow_system": True,
        "train_start_index": -1,
        "need_valid": True,
        "raw_record": {
            "original_trace_id": trace_id,
            "character_name": character_name,
            "book_name": book_name,
            "chapter": chapter,
            "i_c": i_c,
            "num_turns": num_turns,
            "dialogue_indices": [t['dialogue_index'] for t in turns],
            "turns": turns  # 保存原始轮次数据
        }
    }
    
    return vulcan_item


def main():
    parser = argparse.ArgumentParser(description='Step 3.7: 构造 Conversation 级别逻辑一致性改写数据')
    parser.add_argument('--sft_input', type=str,
                        default='/path/to/data/example',
                        help='SFT 增强数据输入')
    parser.add_argument('--sys_thinking_input', type=str,
                        default='/path/to/data/example',
                        help='sys_thinking 生成结果输入')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/data/example',
                        help='输出目录')
    parser.add_argument('--max_turns', type=int, default=30,
                        help='每个对话最大轮数')
    parser.add_argument('--max_input_tokens', type=int, default=30000,
                        help='最大输入 token 数')
    parser.add_argument('--limit', type=int, default=0,
                        help='限制输出条数，0 表示不限制')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Step 3.7: 构造 Conversation 级别逻辑一致性改写数据")
    print("=" * 60)
    
    # 加载聚合的 sys_thinking（从 aggregated_by_training_sample.jsonl）
    sys_thinking_map = load_sys_thinking_aggregated(args.sys_thinking_input)
    
    # 加载并聚合 SFT 数据
    aggregated_data = load_and_aggregate_sft_data(args.sft_input, sys_thinking_map)
    
    # 构造 Vulcan 数据
    output_path = os.path.join(args.output_dir, 'rewrite_consistency.jsonl')
    
    total_convs = 0
    valid_count = 0
    skipped_too_long = 0
    skipped_too_many_turns = 0
    skipped_no_turns = 0
    total_turns = 0
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for agg_key, agg_data in tqdm(aggregated_data.items(), desc="构造 Vulcan 数据"):
            # 检查是否达到 limit
            if args.limit > 0 and valid_count >= args.limit:
                break
            
            total_convs += 1
            turns = agg_data['turns']
            
            if not turns:
                skipped_no_turns += 1
                continue
            
            if len(turns) > args.max_turns:
                skipped_too_many_turns += 1
                continue
            
            vulcan_item = construct_vulcan_item(agg_data, args.max_input_tokens)
            
            if vulcan_item:
                f_out.write(json.dumps(vulcan_item, ensure_ascii=False) + '\n')
                valid_count += 1
                total_turns += len(turns)
            else:
                skipped_too_long += 1
    
    print()
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"总 (trace_id, character, i_c) 组合: {total_convs:,}")
    print(f"有效输出: {valid_count:,} ({valid_count/total_convs*100:.1f}%)")
    print(f"总轮次数: {total_turns:,}")
    print(f"平均轮次/对话: {total_turns/valid_count:.1f}" if valid_count > 0 else "N/A")
    print(f"跳过 (无轮次): {skipped_no_turns:,}")
    print(f"跳过 (轮次过多 > {args.max_turns}): {skipped_too_many_turns:,}")
    print(f"跳过 (token 过多): {skipped_too_long:,}")
    print(f"\n输出文件: {output_path}")


if __name__ == '__main__':
    main()
