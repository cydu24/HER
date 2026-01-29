#!/usr/bin/env python3
"""
Step 4.1: Build Setting Completion Inference Data
Step 4.1: 构建 Setting Completion 推理数据

Core Principle (Demand-Driven Enhancement) / 核心原则（需求驱动增强）:
- Enhancement must be demand-driven! / 补充必须是需求驱动的！
- If DIALOGUE shows no demand, don't add / 如果 DIALOGUE 没有显示需求，就不应该补充
- "DIALOGUE shows: N/A" → do not add that field / "DIALOGUE shows: N/A" → 则不补充该字段
- Avoid adding without purpose / 避免无目的地"原文有就加"
"""

import json
import os
from tqdm import tqdm

# Path Configuration / 路径配置
INPUT_PATH = "/path/to/data/example"
OUTPUT_DIR = "/path/to/data/example"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "setting_completion_data.jsonl")

MAX_SAMPLES = None  # None = 全量

PROMPT_TEMPLATE = '''You are an expert at enriching roleplay character settings based on source text analysis.

## Background: System Setting

System Setting provides context BEFORE dialogue begins. Fields:
- scenario: Scene description (shared)
- character_profile: Character traits, appearance, background
- background: Events BEFORE this scene
- motivation: Goals and inner state at SCENE START  
- description: One-sentence role description
- experience: What character is ABOUT TO DO

**CRITICAL**: Setting = INITIAL STATE before dialogue. Cannot contain events during dialogue.

## Task: Setting Enhancement

### Logic Flow (MUST FOLLOW!)

1. Read DIALOGUE → Notice character exhibits trait/behavior/knowledge/emotion
2. Check SETTING → Is this trait/behavior already described?
3. If MISSING in SETTING → Search original_text for related description
4. If FOUND in original_text → Add to setting
5. If NOT in original_text → DO NOT ADD

### CRITICAL: Demand-Driven Enhancement

**Enhancement MUST be driven by dialogue needs!**

- ✅ CORRECT: Dialogue shows character is anxious → Setting lacks anxiety → Text has "hands trembling" → ADD
- ❌ WRONG: Dialogue shows nothing about sleep → But text mentions "didn't sleep" → ADD anyway

**If DIALOGUE shows no need for certain information, DO NOT add it even if it exists in text!**

The purpose of enhancement is to make settings EXPLAIN dialogue behavior, not to dump all text details.

### Data Sources

- **original_text**: The source narrative → CAN supplement from here
- **dialogues**: Events during scene → Use to DISCOVER what needs explaining

## Input Data

```json
INPUT_JSON_PLACEHOLDER
```

## Output Format

```json
{
  "trace_id": "TRACE_ID_PLACEHOLDER",
  "i_c": I_C_PLACEHOLDER,
  "total_characters": TOTAL_CHAR_PLACEHOLDER,
  "scenario_enriched": "...",
  "scenario_reasoning": "...",
  "characters": {
    "CharacterName": {
      "character_index": 1,
      "character_profile_enriched": "...",
      "background_enriched": "...",
      "motivation_enriched": "...",
      "description_enriched": "...",
      "experience_enriched": "...",
      "reasoning": "..."
    }
  }
}
```

## REASONING FORMAT (CRITICAL!)

For each field, you MUST first identify what DIALOGUE reveals:

```
[field_name]:
- DIALOGUE shows: [specific behavior/trait/emotion observed in dialogue]
- SETTING missing: [what the original setting lacks to explain this]
- TEXT source: "[exact quote from original_text]"
- ADDED: [what you supplemented to explain the dialogue behavior]
```

**If dialogue shows NOTHING relevant to a field:**
```
[field_name]: No enhancement needed - dialogue shows no unexplained behavior for this field.
```

**WRONG EXAMPLE (DO NOT DO THIS):**
```
background:
- DIALOGUE shows: N/A
- SETTING missing: sleep status
- TEXT source: "neither had slept"
- ADDED: didn't sleep last night
```
↑ This is WRONG! If dialogue shows nothing, why add sleep info?

**CORRECT EXAMPLE:**
```
background:
- DIALOGUE shows: character appears exhausted, moves slowly
- SETTING missing: reason for exhaustion
- TEXT source: "neither had slept the night before"
- ADDED: sleepless night explains the exhaustion shown in dialogue
```

## Rules

1. **DEMAND-DRIVEN** - Only add if dialogue shows unexplained behavior
   - Ask: "Does this addition help explain something in dialogue?"
   - If NO → Don't add it

2. **SUPPLEMENT FROM original_text ONLY** - Content must come from text

3. **PRESERVE ORIGINAL** - Keep all original setting text, only ADD

4. **NO HALLUCINATION** - Every addition needs TEXT source quote

5. **TIME LOGIC** - Setting is BEFORE dialogue, no dialogue events in setting

6. **EACH CHARACTER SEPARATE** - Own reasoning for each character

7. **JSON FORMAT ONLY**

Now enhance the settings:'''


def format_input_json(sample):
    """构建输入 JSON"""
    trace_id = sample.get('trace_id_book_chapter', '')
    book_name = sample.get('book_name', '')
    text = sample.get('text', '')
    
    conv = sample.get('conversation', [{}])[0]
    i_c = conv.get('i_c', 0)
    scenario = conv.get('scenario', '')
    dialogues = conv.get('dialogues', [])
    
    character_datasets = sample.get('character_datasets', {})
    
    # 构建对话数据
    dialogue_data = []
    for i, dlg in enumerate(dialogues):
        char = dlg.get('character', '')
        sys_thinking = dlg.get('sys_thinking', '')
        enhanced = dlg.get('enhanced_speech', '') or dlg.get('message', '')
        
        turn = {
            "turn_index": i,
            "character": char
        }
        if sys_thinking:
            turn["system_thinking"] = sys_thinking
        if enhanced:
            turn["response"] = enhanced
        dialogue_data.append(turn)
    
    # 构建角色数据 (带索引和字符数)
    characters_data = {}
    char_index = 0
    for char_name, char_data in character_datasets.items():
        if char_name.lower() in ['environment', 'env']:
            continue
        char_index += 1
        
        profile = char_data.get('character_profile', '')
        background = char_data.get('background', '')
        motivation = char_data.get('motivation', '')
        description = char_data.get('description', '')
        experience = char_data.get('experience', '')
        
        characters_data[char_name] = {
            "character_index": char_index,
            "character_profile": profile,
            "character_profile_chars": len(profile),
            "background": background,
            "background_chars": len(background),
            "motivation": motivation,
            "motivation_chars": len(motivation),
            "description": description,
            "description_chars": len(description),
            "experience": experience,
            "experience_chars": len(experience)
        }
    
    total_chars = char_index
    
    input_data = {
        "trace_id": trace_id,
        "i_c": i_c,
        "book_name": book_name,
        "original_text": text,
        "scenario": scenario,
        "scenario_chars": len(scenario),
        "total_characters": total_chars,
        "dialogues": dialogue_data,
        "characters": characters_data
    }
    
    return input_data, trace_id, i_c, list(characters_data.keys()), total_chars


def construct_vulcan_item(sample):
    """构建 Vulcan 推理项"""
    
    input_data, trace_id, i_c, character_names, total_chars = format_input_json(sample)
    
    if not character_names:
        return None
    
    input_json = json.dumps(input_data, indent=2, ensure_ascii=False)
    
    prompt = PROMPT_TEMPLATE.replace('INPUT_JSON_PLACEHOLDER', input_json)
    prompt = prompt.replace('TRACE_ID_PLACEHOLDER', trace_id)
    prompt = prompt.replace('I_C_PLACEHOLDER', str(i_c))
    prompt = prompt.replace('TOTAL_CHAR_PLACEHOLDER', str(total_chars))
    
    vulcan_item = {
        "trace_id": f"setting_{trace_id}_{i_c}",
        "data": [
            {
                "role": "system",
                "text": "You are a helpful assistant. Output valid JSON only."
            },
            {
                "role": "user",
                "text": prompt
            }
        ],
        "model_control": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 8192
        },
        "follow_system": True,
        "train_start_index": 2,
        "need_valid": False,
        "raw_record": {
            "original_trace_id": trace_id,
            "i_c": i_c,
            "character_names": character_names,
            "total_characters": total_chars
        }
    }
    
    return vulcan_item


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Step 4.1: 构建 Setting Completion 推理数据")
    print("=" * 60)
    print("核心原则（需求驱动增强）:")
    print("  - 补充必须是需求驱动的！")
    print("  - 如果 DIALOGUE 没有显示需求，就不应该补充")
    print("  - 避免无目的地'原文有就加'")
    print()
    
    total_samples = 0
    total_items = 0
    total_characters = 0
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="处理中"):
            sample = json.loads(line)
            total_samples += 1
            
            if MAX_SAMPLES and total_samples > MAX_SAMPLES:
                break
            
            vulcan_item = construct_vulcan_item(sample)
            if vulcan_item:
                f_out.write(json.dumps(vulcan_item, ensure_ascii=False) + '\n')
                total_items += 1
                total_characters += len(vulcan_item['raw_record']['character_names'])
    
    print()
    print("=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"处理样本数: {total_samples}")
    print(f"生成推理项: {total_items}")
    print(f"包含角色数: {total_characters}")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

