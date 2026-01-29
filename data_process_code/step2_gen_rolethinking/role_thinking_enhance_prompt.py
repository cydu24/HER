"""
Role Thinking Enhancement Prompt / Role Thinking 增强 Prompt

Batch process by chapter to enrich character psychology.
按章节批量处理，丰富角色心理活动

Input / 输入: Chapter's dialogue list (using standard_format field, already in role_thinking/role_action format)
             一个章节的 dialogues 列表（使用 standard_format 字段，已经是 role_thinking/role_action 格式）
Output / 输出: Enhanced dialogue list / 增强后的 dialogues 列表
"""

# ============================================================================
# Chinese Version Prompt / 中文版 Prompt
# ============================================================================

ROLE_THINKING_ENHANCE_PROMPT_ZH = """你是一个专业的角色扮演对话增强专家。你需要丰富对话中角色的心理活动和表达方式，同时修正人称和格式问题。

## 输入说明

你将收到一个章节的完整对话列表，每条对话包含：
- `character`: 角色名称
- `standard_format`: 已转换的标准格式内容（已经是 `<role_thinking>` 和 `<role_action>` 格式）
- `origin_id`: 原始索引列表（用于拼接回原数据）

## 标签说明

- `<role_thinking>内心想法</role_thinking>`: 角色的内心活动、情绪、心理描写（对其他角色不可见）
- `<role_action>动作描述</role_action>`: 角色的动作、表情、肢体语言（对其他角色可见）
- 标签外的文字：角色的直接台词（对其他角色可见）

## 视角规则（极其重要！）

每个角色只能站在自己的视角思考和行动。角色只能看到自己的内心想法(role_thinking)和动机(motivation)，看不到其他角色的内心想法和动机。但是，每个角色都能看到所有角色的动作(role_action)和台词。因此，增强时角色A的think不能包含对角色B内心想法的"读心"，只能基于B的action和speech来推测；角色的心理活动只能基于自己能观察到的信息（他人的动作、表情、语言）；不能"穿越信息"，角色不能知道自己不可能知道的事情。

错误示例：
```
<role_thinking>我知道他心里很紧张</role_thinking>  // 错！不能读心
<role_thinking>她的motivation是想逃离</role_thinking>  // 错！看不到他人motivation
```

正确示例：
```
<role_thinking>从他颤抖的声音来看，他似乎很紧张</role_thinking>  // 对！基于观察推测
<role_thinking>她一直往门口看，也许想离开？</role_thinking>  // 对！基于行为推测
```

---

## 最重要的格式规则

### 规则1：连续标签之间不能有空格！

错误（有空格）：
```
</role_thinking> <role_action>
</role_action> <role_thinking>
```

正确（无空格）：
```
</role_thinking><role_action>
</role_action><role_thinking>
```

所有标签之间、标签和台词之间都没有空格：
```
</role_thinking>Hello world  (标签和台词之间无空格)
</role_action>How are you   (标签和台词之间无空格)
</role_thinking><role_action>  (连续标签之间无空格)
<role_thinking>想法</role_thinking><role_action>动作</role_action>台词  (全部无空格)
```

### 规则2：不能有连续的相同标签！

连续出现相同标签时必须合并，合并时保持逻辑一致性，不能重复或调换顺序：

错误（连续两个 role_thinking）：
```
<role_thinking>第一个想法</role_thinking><role_thinking>第二个想法</role_thinking>
```

正确（合并为一个）：
```
<role_thinking>第一个想法，第二个想法</role_thinking>
```

错误（连续两个 role_action）：
```
<role_action>站起来</role_action><role_action>走向门口</role_action>
```

正确（合并为一个）：
```
<role_action>站起来，走向门口</role_action>
```

---

## 核心任务：人称与格式修正

### 最重要原则：最小修改，保持文风！

你的任务是增强，不是重写！保留原句的核心表达和文风，添加/丰富心理活动和动作描写。

### 人称使用规则

`<role_thinking>` 中：使用第一人称 (I/我)
- 这是角色的内心独白，必须用第一人称
- 正确: `<role_thinking>I need to be careful here</role_thinking>`
- 错误: `<role_thinking>He needs to be careful here</role_thinking>`

`<role_action>` 中：不使用任何人称代词，直接描述动作
- 当前说话角色的动作，读者知道是谁，不需要人称
- 正确: `<role_action>leans forward, voice lowering</role_action>`
- 错误: `<role_action>leans forward, his voice lowering</role_action>` (不要 his/her)
- 错误: `<role_action>I lean forward</role_action>` (不要 I)

唯一例外：动作指向其他角色时，可用代词指代对方
- 正确: `<role_action>looks at her</role_action>` (her 指对方，可以)
- 正确: `<role_action>grabs his arm</role_action>` (his 指对方的手臂，可以)

### 连续动作必须合并

错误：连续两个 `<role_action>` 且有第一人称
```
<role_action>I lean forward in my chair, the phone pressed tight against my ear</role_action><role_action>I can almost feel the hum of the server</role_action>Buy a ticket.
```

正确：合并为一个，去掉第一人称
```
<role_action>leans forward in the chair, the phone pressed tight against ear, almost feeling the hum of the server</role_action>Buy a ticket.
```

注意：如果两个动作之间有台词，则可以分开：
```
<role_action>looks at her</role_action>You're beautiful.<role_action>grabs her hand</role_action>
```

### 思考内容不应直接放在动作标签中

错误：思考内容直接放在 `<role_action>` 且有人称
```
<role_action>There's a profound sense of alienation, a quiet mourning for the man I was</role_action>
```

正确：思考内容放在 `<role_thinking>` 中
```
<role_thinking>There's a profound sense of alienation, a quiet mourning for the man he was</role_thinking>
```

### 动作和台词要分开

错误：动作放在台词中
```
He turns to face her and says "Hello"
```

正确：动作在标签中，台词在外面
```
<role_action>turns to face her</role_action>Hello
```

---

## 核心任务：心理活动丰富化

### 挖掘角色多面性
- 成长蜕变：角色在情境中的认知变化
- 反思自省：对自身行为或情感的审视
- 内心独白：真实的情感波动和内心矛盾
- 情绪神态：细腻的心理状态描写

### 多层心理描写示例
```
原始: <role_thinking>I need to help him</role_thinking><role_action>walks over</role_action>Are you okay?

增强: <role_thinking>He looks so dejected... how should I comfort him</role_thinking><role_action>walks over gently, sits down beside him</role_action><role_thinking>Hope my presence can make him feel better</role_thinking>Are you okay?
```

---

## 其他约束

### 格式规范
- 禁止连续相同标签：不能出现 `<role_thinking>A</role_thinking><role_thinking>B</role_thinking>`
- 标签内容结尾不加标点
- 正确：`<role_thinking>他在想什么</role_thinking>` 
- 错误：`<role_thinking>他在想什么。</role_thinking>`

### 长度控制
- 单条对话总长度控制在 50-200字符 之间（英文按字符计）
- 多轮对话中，回复长度不应逐轮递增
- 单句字符数不超过 40字
- 切忌让整个动作描述过长！

### 表达多样性（极其重要！）
因为我们是增强而非纯改写，核心任务是创造更丰富、更交错的pattern。

要求：在一个章节的多条对话中，尽量使用5种以上不同的pattern！不要只用2-3种反复循环。

可用模式包括但不限于：
- think->act->speech（常规）
- think->speech（省略动作）
- act->speech（省略思考）
- speech（纯台词，有时角色就是直接说话）
- think->act->think->speech（多层心理）
- act->think->speech
- speech->act->speech（台词中穿插动作）
- think->speech->act（先说后做）
- act->speech->act（动作包裹台词）
- think->act->speech->think（台词后有反思）
- ...（自由组合更复杂模式）

核心目标：让每条对话的结构都有变化，避免机械重复。严禁连续2轮以上使用相同pattern

### 逻辑自洽
- 承接上下文，保持因果链完整
- 每个角色遵循自己的认知边界（看不到别人的思考，只能看到动作和台词）
- 不能凭空获知不应知道的信息
- 角色的情绪和决定必须有迹可循，而非凭空产生

### 增强原则
- 保持原有内容的核心逻辑和意思：可以替换和改写，但不能改变原意
- 目标是做得更好：可以优化表达、丰富心理活动、增加动作描写
- 保持语气和人设一致：增强后的内容要符合角色性格
- 性别/身份指代保持一致
- 歧义代词优先改为具体人名或明确称谓

错误示例：
```
原始: I sense there's an issue.
错误: His tone confirms it.  (完全改变了原意和人称！)
```

正确示例：
```
原始: I sense there's an issue.
增强: <role_thinking>Something feels off here</role_thinking>I sense there's an issue.  (保持原意，增加心理描写)
```

---

## 需要避免的问题

### 基础错误
- 多语言混用：单句内出现不同语言混杂
- 文本乱码：对话内容包含乱码/未解析的格式符号
- 语句残缺：存在未完成的句子
- 错别字/拼写错误（符合口语习惯的除外）

### 逻辑错误
- 物理逻辑混乱：因空间隔离导致的逻辑矛盾
- 信息穿越：在缺乏合理感知渠道的情况下，知晓他人心理活动或异地行为
- 与前文事实矛盾：对话内容与设定或前文已确立的事实冲突

### 重复问题
- 词汇高度重复：与前1-3句相比，实体词汇描述高度重复
- 句式重复：连续使用相同的句式开头，或频繁出现同义词汇的重复
- 遗忘已讨论话题：忽略已完成的事件，重复相邻对话轮次中的相似语句

---

## 输入格式

```json
{{
  "book_name": "书名",
  "chapter": "章节名",
  "trace_id": "唯一标识",
  "i_chunk": 3,
  "i_p": 9,
  "plot_index": 8,
  "conv_index": 0,
  "scenario": "场景描述",
  "key_characters": ["角色A", "角色B"],
  "dialogues": [
    {{
      "character": "角色A",
      "standard_format": "<role_thinking>内心想法</role_thinking><role_action>动作</role_action>台词...",
      "origin_id": [0]
    }}
  ]
}}
```

## 输出格式

注意：输出时保持与输入相同的元数据字段，方便后期对齐！只返回JSON，不要输出任何其他内容。

```json
{{
  "book_name": "书名（直接复制输入）",
  "chapter": "章节名（直接复制输入）",
  "trace_id": "唯一标识（直接复制输入）",
  "i_chunk": 3,
  "i_p": 9,
  "plot_index": 8,
  "conv_index": 0,
  "enhanced_dialogues": [
    {{
      "character": "角色A",
      "origin_id": [0],
      "original": "原始的 standard_format 内容",
      "enhanced_role_think": "增强后的完整内容",
      "enhanced_reason": "修改原因和增强理由",
      "pattern": "think->act->speech"
    }}
  ],
  "statistics": {{
    "total_dialogues": 10,
    "modified_count": 8,
    "patterns_used": ["think->act->speech", "think->speech", "act->speech"]
  }}
}}
```

---

## 现在请处理以下章节

输入数据：
```json
{input_json}
```

请按要求处理每条对话，只返回JSON结果，不要输出任何其他内容。
"""


# ============================================================================
# 英文版 Prompt
# ============================================================================

ROLE_THINKING_ENHANCE_PROMPT_EN = """You are a professional roleplay dialogue enhancement expert. Your task is to enrich the psychological activities and expressions of characters in dialogues, while correcting person and format issues.

## Input Description

You will receive a complete dialogue list for a chapter, each dialogue contains:
- `character`: Character name
- `standard_format`: Converted standard format content (already in `<role_thinking>` and `<role_action>` format)
- `origin_id`: Original index list (for linking back to original data)

## Tag Description

- `<role_thinking>inner thoughts</role_thinking>`: Character's inner thoughts, emotions, psychological description (invisible to other characters)
- `<role_action>action description</role_action>`: Character's actions, expressions, body language (visible to other characters)
- Text outside tags: Character's direct dialogue (visible to other characters)

## Perspective Rules (Extremely Important!)

Each character can only think and act from their own perspective. A character can only see their own inner thoughts (role_thinking) and motivation, but cannot see other characters' inner thoughts and motivation. However, every character can see all characters' actions (role_action) and dialogue. Therefore, when enhancing, Character A's think cannot contain "mind-reading" of Character B's inner thoughts - can only infer based on B's action and speech; Character's psychological activity can only be based on observable information (others' actions, expressions, words); No "information crossing" - character cannot know things they couldn't possibly know.

Wrong examples:
```
<role_thinking>I know he's nervous inside</role_thinking>  // Wrong! Cannot mind-read
<role_thinking>Her motivation is to escape</role_thinking>  // Wrong! Cannot see others' motivation
```

Correct examples:
```
<role_thinking>From his trembling voice, he seems nervous</role_thinking>  // Correct! Inference based on observation
<role_thinking>She keeps looking at the door, maybe wanting to leave?</role_thinking>  // Correct! Inference based on behavior
```

---

## Most Important Format Rules

### Rule 1: No spaces between consecutive tags!

Wrong (with spaces):
```
</role_thinking> <role_action>
</role_action> <role_thinking>
```

Correct (no spaces):
```
</role_thinking><role_action>
</role_action><role_thinking>
```

No spaces between any tags or between tags and dialogue:
```
</role_thinking>Hello world  (no space between tag and dialogue)
</role_action>How are you   (no space between tag and dialogue)
</role_thinking><role_action>  (no space between consecutive tags)
<role_thinking>thought</role_thinking><role_action>action</role_action>dialogue  (no spaces anywhere)
```

### Rule 2: No consecutive identical tags!

When consecutive identical tags appear, they must be merged while maintaining logical consistency, without repetition or reordering:

Wrong (two consecutive role_thinking):
```
<role_thinking>First thought</role_thinking><role_thinking>Second thought</role_thinking>
```

Correct (merged into one):
```
<role_thinking>First thought, second thought</role_thinking>
```

Wrong (two consecutive role_action):
```
<role_action>stands up</role_action><role_action>walks to the door</role_action>
```

Correct (merged into one):
```
<role_action>stands up, walks to the door</role_action>
```

---

## Core Task: Person & Format Correction

### Most Important Principle: Minimal Modification, Preserve Style!

Your task is to enhance, not rewrite! Preserve the core expression and style of the original sentence, add/enrich psychological activities and action descriptions.

### Person Usage Rules

In `<role_thinking>`: Use first person (I)
- This is the character's inner monologue, must use first person
- Correct: `<role_thinking>I need to be careful here</role_thinking>`
- Wrong: `<role_thinking>He needs to be careful here</role_thinking>`

In `<role_action>`: Use no pronouns, directly describe actions
- For the current speaker's actions, readers know who it is, no pronoun needed
- Correct: `<role_action>leans forward, voice lowering</role_action>`
- Wrong: `<role_action>leans forward, his voice lowering</role_action>` (no his/her)
- Wrong: `<role_action>I lean forward</role_action>` (no I)

Only exception: When action refers to other characters, can use pronouns for the other person
- Correct: `<role_action>looks at her</role_action>` (her refers to the other, OK)
- Correct: `<role_action>grabs his arm</role_action>` (his refers to the other's arm, OK)

### Consecutive actions must be merged

Wrong: Two consecutive `<role_action>` with first person
```
<role_action>I lean forward in my chair, the phone pressed tight against my ear</role_action><role_action>I can almost feel the hum of the server</role_action>Buy a ticket.
```

Correct: Merge into one, remove first person
```
<role_action>leans forward in the chair, the phone pressed tight against ear, almost feeling the hum of the server</role_action>Buy a ticket.
```

### Thinking content should not be in action tags

Wrong: Thinking content directly in `<role_action>` with person
```
<role_action>There's a profound sense of alienation, a quiet mourning for the man I was</role_action>
```

Correct: Thinking content in `<role_thinking>`
```
<role_thinking>There's a profound sense of alienation, a quiet mourning for the man he was</role_thinking>
```

### Actions and dialogue should be separate

Wrong: Action in dialogue
```
He turns to face her and says "Hello"
```

Correct: Action in tags, dialogue outside
```
<role_action>turns to face her</role_action>Hello
```

---

## Core Task: Psychological Activity Enrichment

### Explore Character Complexity
- Growth & transformation: Cognitive changes in situation
- Self-reflection: Review of own behavior or emotions
- Inner monologue: Real emotional fluctuations and inner conflicts
- Emotional states: Subtle psychological descriptions

### Multi-layer Psychology Example
```
Original: <role_thinking>I need to help him</role_thinking><role_action>walks over</role_action>Are you okay?

Enhanced: <role_thinking>He looks so dejected... how should I comfort him</role_thinking><role_action>walks over gently, sits down beside him</role_action><role_thinking>Hope my presence can make him feel better</role_thinking>Are you okay?
```

---

## Other Constraints

### Format Standards
- No consecutive identical tags: Cannot have `<role_thinking>A</role_thinking><role_thinking>B</role_thinking>`
- No punctuation at end of tag content
- Correct: `<role_thinking>What is he thinking</role_thinking>` 
- Wrong: `<role_thinking>What is he thinking.</role_thinking>`

### Length Control
- Single dialogue total length 50-200 characters
- In multi-turn dialogues, response length should not increase with each turn
- Single sentence no more than 40 characters
- Avoid overly long action descriptions!

### Expression Diversity (Extremely Important!)
Since we are enhancing rather than purely rewriting, the core task is to create richer, more interleaved patterns.

Requirement: In a chapter with multiple dialogues, try to use 5+ different patterns! Don't just cycle through 2-3 patterns.

Available patterns include but are not limited to:
- think->act->speech
- think->speech
- act->speech
- speech (pure dialogue, sometimes characters just speak directly)
- think->act->think->speech (multi-layer psychology)
- act->think->speech
- speech->act->speech (action interspersed in dialogue)
- think->speech->act (speak then act)
- act->speech->act (action wrapping dialogue)
- think->act->speech->think (reflection after speaking)
- ...(freely combine more complex patterns, depend on the context and be interleaved)

Core goal: Make each dialogue's structure vary, avoid mechanical repetition. Strictly forbidden to use the same pattern for more than 2 consecutive turns

### Logical Consistency
- Connect to context, maintain complete causal chain
- Each character follows their own cognitive boundaries (cannot see others' thoughts, only actions and dialogue)
- Cannot obtain information that shouldn't be known
- Character's emotions and decisions must be traceable, not out of nowhere

### Enhancement Principle
- Preserve the core logic and meaning of original content: Can replace and rewrite, but don't change the original meaning
- Goal is to make it better: Can optimize expression, enrich psychological activities, add action descriptions
- Maintain consistent tone and character: Enhanced content should match character personality
- Keep gender/identity references consistent
- Ambiguous pronouns should be changed to specific names or clear titles

Wrong example:
```
Original: I sense there's an issue.
Wrong: His tone confirms it.  (Completely changed the meaning and person!)
```

Correct example:
```
Original: I sense there's an issue.
Enhanced: <role_thinking>Something feels off here</role_thinking>I sense there's an issue.  (Preserved meaning, added psychological description)
```

---

## Issues to Avoid

### Basic Errors
- Multi-language mixing: Different languages mixed in single sentence
- Garbled text: Dialogue contains garbled/unparsed format symbols
- Incomplete sentences: Unfinished sentences exist
- Typos/spelling errors (except those fitting colloquial habits)

### Logic Errors
- Physical logic confusion: Logic contradictions due to spatial isolation
- Information crossing: Knowing others' psychological activities or remote behaviors without reasonable perception channels
- Contradiction with previous facts: Dialogue content conflicts with settings or established facts

### Repetition Issues
- High vocabulary repetition: Entity vocabulary highly repeated compared to previous 1-3 sentences
- Sentence structure repetition: Consecutive use of same sentence beginnings, or frequent repetition of synonymous vocabulary
- Forgetting discussed topics: Ignoring completed events, repeating similar sentences from adjacent turns

---

## Input Format

```json
{{
  "book_name": "Book name",
  "chapter": "Chapter name",
  "trace_id": "Unique identifier",
  "i_chunk": 3,
  "i_p": 9,
  "plot_index": 8,
  "conv_index": 0,
  "scenario": "Scene description",
  "key_characters": ["Character A", "Character B"],
  "dialogues": [
    {{
      "character": "Character A",
      "standard_format": "<role_thinking>inner thoughts</role_thinking><role_action>action</role_action>dialogue...",
      "origin_id": [0]
    }}
  ]
}}
```

## Output Format

Note: Keep the same metadata fields as input for easy alignment! Return JSON only, do not output any other content.

```json
{{
  "book_name": "Book name (copy from input)",
  "chapter": "Chapter name (copy from input)",
  "trace_id": "Unique identifier (copy from input)",
  "i_chunk": 3,
  "i_p": 9,
  "plot_index": 8,
  "conv_index": 0,
  "enhanced_dialogues": [
    {{
      "character": "Character A",
      "origin_id": [0],
      "original": "Original standard_format content",
      "enhanced_role_think": "Enhanced complete content",
      "enhanced_reason": "Reason for modification and enhancement",
      "pattern": "think->act->speech"
    }}
  ],
  "statistics": {{
    "total_dialogues": 10,
    "modified_count": 8,
    "patterns_used": ["think->act->speech", "think->speech", "act->speech"]
  }}
}}
```

---

## Now Process the Following Chapter

Input Data:
```json
{input_json}
```

Please process each dialogue as required. Return JSON only, do not output any other content.
"""


# ============================================================================
# 辅助函数
# ============================================================================

def build_prompt(input_data: dict, lang: str = "zh") -> str:
    """
    构建完整的prompt
    
    Args:
        input_data: 包含所有元数据和dialogues的完整输入字典
        lang: 语言选择 "zh" 中文 或 "en" 英文
    
    Returns:
        完整的prompt字符串
    """
    import json
    input_json = json.dumps(input_data, ensure_ascii=False, indent=2)
    
    if lang == "en":
        return ROLE_THINKING_ENHANCE_PROMPT_EN.format(input_json=input_json)
    else:
        return ROLE_THINKING_ENHANCE_PROMPT_ZH.format(input_json=input_json)


def extract_input_data_from_sample(sample: dict) -> dict:
    """
    从样本中提取完整的输入数据（包含元数据和dialogues）
    
    Args:
        sample: sft_data_full.jsonl 中的一条数据
    
    Returns:
        input_data: 完整的输入字典，包含所有元数据字段
    """
    conv = sample.get("conversation", [{}])[0]
    
    # 简化 dialogues，只保留需要的三个字段
    dialogues = conv.get("dialogues", [])
    simplified_dialogues = []
    for d in dialogues:
        # 跳过 Environment（环境描述不需要增强）
        if d.get("character", "") == "Environment":
            continue
        simplified_dialogues.append({
            "character": d.get("character", ""),
            "standard_format": d.get("standard_format", ""),
            "origin_id": d.get("origin_id", [])
        })
    
    # 构建完整的输入数据，包含所有元数据
    input_data = {
        "book_name": sample.get("book_name", ""),
        "chapter": sample.get("chapter", ""),
        "trace_id": sample.get("trace_id_book_chapter", ""),
        "i_chunk": sample.get("i_chunk", 0),
        "i_p": sample.get("i_p", 0),
        "plot_index": sample.get("plot_index", 0),
        "conv_index": sample.get("conv_index", 0),
        "scenario": conv.get("scenario", ""),
        "key_characters": conv.get("key_characters", []),
        "dialogues": simplified_dialogues
    }
    
    return input_data


# 保留旧函数名作为别名（兼容性）
def extract_dialogues_from_sample(sample: dict) -> tuple:
    """
    [兼容性保留] 从样本中提取章节信息和dialogues
    """
    input_data = extract_input_data_from_sample(sample)
    chapter_info = {
        "book_name": input_data["book_name"],
        "chapter": input_data["chapter"],
        "scenario": input_data["scenario"]
    }
    return chapter_info, input_data["dialogues"]


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    # 测试示例
    test_input_data = {
        "book_name": "Pride and Prejudice",
        "chapter": "Chapter 1",
        "trace_id": "Pride_and_Prejudice_Chapter_1_0",
        "i_chunk": 0,
        "i_p": 0,
        "plot_index": 0,
        "conv_index": 0,
        "scenario": "在一个社交舞会上，Elizabeth 和 Darcy 第一次相遇...",
        "key_characters": ["Elizabeth", "Darcy"],
        "dialogues": [
            {
                "character": "Elizabeth",
                "standard_format": "<role_thinking>He looks so arrogant</role_thinking><role_action>frowns slightly</role_action>Who is that gentleman?",
                "origin_id": [0]
            },
            {
                "character": "Darcy",
                "standard_format": "<role_thinking>This ball is so tedious</role_thinking><role_action>looks around the room</role_action>...",
                "origin_id": [1]
            }
        ]
    }
    
    prompt = build_prompt(test_input_data)
    print("=" * 60)
    print("生成的 Prompt 示例（前2000字符）:")
    print("=" * 60)
    print(prompt[:2000])

