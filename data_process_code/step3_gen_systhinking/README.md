# Step 3: System Thinking 生成与改写

**状态**: ✅ 代码可用

---

## 🎯 目标

为每个对话轮次生成 **System Thinking** (第三人称视角的角色扮演规划)，并进行逻辑一致性改写，使其与增强后的 `enhanced_speech` 对齐。

---

## 📊 数据流

```
sft_data_enhanced.jsonl (Step 2 输出，含 enhanced_speech)
        ↓
[Step 3.1-3.3] 生成原始 sys_thinking
        ↓
all_success_final.jsonl (turn 级别结果)
        ↓
[Step 3.6] 合并到 SFT 数据
        ↓
[Step 3.7] 构建改写数据 → 模型推理
        ↓
[Step 3.8/3.10] 合并改写结果
        ↓
[Step 3.11] 合并到 dialogues
        ↓
sft_data_final.jsonl (最终输出)
```

---

## 📁 目录结构

```
step3_gen_systhinking/
│
├── README.md                                 # [本文档]
│
├── step3_1_extract_sys_thinking_samples.py   # 3.1 提取推理样本
├── step3_2_construct_vulcan_data.py          # 3.2 构建推理数据
├── step3_3_extract_model_think.py            # 3.3 提取模型思考
├── step3_6_merge_to_sft.py                   # 3.6 合并到 SFT
├── step3_7_construct_rewrite_data.py         # 3.7 构建改写数据
├── step3_8_merge_rewrite_results_parallel.py # 3.8 并行合并改写结果
├── step3_10_fix_and_merge.py                 # 3.10 修复 JSON + 合并
└── step3_11_merge_to_dialogues.py            # 3.11 合并到 dialogues
```

> **注意**: 运行前请修改脚本中的路径配置（标记为 `/path/to/data/example` 的地方）

---

## 🔧 执行步骤

### Phase 1: 生成原始 System Thinking

```bash
# 3.1 提取样本
python step3_1_extract_sys_thinking_samples.py

# 3.2 构建推理数据
python step3_2_construct_vulcan_data.py

# 3.3 模型推理 (使用你的推理平台)
# 输入: step3_2 的输出
# 输出: 模型生成的 sys_thinking

# 3.3 提取模型思考结果
python step3_3_extract_model_think.py

# 3.6 合并到 SFT
python step3_6_merge_to_sft.py
```

### Phase 2: 改写 System Thinking (对齐 enhanced_speech)

```bash
# 3.7 构建改写数据
python step3_7_construct_rewrite_data.py

# 模型推理 (改写)
# 输入: step3_7 的输出
# 输出: 改写后的 sys_thinking

# 3.8/3.10 合并结果
python step3_8_merge_rewrite_results_parallel.py  # 并行合并
# 或
python step3_10_fix_and_merge.py                  # 修复 JSON + 合并

# 3.11 合并到 dialogues
python step3_11_merge_to_dialogues.py
```

---

## 📝 数据格式

### 输出格式

```json
{
  "conversation": [{
    "scenario": "...",
    "dialogues": [
      {
        "character": "Heywood Floyd",
        "enhanced_speech": "<role_action>...</role_action>...",
        "sys_thinking": "I need to portray Heywood Floyd as..."
      }
    ]
  }],
  "training_samples": {
    "Heywood Floyd": [
      {"role": "system", "content": "..."},
      {
        "role": "assistant",
        "content": "...",
        "sys_thinking_revised": "...",
        "sys_thinking_original": "..."
      }
    ]
  }
}
```

### System Thinking 格式要求

- **第三人称视角**: "I need to portray {character} as..."
- **不使用 "user"**: 用角色名代替
- **长度保持**: 输出长度与原始输入一致
- **格式保持**: 保留原始结构 (如 Context:, Goal:, Plan: 等)
- **逻辑对齐**: 与 `enhanced_speech` 的 `<role_thinking>`, `<role_action>`, 台词一致

---

## 🔗 依赖文件

| 输入 | 来源 |
|------|------|
| `sft_data_enhanced.jsonl` | Step 2 输出 |
| sys_thinking 推理结果 | Step 3.3 生成 |

---

## 📚 Prompt 设计要点

主要原则:
1. 第三人称视角强制 ("I need to portray X as...")
2. 禁止使用 "user"，用角色名代替
3. 长度保持 (target_chars)
4. 格式保持 (PRESERVE FORMAT)
5. system_info 嵌入 JSON
6. 首轮场景分析
