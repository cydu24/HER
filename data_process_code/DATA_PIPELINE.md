# HER Dataset 数据合成流程文档

**目标**: 将 CoSER 原始数据转换为包含 System Thinking + Role Thinking 的 HER Dataset

---

## 🚀 当前进度

| 阶段 | 目的 | 状态 | 输出 |
|------|------|------|------|
| **Step 1** | 数据预处理 | ✅ 已完成 | `sft_data_full.jsonl` |
| **Step 2** | Role Thinking 增强 | ✅ 已完成 | `sft_data_enhanced.jsonl` |
| **Step 3** | System Thinking 生成+改写 | ✅ 已完成 | `sft_data_final.jsonl` |
| **Step 4** | Setting Completion | ✅ 已完成 | `sft_data_final_enriched.jsonl` |
| **Step 5** | SFT 数据构建 (消融实验) | ✅ 已完成 | `ablation_sft/` |

---

## 📊 数据流总览（主线）

```
CoSER 原始数据 (760 本书)
        ↓
[Step 1] 数据清洗 + 格式转换
        ↓
sft_data_full.jsonl (29,081 对话, 383,654 轮)
        ↓
[Step 2] Role Thinking 增强 (model 推理)
        ↓
sft_data_enhanced.jsonl (+ enhanced_standard_format)
        ↓
[Step 3] System Thinking 生成 + 改写
        ↓
sft_data_final.jsonl (+ sys_thinking)
        ↓
[Step 4] Setting Completion (丰富角色设定)
        ↓
sft_data_final_enriched.jsonl (+ *_enriched 字段)
        ↓
[Step 5] SFT 数据构建 (消融实验)
        ↓
ablation_sft/ (342,493 样本 × 2 版本)
        ↓
✅ 训练任务 (with/without system_thinking)
```

---

## 📊 Step 1: 数据预处理

### 处理步骤

| 步骤 | 处理代码 | 输出 | 目的 |
|------|----------|------|------|
| 1.1 | `clean_empty_data.py` | 760个JSON | 清洗空数据 |
| 1.2 | `normalize_filenames.py` | 文件名规范化 | 规范化文件名 |
| 1.3 | `convert_to_sft_format.py` | `sft_data_full.jsonl` | 转换为 SFT 格式 |

### 数据统计

| 指标 | 数量 |
|------|------|
| 书籍数 | 760 |
| 对话样本 | 29,081 |
| 对话轮次 | 383,654 |
| 训练样本 (每角色) | 76,883 |

---

## 📊 Step 2: Role Thinking 增强

### 处理步骤

| 步骤 | 处理代码 | 输出 | 目的 |
|------|----------|------|------|
| 2.1 | `construct_vulcan_data.py` | 推理输入 | 构建推理数据 |
| 2.2 | 模型推理 (model) | 推理结果 | LLM 增强心理活动 |
| 2.3 | `merge_extract_results.py` | 增强对话 | 合并推理结果 |
| 2.4 | `merge_enhanced_to_sft.py` | `sft_data_enhanced.jsonl` | 合并回原数据 |


## 📊 Step 3: System Thinking 生成与改写 ✅

### 概述

基于论文 **Stage 2: System Thinking Construction**:
1. **Phase 1**: 让推理模型为每个对话轮次生成系统级思考过程
2. **Phase 2**: 改写 sys_thinking 使其与 enhanced_speech 逻辑对齐

### 处理步骤

| 步骤 | 处理代码 | 输出 | 目的 |
|------|----------|------|------|
| 3.1 | `step3_1_extract_sys_thinking_samples.py` | 342,493 条 turn | 提取推理样本 |
| 3.2 | `step3_2_construct_vulcan_data.py` | 推理输入 | 构建推理数据 |
| 3.3 | 模型推理 + `step3_3_extract_model_think.py` | `all_success_final.jsonl` | 生成原始 thinking |
| 3.4 | 聚合脚本 | `aggregated_by_training_sample.jsonl` | 按角色聚合 |
| 3.5 | `step3_7_construct_rewrite_data.py` | 改写数据 | 构建改写 prompt |
| 3.6 | 模型推理 (改写) | 改写结果 | LLM 改写对齐 |
| 3.7 | `step3_10_fix_and_merge.py` | 合并到 training_samples | 修复 JSON + 合并 |
| 3.8 | `step3_11_merge_to_dialogues.py` | `sft_data_final.jsonl` | 合并到 dialogues |


---

## 📊 Step 4: Setting Completion ✅

### 概述

根据论文 **Stage 3: Integration and Setting Completion**:

> The original character setting may lack sufficient detail to support the newly enriched psychological depth and reasoning traces.

使用**原文 text** 和**生成的对话** (sys_thinking + enhanced_speech) 来丰富角色设定。

### 处理步骤

| 步骤 | 处理代码 | 输出 | 目的 |
|------|----------|------|------|
| 4.1 | `step4_1_construct_setting_data.py` | `setting_completion_data.jsonl` | 构建推理输入 |
| 4.2 | 模型推理 (model) | `output/` | LLM 分析原文 |
| 4.3 | `step4_2_merge_setting_results.py` | `sft_data_final_enriched.jsonl` | 合并结果 |


### 新增字段

| 字段 | 说明 |
|------|------|
| `character_profile_enriched` | 丰富的角色描述（基于原文） |
| `background_enriched` | 丰富的背景（解释行为） |
| `motivation_enriched` | 深入的动机分析 |
| `description_enriched` | 丰富的描述 |
| `experience_enriched` | 丰富的经历 |
| `setting_enrichment_reasoning` | 增强推理过程 |

### 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `sft_data_final_enriched.jsonl` | **3.92GB** | **✅ Step 4 最终输出** |

---

## 📁 目录结构

```
/path/to/project/data_process/
│
├── DATA_PIPELINE.md                  # [本文档]
│
├── step1_data_process/               # Step 1: 数据预处理
│   ├── full_final_cleaned/           # 清洗后的数据 (760个JSON)
│   └── sft_data_full.jsonl           # [输出] 2.3GB
│
├── step2_gen_rolethinking/           # Step 2: Role Thinking 增强
│   ├── sft_data_enhanced.jsonl       # [输出] 2.5GB
│   │
│   ├── construct_vulcan_data.py      # 构建推理数据
│   ├── merge_extract_results.py      # 合并推理结果
│   ├── merge_enhanced_to_sft.py      # 合并回原数据
│   │
│
├── step3_gen_systhinking/            # Step 3: System Thinking 生成+改写
│   ├── README.md                     # Step 3 详细说明
│   ├── sft_data_final.jsonl          # [输出] 3.7GB
│   │
│   ├── step3_1_extract_sys_thinking_samples.py
│   ├── step3_2_construct_vulcan_data.py
│   ├── step3_3_extract_model_think.py
│   ├── step3_7_construct_rewrite_data.py
│   ├── step3_10_fix_and_merge.py
│   ├── step3_11_merge_to_dialogues.py
│   │
│   └── main/
│       ├── final_data/
│       │   ├── aggregated_by_training_sample.jsonl
│       │   └── all_success_final.jsonl
│       └── rewrite_data/
│           └── output_full/          # 改写推理输出
│
└── step4_setting_completion/         # Step 4: Setting Completion
    ├── README.md
    │
    ├── step4_1_construct_setting_data.py    # 构建推理数据
    ├── step4_2_merge_setting_results.py     # 合并脚本
    │
    ├── sft_data_final_enriched.jsonl        # [✅ 最终输出] 3.92GB
    │
    └── main/
        ├── setting_completion_data.jsonl    # 推理输入数据
        └── output/                          # 推理输出
```

---

## 📚 关键文件清单（主线）

| 文件 | 大小 | 说明 | 位置 |
|------|------|------|------|
| `sft_data_full.jsonl` | 2.3GB | Step 1 输出 | step1_data_process/ |
| `sft_data_enhanced.jsonl` | 1.9GB | Step 2 输出 | step2_gen_rolethinking/ |
| `sft_data_final.jsonl` | 3.7GB | Step 3 输出 | step3_gen_systhinking/ |
| `sft_data_final_enriched.jsonl` | 4.0GB | Step 4 输出 | step4_setting_completion/ |
| **`sft_data_final_full_prompt.jsonl`** | **4.6GB** | **✅ 最终输出（完整prompt）** | step4_setting_completion/ |

---

## 📈 整体统计

| 指标 | 数量 |
|------|------|
| 书籍数 | 760 |
| 对话样本 | 29,081 |
| 对话轮次 | 371,775 |
| Assistant turns | 342,493 |
| 有 sys_thinking | 342,489 (100%) |
| 有 enhanced_speech | 327,744 (88.2%) |
| Setting 增强覆盖 | 28,592 (98.32%) |

---

## 🔄 数据格式

### 最终输出格式 (`sft_data_final_enriched.jsonl`)

```json
{
  "conversation": [{
    "scenario": "...",
    "dialogues": [
      {
        "character": "Character Name",
        "enhanced_speech": "<role_action>...</role_action><role_thinking>...</role_thinking>...",
        "sys_thinking": "I need to portray Character as..."
      }
    ]
  }],
  "training_samples": {
    "Character Name": [
      {"role": "system", "content": "角色设定..."},
      {
        "role": "assistant",
        "content": "...",
        "sys_thinking_revised": "改写后的 sys_thinking"
      },
      {"role": "user", "content": "..."},
      ...
    ]
  },
  "character_datasets": {
    "Character Name": {
      // 原始字段
      "character_profile": "...",
      "background": "...",
      "motivation": "...",
      // 增强字段
      "character_profile_enriched": "...",
      "background_enriched": "...",
      "motivation_enriched": "...",
      "setting_enrichment_reasoning": "..."
    }
  }
}
```


---

## 📊 Step 5: SFT 数据构建 (消融实验) ✅

### 概述

基于 `sft_data_final_patched.jsonl` 构建消融实验数据，用于验证 `<system_thinking>` 的作用。

**关键设计原则（训推一致）**：
1. 历史轮的 assistant 消息**不包含** `<system_thinking>`
2. 只有**最后一轮（trainable）** 的 assistant 消息包含 `<system_thinking>`
3. 连续的 user 消息需要合并（框架要求）
4. 其他角色的发言需要移除 `<role_thinking>`（对方看不到）

### 处理步骤

| 步骤 | 处理代码 | 输出 | 目的 |
|------|----------|------|------|
| 5.1 | `rebuild_sft_with_systhink.py` | `sft_roleplay_all_systhink.jsonl` | 构建单轮训练样本 |
| 5.2 | `build_ablation.py` | `ablation_sft/` | 生成消融实验数据 |

### 数据格式（训推一致）

```
[system] You are {character} from {book}...
         ==={character}'s Profile===
         {enriched_profile}
         ===Background===
         {enriched_background}
         ...
         ===Requirements===
         {with/without system_thinking 说明}

[user]   ===Conversation Start===
         
         {other_char}: {content without role_thinking}
         ...

[assistant] {char}: {content}  <- 历史轮，无 <system_thinking>

[user]   {other_char}: {content without role_thinking}

[assistant] <system_thinking>...</system_thinking>{char}: {content}  <- 最后轮，有 <system_thinking>
```

### 消融实验数据

| 文件 | 样本数 | 说明 |
|------|--------|------|
| `sft_roleplay_with_systhink.jsonl` | 342,493 | 最后轮有 `<system_thinking>`, prompt 有说明 |
| `sft_roleplay_without_systhink.jsonl` | 342,493 | 最后轮无 `<system_thinking>`, prompt 无说明 |

### 训练配置

| 参数 | 值 |
|------|------|
| global_batch_size | 32 |
| max_epochs | 4 |
| 每 epoch batches | 10,702 |
| **总 batches** | **42,808** |
| loss_scale | last_round (只训练最后一轮) |

### 输出文件位置

```
/path/to/project/
├── data/
│   ├── sft_with_systhink/
│   │   └── sft_roleplay_all_systhink.jsonl    # 中间文件 (342,493)
│   │
│   └── ablation_sft/                           # ✅ 最终消融数据
│       ├── sft_roleplay_with_systhink.jsonl    # 有 system_thinking
│       ├── sft_roleplay_without_systhink.jsonl # 无 system_thinking
│       └── ablation_stats.json                 # 统计信息
│
└── code/step1_roleplay_sft/
    ├── rebuild_sft_with_systhink.py            # Step 5.1
    └── build_ablation.py                       # Step 5.2
```

### 训练任务

| 实验 | 说明 |
|------|------|
| 有 system_thinking | `model-with-systhink` |
| 无 system_thinking | `model-without-systhink` |

---
