# Step 4: Setting Completion

根据论文 **Stage 3: Integration and Setting Completion**:

> The original character setting $\mathcal{S}$ may lack sufficient detail to support the newly enriched psychological depth and reasoning traces. Therefore, we perform a **Setting Completion** step.

使用**原文 text** 和**生成的对话** (sys_thinking + enhanced_speech) 来丰富角色设定，确保：
1. **基于原文**: 所有细节都来自源文本
2. **解释行为**: 设定能解释对话中角色的行为
3. **防止幻觉**: 补充缺失信息，避免模型编造

---

## 📊 数据流

```
sft_data_final_v4.jsonl (Step 3 输出)
├── text (原文)
├── dialogues (含 sys_thinking + enhanced_speech)
└── character_datasets (当前设定)
        ↓
[Step 4.1] 构建推理数据
        ↓
setting_completion_data.jsonl → Vulcan 推理
        ↓
[Step 4.2] 合并结果
        ↓
sft_data_final_enriched.jsonl (丰富的设定)
```

---

## 📁 目录结构

```
step4_setting_completion/
│
├── README.md                              # [本文档]
├── sft_data_final_enriched.jsonl          # [输出] 丰富设定后的数据
│
├── step4_1_construct_setting_data.py      # 4.1 构建推理数据
├── step4_2_merge_setting_results.py       # 4.2 合并结果
│
├── step4_0_fix_system_prompt.py           # 辅助: 修复 system prompt
├── step4_3_rebuild_system_prompt.py       # 辅助: 重建 system prompt
├── step4_4_add_prompt_config.py           # 辅助: 添加 prompt config
├── step4_5_merge_patches.py               # 辅助: 合并补丁
├── generate_training_samples.py           # 辅助: 生成训练样本
│
└── main/
    ├── setting_completion_data.jsonl      # Vulcan 输入
    └── output/                            # Vulcan 输出
```

---

## 🔧 执行步骤

```bash
cd /path/to/project/data_process/step4_setting_completion

# 4.1 构建推理数据
python step4_1_construct_setting_data.py

# 4.2 Vulcan 推理
mmctl vulcan tide-job create \
    --name setting_completion \
    --queue qu-xxxxx \
    --priority 1000 \
    --rule-version r-xxxxx@v6 \
    --task-max-retry 100 \
    --input-jfs-path users/username/project/data_process/step4_setting_completion/main/setting_completion_data.jsonl \
    --input-jfs-file-sys-name jfs-xxxxx \
    --input-cluster-name xxxxx \
    --output-jfs-path users/username/project/data_process/step4_setting_completion/main/output/ \
    --output-jfs-file-sys-name jfs-xxxxx \
    --output-cluster-name xxxxx

# 4.3 合并结果
python step4_2_merge_setting_results.py
```

---

## 🎯 核心原则：需求驱动增强

关键设计是**需求驱动的增强**：

1. **分析对话** → 发现角色展现的行为/特质/情感
2. **检查设定** → 该行为在原设定中是否有解释
3. **搜索原文** → 如果缺失，从原文中找相关描述
4. **补充设定** → 添加能解释对话行为的设定

### ✅ 正确示例

```
对话显示: 角色表现出焦虑，动作迟缓
原设定缺失: 没有解释焦虑的原因
原文支持: "neither had slept the night before"
增强结果: 添加 "一夜未眠" 来解释对话中的疲惫表现
```

### ❌ 错误示例（需避免）

```
对话显示: 无相关内容
原设定缺失: 睡眠状态
原文支持: "neither had slept"
增强结果: ❌ 不应添加（对话中没有需要解释的行为）
```

---

## 📝 Setting 字段说明

### 输入 (原始设定)

| 字段 | 说明 | 来源 |
|------|------|------|
| `character_profile` | 角色描述 | CoSER 原始 |
| `background` | 背景/剧情摘要 | CoSER 原始 |
| `scenario` | 当前场景 | CoSER 原始 |
| `motivation` | 角色动机 | CoSER 原始 |
| `description` | 简短描述 | CoSER 原始 |
| `experience` | 角色经历 | CoSER 原始 |

### 输出 (增强设定)

| 字段 | 说明 |
|------|------|
| `character_profile_enriched` | 增强后的角色描述（基于原文和对话需求） |
| `background_enriched` | 增强后的背景 |
| `motivation_enriched` | 增强后的动机 |
| `description_enriched` | 增强后的描述 |
| `experience_enriched` | 增强后的经历 |
| `setting_enrichment_reasoning` | 增强推理过程（说明为什么添加这些内容） |

---

## 💻 使用示例

### 使用增强字段进行训练

```python
import json

with open('sft_data_final_enriched.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)

        for char_name, char_data in sample['character_datasets'].items():
            # 使用增强后的字段（如果存在，否则 fallback 到原始字段）
            enriched_profile = char_data.get('character_profile_enriched',
                                            char_data.get('character_profile', ''))
            enriched_background = char_data.get('background_enriched',
                                               char_data.get('background', ''))
            # ... 其他字段同理
```

### 对比实验：原始 vs 增强

```python
# 实验A: 使用原始字段
profile_A = char_data['character_profile']
background_A = char_data['background']

# 实验B: 使用增强字段
profile_B = char_data['character_profile_enriched']
background_B = char_data['background_enriched']

# 对比训练效果
```

### 查看增强推理

```python
# 了解为什么添加了某些信息
reasoning = char_data.get('setting_enrichment_reasoning', '')
print(reasoning)
```

---

## ⚠️ 注意事项

1. **原文限制**: 原文 `text` 限制为 8000 字符（避免超长）
2. **对话摘要**: 只取前 5 轮（代表性）
3. **跳过 Environment**: Environment 角色无需丰富
4. **原始字段保留**: 所有原始字段完全保留，增强字段使用 `_enriched` 后缀
5. **兼容性**: 如果没有 enriched 字段，fallback 到原始字段，向后兼容

---

## 🔗 依赖文件

| 文件 | 说明 |
|------|------|
| `step3_gen_systhinking/sft_data_final_v4.jsonl` | Step 3 输出 |
