# Step 2: Role Thinking 增强

## 📁 目录结构

```
step2_gen_rolethinking/
│
├── sft_data_enhanced.jsonl             # 🎯 最终输出 (2.4GB)
│
├── construct_vulcan_data.py            # Step 2.1: 构建推理数据
├── merge_extract_results.py            # Step 2.3: 合并推理结果
├── merge_enhanced_to_sft.py            # Step 2.4: 合并回原数据
├── role_thinking_enhance_prompt.py     # Prompt 定义 (中英双语)
├── analyze_pattern_diversity.py        # 多样性分析脚本
│
```

## 🔄 复现步骤

### 前置条件

- Step 1 输出: `/path/to/project/data_process/step1_data_process/sft_data_full.jsonl`

### Step 2.1: 构建推理数据

```bash
python construct_vulcan_data.py --language en
```

输出: `inference_data/en/role_thinking_enhance_en_full.jsonl`

### Step 2.2: 模型推理

提交到推理平台进行 model 推理

输出: `inference_data/en/role_thinking_enhance_en_full/*.jsonl`

### Step 2.3: 合并推理结果

```bash
python merge_extract_results.py --lang en
```

输出: `enhanced_output/en/enhanced_dialogues_en.jsonl`

### Step 2.4: 合并回原数据

```bash
python merge_enhanced_to_sft.py
```

输出: `sft_data_enhanced.jsonl`

### 分析多样性

```bash
python analyze_pattern_diversity.py
```

输出: `enhanced_output/en/pattern_diversity_report.md`

## 📝 输出格式

每条 dialogue 新增字段:

```json
{
  "character": "角色",
  "origin_id": [0],
  "standard_format": "原始格式",
  "without_think": "无思考版本",
  "enhanced_standard_format": "<role_action>动作</role_action><role_thinking>深层思考</role_thinking>对话...",
  "enhanced_reason": "修改原因说明",
  "enhanced_pattern": "act->think->speech"
}
```
