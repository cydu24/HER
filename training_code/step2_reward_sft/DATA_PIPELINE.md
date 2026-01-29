# 完整数据处理流程

## 📊 数据流程表格

| 步骤 | 脚本 | 输入文件 | 输出文件 | 说明 |
|------|------|----------|----------|------|
| 1. 合并推理结果 | `merge_datasets_for_rm.py` | 多个推理结果文件 | `merged_for_rm_final.jsonl` | 合并原始模型输出（8个候选） |
| 2. 构造评估任务 | `process_inference_results.py` | `merged_for_rm_final.jsonl` | `inference_data_choices_0_1.jsonl` | 构造评估任务（包含 principles） |
| 3. 模型推理评估 | `rm.py` | `inference_data_choices_0_1.jsonl` | `model_processed_enhanced_result.jsonl` | 生成评估结果 |
| 4. 筛选和分配数据 | `filter_high_quality_sft.py` | `model_processed_enhanced_result.jsonl` | `sft_final.jsonl`, `rl_final.jsonl`, `test_final.jsonl` | 按质量分配到SFT/RL/Test |
| 5. 构造RM训练数据 | `construct_rm_training_data.py` | `sft_final.jsonl`, `rl_final.jsonl` | `rm_training_sft.jsonl`, `rm_training_rl.jsonl` | 转换为RM训练格式（不包含给定principles） |

---

## 🔄 详细流程说明

### 步骤1: 合并推理结果
**目的**: 将多个批次的推理结果合并成一个文件
- 输入: 分批次的推理结果（包含8个候选回复）
- 输出: 合并后的完整文件
- 数据格式: 

```json
{
  "trace_id": "xxx",
  "messages": [...],
  "model_response": {
    "choices": [
      {"message": {"content": "候选1"}},
      {"message": {"content": "候选2"}},
      ...
      {"message": {"content": "候选8"}}
    ]
  }
}
```

### 步骤2: 构造评估任务
**目的**: 从8个候选中选择2个，构造评估任务
- **关键特点**: 包含精心设计的 principles 作为评估标准
- 输入: 合并后的推理结果
- 输出: 评估任务（选择候选0和1进行对比）
- Prompt特点: 
  - ✅ 包含给定的 evaluation principles
  - ✅ 要求从这些 principles 中选择相关的
  - ✅ 用于获得高质量的评估标注

### 步骤3: 模型推理评估
**目的**: 让模型按照给定的 principles 评估两个候选
- 输入: 评估任务
- 输出: 评估结果
- 输出包含:
  - 选择的 principles（从给定的 principles 中）
  - 详细的分析
  - winner 判断

### 步骤4: 筛选和分配数据
**目的**: 按照质量和 winner 分布分配数据
- 筛选条件: principle >= 3
- 分配策略:
  - **SFT数据**: both_sides + cand_1_only + cand_2_only + tie_only + no_winner
  - **RL数据**: 剩余的 both_sides + cand_1 + cand_2 + 部分 tie
  - **Test数据**: 从 RL 和 tie 中抽取

### 步骤5: 构造RM训练数据 ⭐ **关键步骤**
**目的**: 将评估结果转换为RM训练数据
- **关键改变**: 
  - ❌ Input Prompt: **不包含给定的 principles**
  - ✅ Input Prompt: 要求模型**自己生成 principles**
  - ✅ Output Label: 使用完整评估结果
  
- 训练目标:
  1. 学会根据对话上下文生成合适的 evaluation principles
  2. 学会应用这些 principles 分析回复
  3. 学会判断哪个回复更好

---

## 📝 运行命令

```bash
cd step2_reward_sft

# 合并推理结果
python merge_datasets_for_rm.py

# 构造评估任务
python process_inference_results.py

# 运行模型评估
python rm.py

# 筛选和分配数据
python filter_high_quality_sft.py

# 构造RM训练数据
python construct_rm_training_data.py
```

---

## 📁 最终输出文件

```
final_datasets/
├── rm_training_sft.jsonl   (SFT 数据)
├── rm_training_rl.jsonl    (RL 数据)
└── rm_training_test.jsonl  (测试数据)
```

---

## 🎯 关键设计理念

### 为什么分两阶段？

1. **第一阶段（模型评估）**: 
   - 使用精心设计的 principles
   - 获得高质量的评估标注
   - 确保评估的准确性和一致性

2. **第二阶段（RM训练）**:
   - 不给定 principles
   - 让RM学会自己生成
   - 提高泛化能力和灵活性

### RM能学到什么？

通过学习评估结果，RM将学会:
- ✅ 在不同对话场景下应该关注哪些评估维度
- ✅ 如何定义和描述这些评估原则
- ✅ 如何应用原则来判断回复质量
- ✅ 什么样的差异值得用什么样的原则来评估
