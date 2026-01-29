# SFT 数据准备流程

---

## 📋 流程概览

```
原始数据 (sft_data_final_patched.jsonl)
    ↓ sync_dialogues_to_training_samples.py
    ↓ (同步 dialogues 到 training_samples，去掉别人的 role_thinking)
    
sft_data_final_synced.jsonl (同步后)
    ↓ convert_to_sft.py
    ↓ (转换为 SFT 格式)

sft_train_data.jsonl (多轮样本)
    ↓ split_train_test.py
    ↓ (按 test_set_coser.json 分离测试集)
    
split_data/
├── sft_train.jsonl (训练集)
├── sft_val.jsonl (验证集, 5%) 
└── sft_test.jsonl (测试集)
    ↓ split_to_single_turn.py
    ↓ (拆分多轮为单轮，每轮都有 system_thinking)

single_turn_data/
├── sft_train_single.jsonl (单轮训练集, ~4.5x 扩展)
└── sft_val_single.jsonl (单轮验证集)
    ↓ split_by_purpose.py
    ↓ (按用途分流，用完全部数据)

final_split/
├── sft_roleplay.jsonl (~33%)  ← 角色扮演 SFT
├── roleplay_rl.jsonl (~8%)    ← 角色扮演 RL
├── rm_sft.jsonl (~34%)        ← RM SFT 训练
├── rm_rl.jsonl (~25%)         ← RM RL 训练
└── rl_test.jsonl              ← RL 测试集
```

---

## 📊 数据处理步骤

### Step 1: 同步 dialogues 到 training_samples

| 操作 | 说明 |
|------|------|
| 输入 | `sft_data_final_patched.jsonl` |
| 输出 | `sft_data_final_synced.jsonl` |
| 脚本 | `sync_dialogues_to_training_samples.py` |
| 逻辑 | - assistant 消息: 保留完整 `enhanced_standard_format` (含 `<role_thinking>`) |
|      | - user 消息: 去掉 `<role_thinking>` (别人的思考不可见) |

### Step 2: 转换为 SFT 格式

| 操作 | 说明 |
|------|------|
| 输入 | `sft_data_final_synced.jsonl` (Step 1 输出) |
| 输出 | `sft_train_data.jsonl` |
| 脚本 | `convert_to_sft.py` |
| 逻辑 | - 提取每个角色的 training_samples |
|      | - **只有最后一轮 assistant 加 `<system_thinking>`** |

### Step 3: 分离测试集

| 数据集 | 说明 |
|--------|------|
| **sft_train.jsonl** | 训练集 |
| **sft_val.jsonl** | 验证集 (5%) |
| **sft_test.jsonl** | 测试集 (基于 test_set_coser.json) |

脚本: `split_train_test.py`

### Step 4: 拆分为单轮

将多轮对话拆分为单轮样本，每轮都作为独立训练样本。

脚本: `split_to_single_turn.py`

**历史轮数分布**:
- 0 轮历史: ~22%
- 1 轮历史: ~19%
- 2 轮历史: ~17%
- 3 轮历史: ~14%
- 4 轮历史: ~12%
- 5+ 轮历史: ~16%

### Step 5: 按用途分流

| 数据集 | 占比 | 用途 |
|--------|------|------|
| **sft_roleplay.jsonl** | ~33% | 角色扮演 SFT |
| **roleplay_rl.jsonl** | ~8% | 角色扮演 RL |
| **rm_sft.jsonl** | ~34% | RM SFT 训练 |
| **rm_rl.jsonl** | ~25% | RM RL 训练 |
| **rl_test.jsonl** | - | RL 测试集 |

脚本: `split_by_purpose.py`（用完全部单轮数据）

---

## 📁 目录结构

```
step1_roleplay_sft/
├── # 主要脚本
├── sync_dialogues_to_training_samples.py  # Step 1: 同步 dialogues
├── convert_to_sft.py                      # Step 2: 转换 SFT 格式
├── split_train_test.py                    # Step 3: 分离测试集
├── split_to_single_turn.py                # Step 4: 拆分单轮
├── split_by_purpose.py                    # Step 5: 按用途分流
│
├── # 中间数据
├── sft_data_final_synced.jsonl            # Step 1 输出 (同步后)
├── sft_train_data.jsonl                   # Step 2 输出
│
├── split_data/                            # Step 3 输出
│   ├── sft_train.jsonl                    # 训练集
│   ├── sft_val.jsonl                      # 验证集
│   └── sft_test.jsonl                     # 测试集
│
├── single_turn_data/                      # Step 4 输出
│   ├── sft_train_single.jsonl             # 单轮训练集
│   ├── sft_val_single.jsonl               # 单轮验证集
│   └── split_stats.json                   # 统计信息
│
├── final_split/                           # Step 5 输出 (最终数据)
│   ├── sft_roleplay.jsonl                 # 角色扮演 SFT
│   ├── roleplay_rl.jsonl                  # 角色扮演 RL
│   ├── rm_sft.jsonl                       # RM SFT 训练
│   ├── rm_rl.jsonl                        # RM RL 训练
│   ├── rl_test.jsonl                      # RL 测试集
│   └── split_by_purpose_stats.json        # 统计信息
│
└── README_SFT_PIPELINE.md                 # 本文档
```

---

## 🔧 复现命令

```bash
cd step1_roleplay_sft

# Step 1: 同步 dialogues
python3 sync_dialogues_to_training_samples.py

# Step 2: 转换 SFT 格式
python3 convert_to_sft.py

# Step 3: 分离测试集
python3 split_train_test.py

# Step 4: 拆分单轮
python3 split_to_single_turn.py

# Step 5: 按用途分流
python3 split_by_purpose.py
```

---

## 📝 数据格式

### 输入格式 (sft_train_data.jsonl)

```json
{
  "trace_id": "书名_章节_i_p_i_c_角色名",
  "messages": [
    {"role": "system", "content": "You are {角色名} from {书名}. ===Profile===..."},
    {"role": "assistant", "content": "{角色名}: <role_action>...</role_action>..."},
    {"role": "user", "content": "{其他角色}: <role_action>...</role_action>..."},
    ...
    {"role": "assistant", "content": "<system_thinking>...</system_thinking>{角色名}: ..."}
  ]
}
```

### 单轮格式 (single_turn_data/*.jsonl)

```json
{
  "trace_id": "原始trace_id_turn_N",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<system_thinking>...</system_thinking>..."}
  ],
  "original_trace_id": "原始trace_id",
  "turn_index": N,
  "history_turns": M
}
```

---

## ⚠️ 注意事项

1. **System Thinking**: 只有每个样本的**最后一轮 assistant** 才有 `<system_thinking>`
2. **Role Thinking**: 自己的发言保留 `<role_thinking>`，别人的发言去掉
3. **测试集**: 按 `test_set_coser.json` 精确匹配 (book, i_p, i_c)
4. **单轮拆分**: 一个 N 轮对话拆成 N 个样本，保留历史上下文
