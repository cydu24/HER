# HER-Eval: Hierarchical Evaluation for Roleplay Models

A comprehensive evaluation framework for roleplay language models, featuring multi-turn dialogue simulation and fine-grained evaluation metrics.

## Features

- **Multi-Turn Evaluation**: Supports multi-agent conversation simulation
- **CoSER Benchmark**: Group Conversation Ability (GCA) evaluation
- **Flexible Model Support**: Works with vLLM, OpenAI API, Anthropic, and any OpenAI-compatible endpoint
- **Caching & Resumption**: Built-in caching for long-running evaluations

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Models

Edit `configs/models.yaml` to add your models:

```yaml
models:
  # Local vLLM deployment
  my-roleplay-model:
    type: vllm
    base_url: http://localhost:8000
    chat_template: her  # Options: her, coser, api
  
  # Judge model (for evaluation)
  qwen-judge:
    type: vllm
    base_url: http://judge-server:8000
    chat_template: api
```

### 2. Run CoSER Evaluation

```bash
# Full evaluation (inference + evaluation)
python run_coser.py --actor my-roleplay-model --judge qwen-judge

# Inference only
python run_coser.py --actor my-roleplay-model --simulation-only

# Evaluation only (using cached inference results)
python run_coser.py --actor my-roleplay-model --evaluation-only --simulation-file results/coser/simulation_xxx.json

# With custom parameters
python run_coser.py \
    --actor my-roleplay-model \
    --judge qwen-judge \
    --max-rounds 20 \
    --num-samples 100 \
    --workers 50 \
    --format her
```

## Directory Structure

```
HER-EVAL/
├── benchmarks/
│   ├── multi_turn/
│   │   └── coser/          # CoSER GCA benchmark
│   └── single_turn/        # Single-turn benchmarks
├── models/
│   ├── api_models.py       # OpenAI/Anthropic adapters
│   ├── vllm_models.py      # vLLM adapter
│   └── factory.py          # Model factory
├── utils/                  # Utility functions
├── configs/
│   ├── models.yaml         # Model configurations
│   └── benchmarks.yaml     # Benchmark settings
├── data/
│   └── coser/              # CoSER test data
└── run_coser.py            # CoSER evaluation script
```

## Chat Template Formats

The framework supports multiple roleplay formats:

| Format | Description |
|--------|-------------|
| `her` | HER format with `<role_thinking>` and `<role_action>` tags |
| `coser` | CoSER format (compatible with Llama-3) |
| `api` | Standard OpenAI chat format |

## Evaluation Metrics

### CoSER (Group Conversation Ability)
- **Consistency**: Character consistency across turns
- **Engagement**: Dialogue engagement quality
- **Naturalness**: Response naturalness
- **Relevance**: Topic relevance

## Model Configuration Examples

### vLLM Local Deployment
```yaml
my-model:
  type: vllm
  base_url: http://localhost:8000
  chat_template: her
```

### OpenAI API
```yaml
gpt-4:
  type: openai
  model_name: gpt-4
  # Set OPENAI_API_KEY environment variable
```

### Anthropic Claude
```yaml
claude-3:
  type: anthropic
  model_name: claude-3-opus-20240229
  # Set ANTHROPIC_API_KEY environment variable
```

### Load Balanced vLLM
```yaml
distributed-model:
  type: vllm
  base_urls:
    - http://server1:8000
    - http://server2:8000
  chat_template: her
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `FULL_LOG` | Set to "1" for verbose logging |

## Output Files

Results are saved to the `results/` directory:

- `simulation_*.json`: Inference results (dialogue histories)
- `evaluation_*.json`: Evaluation scores and details

## Citation

If you use this evaluation framework, please cite:

```bibtex
@article{her-eval,
  title={HER-Eval: Hierarchical Evaluation for Roleplay Models},
  author={...},
  year={2024}
}
```

## License

MIT License

