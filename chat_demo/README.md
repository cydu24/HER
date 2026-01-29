# Interactive Chat Demo

This directory contains an interactive chat tool to test the HER model with character role-playing scenarios.

## Quick Start

```bash
# Basic chat (uses 200 CoSER scenarios from classic books)
python chat_demo.py

# Show system thinking process
python chat_demo.py --show-think

# Show role thinking
python chat_demo.py --show-rolethink

# Use simple built-in scenarios (2 scenarios: Pride and Prejudice, The Great Gatsby)
python chat_demo.py --simple
```

## Features

- **Character Role-Playing**: Chat with AI as characters from classic literature
- **Multi-turn Dialogue**: Maintains conversation history and context
- **Dual-layer Thinking Display**: Optional display of system thinking and role thinking
- **Format Transformation**: Converts XML tags to readable format:
  - `<role_thinking>` → `[inner thought]`
  - `<role_action>` → `(physical action)`
- **Auto-save**: Saves conversation logs on exit

## Usage

### Interactive Mode

```bash
python chat_demo.py
```

The script will prompt you to:
1. Choose a scenario (book and scene)
2. Select which character the AI should play
3. Select which character you want to play
4. Start chatting!

### Commands During Chat

| Command | Function |
|---------|----------|
| `quit` / `exit` / `q` | Exit chat |
| `clear` | Clear conversation history |
| `history` | View current conversation history |
| `prompt` | View full prompt |

## Example Scenarios

### CoSER Dataset (200 Scenarios)

By default, the demo uses the **CoSER test dataset** (`coser_scenarios.json`) with 200 rich scenarios from classic literature:

- **Pride and Prejudice** (Elizabeth Bennet, Mr. Darcy, Mr. Bennet, etc.)
- **A Game of Thrones** (Jon Snow, Tyrion Lannister, Daenerys Targaryen, etc.)
- **The Great Gatsby** (Jay Gatsby, Nick Carraway, Daisy Buchanan, etc.)
- **To Kill a Mockingbird** (Atticus Finch, Scout Finch, etc.)
- **1984** (Winston Smith, Julia, O'Brien, etc.)
- **Harry Potter** (Harry Potter, Hermione Granger, Ron Weasley, etc.)
- **The Lord of the Rings** (Frodo, Gandalf, Aragorn, etc.)
- And 150+ more scenarios from renowned novels!

Each scenario includes:
- **Book title** and author context
- **Scene description** with detailed setting
- **Character profiles** for all participants
- **Initial character thoughts** and motivations
- **Topic/situation** summary

### Built-in Scenarios (2 Simple Examples)

If you prefer simpler scenarios or want to test without the full dataset, use `--simple` flag to load 2 basic scenarios:
- Pride and Prejudice (Mr. Bennet and Elizabeth)
- The Great Gatsby (Gatsby and Nick Carraway)

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path` | Path to HER model directory | `.` (current dir) |
| `--show-think` | Show `<system_thinking>` | False |
| `--show-rolethink` | Show `<role_thinking>` | False |
| `--scenario` | Scenario index | Interactive |
| `--character` | Character index | Interactive |
| `--simple` | Use 2 built-in scenarios instead of 200 CoSER scenarios | False |

## Output Format

The model generates responses with:

1. **System Thinking** (optional display):
   - Third-person analysis of how to portray the character
   - Planning and reasoning about the response

2. **Role Response**:
   - **Role Thinking** `[...]`: Character's inner thoughts (invisible to others)
   - **Role Action** `(...)`: Physical actions and expressions
   - **Speech**: Natural dialogue

### Example Output

```
════════════════════════════════════════════════════════════════════════════════
🎭 【Elizabeth Bennet's Response】
════════════════════════════════════════════════════════════════════════════════
[His tone is light, but the air feels heavy. I cannot let him see how much
Lady Catherine's intrusion still stings.]
(takes a steadying breath, smoothing the folds of her dress)
I believe I can manage, Father. Though I must admit, I am curious about
what this letter contains.
════════════════════════════════════════════════════════════════════════════════
```

## Requirements

- Python 3.8+
- transformers
- torch

Install dependencies:
```bash
pip install transformers torch
```

## File Structure

```
chat_demo/
├── README.md                # This file
├── chat_demo.py             # Main chat script
├── scenarios.json           # Built-in scenarios (auto-created)
└── chat_logs/               # Saved chat logs (auto-created)
    └── {book}_{character}_{timestamp}.txt
```

## Getting CoSER Scenarios (Optional)

To use the full 200 CoSER scenarios from classic literature, download `coser_scenarios.json`:

```bash
# Download from HuggingFace
wget https://huggingface.co/datasets/xxx/HER-data/resolve/main/coser_scenarios.json

# Or use the built-in scenarios (--simple flag)
python chat_demo.py --simple
```

## Notes

1. **System Thinking**: Used for training and analysis, not included in conversation history
2. **Role Thinking**: Character's inner thoughts, invisible to other characters
3. **Role Action**: Physical behaviors visible to others
4. **Speech**: What the character says out loud

## Tips for Best Results

- Stay in character when chatting
- Provide context in your messages
- Use the character's background knowledge
- Be patient - the model generates thoughtful responses with reasoning

## Troubleshooting

**Model not loading?**
- Ensure the model files are in the correct directory
- Check that you have enough GPU memory

**Empty responses?**
- Try adjusting temperature (default: 0.7)
- Check the prompt format

**Inconsistent character behavior?**
- Review the character profile
- Ensure your messages align with the scenario context
