#!/usr/bin/env python3
"""
Interactive Chat Demo for HER Model
Chat with AI characters from classic literature using role-playing scenarios.

Usage:
    python chat_demo.py
    python chat_demo.py --show-think
    python chat_demo.py --show-rolethink
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    GRAY = '\033[90m'
    MAGENTA = '\033[35m'


def remove_system_thinking(text: str) -> str:
    """Remove <system_thinking>...</system_thinking> tags and content"""
    if not text:
        return text
    pattern = r'<system_thinking>.*?</system_thinking>\s*'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned.strip()


def extract_system_thinking(text: str) -> str:
    """Extract system_thinking content (without role tags inside)"""
    if not text:
        return ""
    match = re.search(r'<system_thinking>(.*?)</system_thinking>', text, flags=re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Remove any role tags that might have leaked in
        content = re.sub(r'</?role_\w+>', '', content)
        return content
    return ""


def format_for_display(text: str, show_rolethink: bool = True) -> str:
    """Format for display: replace role_thinking with [], role_action with ()"""
    if not text:
        return text

    result = text

    # Handle role_thinking
    if show_rolethink:
        result = result.replace('<role_thinking>', '[').replace('</role_thinking>', ']')
    else:
        result = re.sub(r'<role_thinking>.*?</role_thinking>', '', result, flags=re.DOTALL)

    # Replace role_action with ()
    result = result.replace('<role_action>', '(').replace('</role_action>', ')')
    result = result.replace('<role_speech>', '').replace('</role_speech>', '')

    return result.strip()


def load_sample_scenarios(use_coser: bool = True):
    """Load or create sample scenarios

    Args:
        use_coser: If True, try to load CoSER scenarios first (200 scenarios from classic books)
    """
    # Try to load CoSER scenarios if available
    if use_coser:
        coser_file = Path(__file__).parent / "coser_scenarios.json"
        if coser_file.exists():
            print(f"{Colors.CYAN}📚 Loading CoSER scenarios (200 book scenes)...{Colors.END}")
            with open(coser_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    # Otherwise, use built-in scenarios
    scenarios_file = Path(__file__).parent / "scenarios.json"

    # If scenarios file exists, load it
    if scenarios_file.exists():
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Otherwise create sample scenarios
    scenarios = [
        {
            "book": "Pride and Prejudice",
            "topic": "Mr. Bennet confronts Elizabeth about Mr. Darcy's proposal",
            "scenario": "The scene is set in Mr. Bennet's private study, a sanctuary of leather-bound books and quiet contemplation. Elizabeth has been summoned unexpectedly, and Mr. Bennet holds a letter that seems to spark his characteristic sardonic amusement.",
            "character_profiles": {
                "Mr Bennet": "Elizabeth's father, known for his sarcastic wit and detachment. Highly intelligent and well-read, preferring the solitude of his library. Known for his biting sarcasm and sardonic humor.",
                "Elizabeth Bennet": "The protagonist, intelligent and strong-willed. Quick-witted with a playful sense of humor. Values honesty and integrity. Maintains composure under pressure."
            },
            "key_characters": [
                {
                    "name": "Mr Bennet",
                    "thought": "It is a delicate matter, this business with Darcy. I must gauge Elizabeth's true feelings without being overly sentimental."
                },
                {
                    "name": "Elizabeth Bennet",
                    "thought": "Father's summoning me at this hour is unusual. I hope this isn't about Lady Catherine's visit."
                }
            ]
        },
        {
            "book": "The Great Gatsby",
            "topic": "Nick Carraway encounters Gatsby at one of his lavish parties",
            "scenario": "The party is in full swing at Gatsby's mansion. Jazz music fills the air, champagne flows freely, and well-dressed guests mingle on the lawn. Nick has been wandering alone, observing the spectacle, when he encounters a mysterious man by the library.",
            "character_profiles": {
                "Jay Gatsby": "The enigmatic millionaire who throws lavish parties. Behind his elegant facade lies a romantic dreamer obsessed with recapturing the past. Charming yet deeply lonely.",
                "Nick Carraway": "The story's narrator, a Yale graduate from the Midwest. Honest, tolerant, and inclined to reserve judgment. Both drawn to and repelled by the excess around him."
            },
            "key_characters": [
                {
                    "name": "Jay Gatsby",
                    "thought": "Another party, another night of waiting. Perhaps tonight she'll come. I must maintain appearances."
                },
                {
                    "name": "Nick Carraway",
                    "thought": "I've never met my host. These parties are magnificent, yet there's something hollow about all this revelry."
                }
            ]
        }
    ]

    # Save scenarios for future use
    with open(scenarios_file, 'w', encoding='utf-8') as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)

    return scenarios


def print_scenarios(scenarios: list):
    """Print available scenarios"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}📚 Available Scenarios{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    for i, s in enumerate(scenarios):
        print(f"{Colors.CYAN}[{i}]{Colors.END} {Colors.BOLD}📖 {s['book']}{Colors.END}")
        print(f"    {Colors.GRAY}{s['topic']}{Colors.END}")
        chars = list(s['character_profiles'].keys())
        print(f"    {Colors.MAGENTA}👥 Characters: {', '.join(chars)}{Colors.END}")
        print()


def print_characters(scenario: dict):
    """Print available characters in the scenario"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}👥 Available Characters - {scenario['book']}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    for i, (name, profile) in enumerate(scenario['character_profiles'].items()):
        print(f"{Colors.CYAN}[{i}]{Colors.END} {Colors.BOLD}{name}{Colors.END}")
        preview = profile[:150] + "..." if len(profile) > 150 else profile
        print(f"    {Colors.GRAY}{preview}{Colors.END}")
        print()


def build_system_prompt(scenario: dict, character_name: str, user_character_name: str) -> str:
    """Build system prompt for the character"""
    book = scenario['book']
    scene = scenario['scenario']
    profiles = scenario['character_profiles']

    char_profile = profiles.get(character_name, "")
    user_profile = profiles.get(user_character_name, "A person interacting with the character.")

    # Find character's initial thought
    char_thought = ""
    for kc in scenario['key_characters']:
        if kc['name'] == character_name:
            char_thought = kc.get('thought', '')
            break

    prompt = f"""You are role-playing as {character_name} from the book "{book}".

==={character_name}'s Profile===
{char_profile}

===Current Scene===
{scene}

===Your Current Thoughts===
{char_thought}

===The Person You Are Interacting With===
{user_character_name}: {user_profile}

===Instructions===
- Stay in character as {character_name} at all times
- Keep responses natural and engaging, consistent with the book's style
- Respond from {character_name}'s perspective
- **IMPORTANT: Speak DIRECTLY to "{user_character_name}" using "you" (second person). Do NOT use third person.**

===Output Format===
Your output should include thought, speech, and action in this two-part structure:

1. System Thinking: A single block at the very beginning, wrapped in <system_thinking> and </system_thinking>. This is third-person analysis of how to portray the character.

2. Role-play Response: The character's actual response including:
   - <role_thinking>inner thoughts</role_thinking> (invisible to others)
   - <role_action>physical actions</role_action> (visible to others)
   - Speech (plain text, what the character says out loud)"""

    return prompt


def save_chat_log(messages: list, scenario: dict, character_name: str, user_character: str):
    """Save chat log to file"""
    log_dir = Path(__file__).parent / "chat_logs"
    log_dir.mkdir(exist_ok=True)

    safe_book = re.sub(r'[^\w\-]', '_', scenario['book'])[:30]
    safe_char = re.sub(r'[^\w\-]', '_', character_name)[:20]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_book}_{safe_char}_{timestamp}.txt"
    filepath = log_dir / filename

    lines = [
        "=" * 80,
        "HER Chat Demo - Conversation Log",
        "=" * 80,
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Book: {scenario['book']}",
        f"AI Character: {character_name}",
        f"User Character: {user_character}",
        "=" * 80,
        "",
        "【Scene】",
        scenario['scenario'][:500],
        "",
        "=" * 80,
        "【Conversation】",
        "=" * 80,
    ]

    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            continue
        elif role == 'user':
            if "===Conversation Start===" not in content:
                lines.append(f"\n【{user_character}】")
                lines.append(content)
        elif role == 'assistant':
            lines.append(f"\n【{character_name}】")
            lines.append(content)

    lines.extend(["\n" + "=" * 80, "--- End of Conversation ---"])

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return filepath


def chat_loop(model, tokenizer, scenario: dict, character_name: str, user_character: str,
              show_think: bool = False, show_rolethink: bool = True):
    """Main chat loop"""
    book = scenario['book']

    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}🎭 Starting Conversation - {book}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.GREEN}You play: {user_character}{Colors.END}")
    print(f"{Colors.MAGENTA}AI plays: {character_name}{Colors.END}")
    print(f"{Colors.GRAY}Show system_thinking: {'Yes' if show_think else 'No'}{Colors.END}")
    print(f"{Colors.GRAY}Show role_thinking: {'Yes' if show_rolethink else 'No'}{Colors.END}")
    print(f"{Colors.GRAY}Commands: 'quit' to exit, 'clear' to reset, 'history' to view{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    # Display scene
    print(f"{Colors.CYAN}📍 Scene:{Colors.END}")
    print(f"{Colors.GRAY}{scenario['scenario'][:300]}...{Colors.END}\n")

    # Build messages
    system_prompt = build_system_prompt(scenario, character_name, user_character)

    # Initial greeting
    greeting = f"*{character_name} looks at you*"
    for kc in scenario.get('key_characters', []):
        if kc['name'] == character_name:
            greeting = f"*enters the scene* Hello, {user_character}."
            break

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "===Conversation Start==="},
        {"role": "assistant", "content": greeting}
    ]

    print(f"{Colors.GREEN}{character_name}:{Colors.END} {greeting}\n")

    while True:
        try:
            user_input = input(f"{Colors.BLUE}{user_character}:{Colors.END} ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.YELLOW}👋 Goodbye!{Colors.END}")
                break

            if user_input.lower() == 'clear':
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "===Conversation Start==="},
                    {"role": "assistant", "content": greeting}
                ]
                print(f"{Colors.YELLOW}🔄 Conversation history cleared{Colors.END}\n")
                print(f"{Colors.GREEN}{character_name}:{Colors.END} {greeting}\n")
                continue

            if user_input.lower() == 'history':
                print(f"\n{Colors.CYAN}📜 Conversation History ({len(messages)} messages):{Colors.END}")
                for i, msg in enumerate(messages[1:], 1):
                    content = msg['content'][:80] + '...' if len(msg['content']) > 80 else msg['content']
                    print(f"  [{i}] {msg['role']}: {content}")
                print()
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Generate response
            print(f"{Colors.GRAY}⏳ Thinking...{Colors.END}", end='\r')

            # Format messages for model
            text = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": "<system_thinking>"}],
                tokenize=False,
                add_generation_prompt=False
            )

            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

                response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=False)

                # Clean up response
                response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

                full_response = "<system_thinking>" + response
                clean_response = remove_system_thinking(full_response)

            except Exception as e:
                print(f"{Colors.RED}❌ Generation failed: {e}{Colors.END}")
                messages.pop()
                continue

            print(" " * 50, end='\r')

            # Display system thinking if requested
            if show_think:
                think_content = extract_system_thinking(full_response)
                if think_content:
                    print(f"\n{Colors.GRAY}{'─'*80}{Colors.END}")
                    print(f"{Colors.GRAY}📝 【System Thinking】{Colors.END}")
                    print(f"{Colors.GRAY}{'─'*80}{Colors.END}")
                    for line in think_content.split('\n')[:10]:  # Limit lines
                        print(f"{Colors.GRAY}  {line}{Colors.END}")
                    print(f"{Colors.GRAY}{'─'*80}{Colors.END}\n")

            # Display character response
            print(f"{Colors.GREEN}{'═'*80}{Colors.END}")
            print(f"{Colors.GREEN}🎭 【{character_name}'s Response】{Colors.END}")
            print(f"{Colors.GREEN}{'═'*80}{Colors.END}")
            display_response = format_for_display(clean_response, show_rolethink=show_rolethink)
            print(f"{Colors.GREEN}{display_response}{Colors.END}")
            print(f"{Colors.GREEN}{'═'*80}{Colors.END}\n")

            messages.append({"role": "assistant", "content": clean_response})

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}👋 Goodbye!{Colors.END}")
            break
        except EOFError:
            print(f"\n{Colors.YELLOW}👋 Goodbye!{Colors.END}")
            break

    return messages


def main():
    parser = argparse.ArgumentParser(description="Interactive Chat Demo for HER Model")
    parser.add_argument("--model-path", type=str, default=".",
                        help="Path to model directory (default: current directory)")
    parser.add_argument("--show-think", action="store_true",
                        help="Show system_thinking")
    parser.add_argument("--show-rolethink", action="store_true",
                        help="Show role_thinking (default: hidden)")
    parser.add_argument("--scenario", type=int, default=None,
                        help="Scenario index (default: interactive selection)")
    parser.add_argument("--character", type=int, default=None,
                        help="Character index (default: interactive selection)")
    parser.add_argument("--simple", action="store_true",
                        help="Use simple built-in scenarios instead of CoSER dataset")

    args = parser.parse_args()

    # Load scenarios
    scenarios = load_sample_scenarios(use_coser=not args.simple)
    print(f"{Colors.GREEN}✅ Loaded {len(scenarios)} scenarios{Colors.END}")

    # Select scenario
    if args.scenario is not None:
        if 0 <= args.scenario < len(scenarios):
            scenario = scenarios[args.scenario]
        else:
            print(f"{Colors.RED}❌ Invalid scenario index{Colors.END}")
            return
    else:
        print_scenarios(scenarios)
        while True:
            try:
                idx = int(input(f"{Colors.CYAN}Select scenario (0-{len(scenarios)-1}): {Colors.END}"))
                if 0 <= idx < len(scenarios):
                    scenario = scenarios[idx]
                    break
                print(f"{Colors.RED}Invalid index{Colors.END}")
            except (ValueError, KeyboardInterrupt, EOFError):
                print(f"\n{Colors.YELLOW}👋 Goodbye!{Colors.END}")
                return

    print(f"\n{Colors.GREEN}✅ Selected: {scenario['book']}{Colors.END}")

    # Select character
    char_names = list(scenario['character_profiles'].keys())
    print_characters(scenario)

    if args.character is not None:
        if 0 <= args.character < len(char_names):
            character_name = char_names[args.character]
        else:
            print(f"{Colors.RED}❌ Invalid character index{Colors.END}")
            return
    else:
        while True:
            try:
                idx = int(input(f"{Colors.CYAN}Select AI character (0-{len(char_names)-1}): {Colors.END}"))
                if 0 <= idx < len(char_names):
                    character_name = char_names[idx]
                    break
                print(f"{Colors.RED}Invalid index{Colors.END}")
            except (ValueError, KeyboardInterrupt, EOFError):
                print(f"\n{Colors.YELLOW}👋 Goodbye!{Colors.END}")
                return

    # Select user character
    remaining_chars = [c for c in char_names if c != character_name]
    if remaining_chars:
        print(f"\n{Colors.CYAN}Who do you want to play?{Colors.END}")
        for i, c in enumerate(remaining_chars):
            print(f"  [{i}] {c}")
        print(f"  [{len(remaining_chars)}] Custom name")

        while True:
            try:
                idx = int(input(f"{Colors.CYAN}Select (0-{len(remaining_chars)}): {Colors.END}"))
                if idx == len(remaining_chars):
                    user_character = input(f"{Colors.CYAN}Your name: {Colors.END}").strip() or "User"
                    break
                elif 0 <= idx < len(remaining_chars):
                    user_character = remaining_chars[idx]
                    break
            except (ValueError, KeyboardInterrupt, EOFError):
                print(f"\n{Colors.YELLOW}👋 Goodbye!{Colors.END}")
                return
    else:
        user_character = "User"

    print(f"\n{Colors.GREEN}✅ AI plays: {character_name}{Colors.END}")
    print(f"{Colors.BLUE}✅ You play: {user_character}{Colors.END}")

    # Load model
    print(f"\n{Colors.CYAN}🔧 Loading model...{Colors.END}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        print(f"{Colors.GREEN}✅ Model loaded successfully{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Model loading failed: {e}{Colors.END}")
        return

    # Start chat
    messages = chat_loop(
        model,
        tokenizer,
        scenario,
        character_name,
        user_character,
        show_think=args.show_think,
        show_rolethink=args.show_rolethink
    )

    # Save log
    if messages and len(messages) > 3:
        try:
            log_path = save_chat_log(messages, scenario, character_name, user_character)
            print(f"\n{Colors.CYAN}📝 Chat log saved: {log_path}{Colors.END}")
        except Exception as e:
            print(f"\n{Colors.RED}❌ Save failed: {e}{Colors.END}")


if __name__ == "__main__":
    main()
