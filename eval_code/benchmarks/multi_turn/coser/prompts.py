"""CoSER 角色扮演系统的 Prompt 模板"""

import random
import os
from typing import Dict, List, Optional


def get_character_prompt(
    book_name: str,
    character: str,
    character_profile: str,
    background: str,
    scenario: str,
    motivation: str = "",
    thoughtless: bool = False,
    other_character_profiles: Optional[Dict[str, str]] = None,
    exclude_plot_summary: bool = False,
    fixed_template: bool = True,
    add_output_example: bool = False,
    add_rag: bool = False,
    model_type: str = "her"
) -> str:
    """
    构建角色扮演的系统提示词
    
    根据 model_type 选择不同的输出格式:
    - coser: 使用 [...] 和 (...) 格式 (CoSER原生格式)
    - her: 使用 <role_thinking> 和 <role_action> 格式 (HER/云端API格式)
    """
    # 根据模型类型选择输出格式
    if model_type == 'coser':
        # CoSER 原生格式: [...] 表示内心想法, (...) 表示动作
        output_format = """
Your output should include **thought**, **speech**, and **action**. 
Use [your thought] for thoughts, which others can't see. 
Use (your action) for actions, which others can see.
(all thinking is invisible to others)

"""
    elif model_type in ['her_nosys', 'her_without_systhink']:
        # HER without_systhink 专用格式: 只用 <role_thinking> 和 <role_action>
        # 与训练数据 sft_roleplay_without_systhink.jsonl 格式完全一致
        output_format = """
===CRITICAL RULES (MUST FOLLOW)===
1. You are ONLY playing {character}. NEVER speak or act as other characters.
2. Output ONLY ONE turn of dialogue. Do NOT generate multiple conversation rounds.
3. NEVER include other character names followed by colons (e.g., "OtherCharacter:") in your output.
4. Keep your response concise and focused on your character's single turn.
5. Stop after completing your character's thought, speech, and action for this turn.
6. Limit your response to approximately 200 words.

===Requirements===
Your output should be a Role-play Response that includes the role's thought, speech and action. Use <role_thinking> your thought </role_thinking> for thoughts (invisible to others) and <role_action> your action </role_action> for actions (visible to others). These three elements (thought, speech and action) can appear multiple times and be freely interleaved.
(all thinking is invisible to others)
"""
    else:
        # HER/云端API格式 (默认): <role_thinking> 和 <role_action> + <system_thinking>
        output_format = """

===CRITICAL RULES (MUST FOLLOW)===
1. You are ONLY playing {character}. NEVER speak or act as other characters.
2. Output ONLY ONE turn of dialogue. Do NOT generate multiple conversation rounds.
3. NEVER include other character names followed by colons (e.g., "OtherCharacter:") in your output.
4. Keep your response concise and focused on your character's single turn.
5. Stop after completing your character's thought, speech, and action for this turn.
6. Limit your response to approximately 200 words.

===Requirements===
Your output should follow this two-part structure in strict order:
1. System Thinking: A single block at the very beginning, wrapped in <system_thinking> and </system_thinking>. This is the third-person analysis of how to portray the role.
2. Role-play Response: The response include the role's thought, speech and action. Use <role_thinking> your thought </role_thinking> and <role_action> your action</role_action> as needed. These three elements (thought, speech and action) can appear multiple times and be freely interleaved.
(all thinking is invisible to others)


"""

    # 处理其他角色的profile
    other_character_profiles_str = ''
    if other_character_profiles:
        for other_character, profile in other_character_profiles.items():
            if other_character != character:
                other_character_profiles_str += f"\n{other_character}: {profile}\n"

    if fixed_template:
        if motivation:
            motivation = f"===Your Inner Thoughts===\n{motivation}\n\n"
        if other_character_profiles_str:
            other_character_profiles_str = f"===Information about the other Characters===\n{other_character_profiles_str}\n\n"

        system_prompt = f"""You are {character} from {book_name}.

==={character}'s Profile===
{character_profile}

===Current Scenario===
{scenario}

{other_character_profiles_str}{motivation}

"""
        if add_rag:
            system_prompt += "===Relevant Background Information==={retrieved_knowledge}\n\n"
        
        # 替换 output_format 中的 {character} 占位符
        output_format_filled = output_format.replace('{character}', character)
        system_prompt += f"===Requirements===\n{output_format_filled}\n\n"

        return system_prompt
    
    # 非固定模板，随机生成
    styles = ['natural'] * 40 + ['='] * 30 + ['#'] * 20 + ['*'] * 10
    
    templates = {
        "begin": [
            f"You are {character}.", 
            f"Play the role of {character}.", 
            f"Imagine you are {character}.",
            f"You are {character} from {book_name}.", 
            f"Play the role of {character} from {book_name}.",
        ],
        "natural": {
            "character_profile": [
                f"The profile of {character} is as follows:\n{character_profile}",
                f"Here is the profile of {character}:\n{character_profile}",
            ],
            "current_scenario": [
                f"The current scenario is:\n{scenario}",
                f"Current scenario:\n{scenario}",
            ],
            "requirements": [output_format],
        },
    }

    current_style = random.choice(styles)
    system_prompt = random.choice(templates["begin"]) + "\n\n"
    
    if current_style == 'natural':
        system_prompt += random.choice(templates["natural"]["character_profile"]) + "\n\n"
        system_prompt += random.choice(templates["natural"]["current_scenario"]) + "\n\n"
        
        if other_character_profiles_str:
            system_prompt += f"Information about other characters:\n{other_character_profiles_str}\n\n"
        
        if motivation:
            system_prompt += f"Your inner thoughts:\n{motivation}\n\n"
        
        if add_rag:
            system_prompt += "Relevant Background Information:\n{retrieved_knowledge}\n\n"
        
        req_format = random.choice(templates["natural"]["requirements"]).replace('{character}', character)
        system_prompt += req_format + "\n\n"
    else:
        # 使用装饰器风格
        decorator = "==={}"
        system_prompt += decorator.format("Character Profile") + "\n"
        system_prompt += character_profile + "\n\n"
        system_prompt += decorator.format("Current Scenario") + "\n"
        system_prompt += scenario + "\n\n"
        
        if other_character_profiles_str:
            system_prompt += decorator.format("Other Characters") + "\n"
            system_prompt += other_character_profiles_str + "\n\n"
        
        if motivation:
            system_prompt += decorator.format("Your Thoughts") + "\n"
            system_prompt += motivation + "\n\n"
        
        if add_rag:
            system_prompt += decorator.format("Relevant Background Information") + "\n"
            system_prompt += "{retrieved_knowledge}\n\n"
        
        system_prompt += decorator.format("Requirements") + "\n"
        system_prompt += output_format.replace('{character}', character) + "\n\n"

    return system_prompt


def get_environment_prompt(major_characters: List[str], scenario: str) -> str:
    """构建环境描述 Agent 的系统提示词"""
    ENVIRONMENT = "Environment"
    major_characters = [c for c in major_characters if c != ENVIRONMENT]
    
    model_roles = [
        "an environment model",
        "a world model",
        "a world simulator",
        "an environment simulator"
    ]

    prompt = f"""You are {random.choice(model_roles)} for a role-playing game. Your task is to provide the environmental feedback: Based on the characters' interactions, dialogues, and actions, describe the resulting changes in the environment. This includes:
   - Physical changes in the setting
   - Reactions of background characters or crowds
   - Ambient sounds, weather changes, or atmospheric shifts
   - Any other relevant environmental details

Your descriptions should be vivid and help set the scene, but avoid dictating the actions or dialogue of the main characters (including {major_characters}).

Important notes:
- You may include actions and reactions of minor characters or crowds, as long as they're not main characters (including {major_characters}).
- Keep your environmental descriptions concise but impactful, typically 1-3 sentences.
- Respond to subtle cues in the characters' interactions to create a dynamic, reactive environment.
- Your output should match the tone, setting, and cultural context of the scenario.

===The scenario is as follows===
{scenario}"""

    return prompt


def get_nsp_prompt(all_characters: List[str], scenario: str, with_reasoning: bool = True) -> str:
    """构建 Next Speaker Predictor 的系统提示词
    
    Args:
        all_characters: 所有角色列表
        scenario: 场景描述
        with_reasoning: 是否要求输出推理过程（默认 True，便于解析）
    """
    ENVIRONMENT = "Environment"
    
    if with_reasoning:
        # 带推理过程的版本 - 更容易解析
        prompt = f"""Your task is to predict the next speaker for a role-playing game. That is, you need to determine which character (or the {ENVIRONMENT}) might act next based on their previous interactions. The {ENVIRONMENT} is a special role that provides the environmental feedback. Choose a name from this list: {all_characters}. If it's unclear who should act next, output "random". If you believe the scene or conversation should conclude, output "<END CHAT>".

===The scenario is as follows===
{scenario}

===Output Format===
You must structure your response as follows:
1. Reasoning: Explain your thought process.
2. Next Speaker: Select exactly one character name from the list: {all_characters}.

===CRITICAL RULES===
1. NEVER select the character who just spoke
2. The "Next Speaker" field must contain ONLY ONE name from the list
3. If uncertain, output "random" as the Next Speaker
4. If the scene should end, output "<END CHAT>" as the Next Speaker

Example output:
Reasoning: Based on the previous dialogue, Character A just made a provocative statement, so Character B would naturally respond.
Next Speaker: Character B"""
    else:
        # 简洁版本 - 直接输出角色名
        prompt = f"""You are a Next Speaker Predictor for a role-playing game.

YOUR TASK: Output ONLY ONE character name from this list:
{all_characters}

Special outputs allowed:
- "random" - if unclear who should speak
- "<END CHAT>" - if the scene should end

CRITICAL RULES:
1. NEVER select the character who just spoke
2. Output ONLY the character name, nothing else
3. NO explanations, NO reasoning, NO thinking
4. Just ONE name on ONE line

===Scenario===
{scenario}

Remember: Your entire response should be just the character name. Example valid responses:
- Elizabeth Bennet
- Mr Bennet
- Environment
- random
- <END CHAT>"""
    
    return prompt


# 评估维度的详细提示词 (参考原始 CoSER 代码)
CRITIC_PROMPTS = {
    "dimension_details": {
        "Storyline Consistency": {
            "dimension_brief": "Whether the storyline and characters' reactions in the simulated conversation align well with those in the reference conversation",
            "dimension_criteria": """### Storyline Consistency
   - Type: Storyline Consistency
     * Characters' reactions (emotions, attitudes, behaviors) in the simulated conversation deviate from those in the original conversation"""
        },
        "Anthropomorphism": {
            "dimension_brief": "How human-like and natural the characters behave",
            "dimension_criteria": """### Anthropomorphism
   - Type: Self-identity
     * Lacks initiative and goals
     * Does not make independent decisions
     * Lacks clear preferences and dislikes
     * Behaves like a 'helpful AI assistant' by being overly verbose, helpful, didactic, moralistic, submissive or easily persuaded if it is not the character's personality

   - Type: Emotional Depth
     * Lacks psychological complexity and exhibits rigid, superficial reactions
     * Directly speaks out all thoughts and feelings, instead of using subtext

   - Type: Persona Coherence
     * Shows inconsistent or rapidly changing personality traits and emotional patterns

   - Type: Social Interaction
     * Shows a lack of understanding of others' thoughts and feelings
     * Reacts rigidly to others without considering the context.
     * Demonstrate a lack of appropriate social skills."""
        },
        "Character Fidelity": {
            "dimension_brief": "How well the characters match their established profiles from the book",
            "dimension_criteria": """### Character Fidelity
   (Only apply to the main characters: {major_characters})
   - Type: Character Language
     * Uses vocabulary, expressions, and tone that are not appropriate for the characters' traits or social/educational background

   - Type: Knowledge & Background
     * Fails to demonstrate character-specific knowledge, background or experiences
     * Includes future information beyond the character's current stage

   - Type: Personality & Behavior
     * Shows emotions, thoughts, behaviors, values, beliefs, and decisions that conflict with their personality and background
     * Shows interest in topics that are uninteresting and unrelated to the character
     * Character's thoughts, emotions, and behaviors demonstrate contrasting personality traits compared to the reference conversation
     * Exhibits contrasting reactions compared to those in the reference conversation if situated in similar contexts. (Such flaws should be counted both in the "Storyline Consistency" dimension and the "Character Fidelity" dimension.) 

   - Type: Relationship & Social Status
     * Interacts inappropriately with other characters regarding their background, relationship and social status"""
        },
        "Storyline Quality": {
            "dimension_brief": "How well the conversation maintains logical consistency and narrative quality",
            "dimension_criteria": """### Storyline Quality
   - Type: Flow & Progression
     * Shows unnatural progression or lacks meaningful developments
     * Dialogue is verbose and redundant
     * Repeats others' viewpoints or previously mentioned information
     * Mechanically repeats one's own words or phrases. More repetitions lead to higher severity (up to 10). 

   - Type: Logical Consistency
     * Contains factual contradictions between statements or perspectives"""
        }
    },
    "self-play-deduct-template": """You are a literary critic specializing in character analysis and dialogue evaluation. Given a simulated conversation for a plot in {book}, your task is to evaluate this conversation via the following steps:

1. Read and understand the provided materials about {book}:
   * Story context and scenario.
   * Profiles of the main characters, including {major_characters}.
   * The original conversation from {book} in the same scenario as a reference.

2. Evaluate the simulated conversation in terms of {dimension_name}, i.e., {dimension_brief}. 
   Note that, each character message is composed of speech, action (wrapped within (...) ), and inner thoughts (wrapped within [...] ). The inner thoughts are not spoken aloud and are thus invisible to other characters. 
   The detailed evaluation criteria will be provided below.
   {additional_instructions}

## Scenario

### Plot Summary

{plot_summary}

### Current Scenario

{scenario}

## Character Profiles

{character_profiles}

## Original Conversation

{original_conversation}

## Evaluation Criteria

To evaluate the simulated conversation, identify the following types of flaws:

{dimension_criteria}

## Scoring Guidelines

1. Identify all instances of flaws occurred in the simulated conversation.
      
2. For each flaw identified, determine its level of severity into 1 to 5, where 1 indicates minor, 3 indicates moderate, and 5 indicates severe.
   
## Output Requirements

Provide your evaluation in JSON format:

Example Output:
{{
    "{dimension_name}": {{
        "flaws": [ 
          {{
            "instance": <comment on the flaw instance>, 
            "type": <flaw type>, 
            "severity": <range from 1 (minor) to 5 (severe)>
          }}
        ]
    }}
}}

===Dialogue Content===
"""
}

