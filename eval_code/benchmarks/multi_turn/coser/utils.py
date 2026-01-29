#!/usr/bin/env python3
"""
CoSER 评测工具函数 - 基于原始 CoSER 代码
支持不同模型类型: coser, her, qwen, llama3
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 文本清理函数
# =============================================================================

def remove_system_thinking(dialogue: str) -> str:
    """移除系统级思考内容（支持多种格式和多行）
    
    支持格式:
    - <system_thinking>...</system_thinking> (HER/云端API格式)
    - <system_think>...</system_think> (Qwen SFT 格式)
    - <system_thinking>...</system_think> (混合格式)
    - <system_think>...</system_thinking> (混合格式)
    - <think>...</think> (Qwen Base 原生格式)
    - <thought>...</thought> (其他格式)
    - <role_think>...</role_think> (模型格式异常，与 <role_thinking> 不同)
    """
    cleaned = dialogue
    
    # 移除所有 system_thinking/system_think 变体 (支持任意开闭组合)
    cleaned = re.sub(r'<\s*system[_\s]*think(?:ing)?\s*>.*?</\s*system[_\s]*think(?:ing)?\s*>', '', cleaned, flags=re.DOTALL)
    # 处理未闭合的标签
    cleaned = re.sub(r'<\s*system[_\s]*think(?:ing)?\s*>.*$', '', cleaned, flags=re.DOTALL)
    # 处理孤立的闭合标签
    cleaned = re.sub(r'</\s*system[_\s]*think(?:ing)?\s*>', '', cleaned)
    
    # 移除 <think>...</think>
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    
    # 移除 <thought>...</thought>
    cleaned = re.sub(r'<thought>.*?</thought>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<thought>.*', '', cleaned, flags=re.DOTALL)
    
    # 移除 <role_think>...</role_think> (注意: 不是 <role_thinking>，这是模型格式异常)
    cleaned = re.sub(r'<role_think>.*?</role_think>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<role_think>.*', '', cleaned, flags=re.DOTALL)
    
    # 清理多余空行
    cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))
    cleaned = re.sub(r'\n+', '\n', cleaned)
    
    return cleaned.strip()


def parse_nsp_response(response: str, valid_characters: List[str], max_length: int = 1500) -> str:
    """解析 NSP (Next Speaker Predictor) 的输出，提取角色名
    
    处理各种可能的格式：
    1. 纯角色名: "Elizabeth Bennet"
    2. 带冒号: "Elizabeth Bennet:"
    3. 带思考标签: "<think>...</think>Elizabeth Bennet"
    4. 中文推理: "伊丽莎白·贝内特\n\n**推理** ..."
    5. 换行后跟推理: "Elizabeth Bennet\n\nReasoning: ..."
    6. "Next Speaker: xxx" 格式
    
    Args:
        response: NSP 模型的原始输出
        valid_characters: 有效的角色名列表
        max_length: 最大响应长度，超过则返回 random (默认 1500)
        
    Returns:
        提取出的角色名，或 "random" 如果无法解析
    """
    # 0. 处理空输出
    if not response or not response.strip():
        return "random"
    
    # 0.1 超长响应直接返回 random
    if len(response) > max_length:
        import logging
        logging.getLogger(__name__).warning(f"NSP 响应过长 ({len(response)} > {max_length})，返回 random")
        return "random"
    
    # 1. 先尝试用 extract_last_speaker 解析 "Next Speaker:" 格式
    next_speaker = extract_last_speaker(response)
    if next_speaker:
        # 检查是否是有效角色
        for char in valid_characters:
            if char.lower() == next_speaker.lower():
                return char
            if char.lower() in next_speaker.lower():
                return char
        # 检查特殊指令
        if "end chat" in next_speaker.lower():
            return "<END CHAT>"
        if "random" in next_speaker.lower():
            return "random"
    
    # 2. 移除所有思考标签
    cleaned = remove_system_thinking(response)
    
    # 3. 移除 Markdown 格式
    cleaned = re.sub(r'\*\*.*?\*\*', '', cleaned)  # **xxx**
    cleaned = re.sub(r'\*.*?\*', '', cleaned)      # *xxx*
    
    # 4. 取第一行（推理内容通常在后面）
    first_line = cleaned.strip().split('\n')[0].strip()
    
    # 5. 移除冒号后面的内容
    if ':' in first_line:
        first_line = first_line.split(':')[0].strip()
    
    # 6. 检查特殊指令
    if "<END CHAT>" in first_line.upper() or "END CHAT" in first_line.upper():
        return "<END CHAT>"
    if "random" in first_line.lower():
        return "random"
    
    # 7. 精确匹配有效角色名
    for char in valid_characters:
        if char.lower() == first_line.lower():
            return char
        # 也检查是否包含完整角色名
        if char.lower() in first_line.lower():
            return char
    
    # 8. 如果第一行不匹配，尝试在整个清理后的文本中查找角色名
    for char in valid_characters:
        if char in cleaned:
            return char
    
    # 9. 无法解析
    logger.warning(f"NSP 输出无法解析: '{response[:200]}...'")
    return "random"


def extract_last_speaker(response: str) -> Optional[str]:
    """从 NSP 响应中提取 "Next Speaker" 字段
    
    支持格式:
    - Next Speaker: Elizabeth Bennet
    - **Next Speaker**: Elizabeth Bennet
    - Next Speaker is Elizabeth Bennet
    - ## Next Speaker: Elizabeth Bennet
    
    Returns:
        提取的角色名，或 None 如果无法解析
    """
    # 正则：找 Next Speaker，允许 Markdown，允许冒号或 is
    pattern = r'(?:\*\*|#)?\s*Next\s+Speaker\s*(?:\*\*|#)?\s*(?:[:\-]|\bis\b)?\s*\*?([^\n\*]+)'
    
    # 使用 findall 找出文中所有匹配项
    matches = re.findall(pattern, response, re.IGNORECASE)
    
    if matches:
        # 取最后一个匹配到的结果
        last_match = matches[-1]
        
        # 数据清洗
        clean_name = last_match.strip().rstrip('.,!"\'').replace('*', '')
        
        if clean_name:
            return clean_name
    
    return None


def remove_inner_thoughts(dialogue: str) -> str:
    """移除内心想法内容（支持多种格式，更全面的正则匹配）
    
    支持格式:
    - <system_thinking>...</system_thinking> 及其变体
    - <role_thinking>...</role_thinking> 及其变体  
    - <think>...</think> (Qwen Base 格式)
    - [...] (CoSER 原生格式)
    """
    cleaned = dialogue
    
    # 移除 <system_thinking> 及其变体 (system_think, system thinking 等)
    cleaned = re.sub(r'\s*<\s*system[_\s]+think(?:ing)?\s*>.*?</\s*system[_\s]+think(?:ing)?\s*>\s*', '', cleaned, flags=re.S).strip()
    
    # 移除 <role_thinking> 及其变体 (role_think, role thinking 等)
    cleaned = re.sub(r'\s*<\s*role[_\s]+think(?:ing)?\s*>.*?</\s*role[_\s]+think(?:ing)?\s*>\s*', '', cleaned, flags=re.S).strip()
    
    # 移除其他 <*_thinking> 格式
    cleaned = re.sub(r'\s*<\s*\w*[_\s]*think(?:ing)?\s*>.*?</\s*\w*[_\s]*think(?:ing)?\s*>\s*', '', cleaned, flags=re.S).strip()
    
    # 移除 [...] 格式的内心想法 (CoSER 原生格式)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    
    # 清理多余空行
    cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))
    cleaned = re.sub(r'\n+', '\n', cleaned)
    
    return cleaned.strip()


def remove_role_thinking(dialogue: str, also_remove_brackets: bool = True) -> str:
    """移除 role_thinking，保留其他格式
    
    Args:
        dialogue: 输入文本
        also_remove_brackets: 是否同时移除 [...] 格式（CoSER原生格式的角色思考）
    """
    cleaned = dialogue
    # 移除 <role_thinking>...</role_thinking>
    cleaned = re.sub(r'<role_thinking>.*?</role_thinking>', '', cleaned, flags=re.DOTALL)
    # 同时移除 [...] 格式（CoSER原生格式）
    if also_remove_brackets:
        cleaned = re.sub(r'\[.*?\]', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def convert_her_format(dialogue: str) -> str:
    """将 HER 格式转换为 CoSER 格式
    
    HER 格式 -> CoSER 格式:
    - <role_thinking>...</role_thinking> -> [...]
    - <role_action>...</role_action> -> (...)
    - <role_speech>...</role_speech> -> 直接内容
    
    用于在 Judge 评分时统一格式
    """
    result = dialogue
    
    # <role_thinking>...</role_thinking> -> [...]
    result = result.replace("<role_thinking>", "[").replace("</role_thinking>", "]")
    
    # <role_action>...</role_action> -> (...)
    result = result.replace("<role_action>", "(").replace("</role_action>", ")")
    
    # <role_speech>...</role_speech> -> 直接内容
    result = result.replace('<role_speech>', '').replace('</role_speech>', '')
    
    return result.strip()


def convert_to_coser_format(dialogue: str, model_type: str) -> str:
    """将任意格式转换为 CoSER 格式
    
    统一转换函数，用于评估时统一格式
    
    Args:
        dialogue: 原始对话内容
        model_type: 模型类型 (her, coser, etc.)
    
    Returns:
        CoSER 格式的对话
    """
    if model_type == 'her':
        return convert_her_format(dialogue)
    else:
        # coser 和其他格式保持不变
        return dialogue


def convert_coser_to_her_format(dialogue: str) -> str:
    """将 CoSER 格式转换为 HER 格式
    
    CoSER 格式 -> HER 格式:
    - [...] -> <role_thinking>...</role_thinking>
    - (...) -> <role_action>...</role_action>
    """
    result = dialogue
    
    # [...] -> <role_thinking>...</role_thinking>
    result = re.sub(r'\[([^\]]+)\]', r'<role_thinking>\1</role_thinking>', result)
    
    # (...) -> <role_action>...</role_action>
    result = re.sub(r'\(([^)]+)\)', r'<role_action>\1</role_action>', result)
    
    return result.strip()


def remove_long_role_thinking(dialogue: str) -> str:
    """将 <long_role_thinking>...</long_role_thinking> 转换为 <role_thinking>"""
    cleaned = re.sub(
        r'<long_role_thinking>(.*?)</long_role_thinking>',
        r'<role_thinking>\1</role_thinking>',
        dialogue,
        flags=re.DOTALL
    )
    return cleaned


def normalize_action_format(dialogue: str) -> str:
    """统一动作格式为 (动作) 格式
    
    支持格式转换:
    - <role_action>...</role_action> -> (...)
    - (...) 保持不变 (CoSER 格式)
    """
    # <role_action>...</role_action> -> (...)
    dialogue = re.sub(r'<role_action>(.*?)</role_action>', r'(\1)', dialogue, flags=re.DOTALL)
    
    return dialogue


def add_speaker_name(dialogue: str, speaker: str) -> str:
    """添加说话者前缀（如果没有的话）"""
    # 检查是否已经有说话者前缀
    if any(line.strip().startswith(f"{speaker}:") or line.strip().startswith(f"{speaker}：") 
           for line in dialogue.split('\n')):
        return dialogue
    
    return f"{speaker}: {dialogue}"


# =============================================================================
# 角色名提取
# =============================================================================

def extract_actor_name_from_system(content: str) -> Optional[str]:
    """从 system prompt 中提取角色名"""
    patterns = [
        r'You are ([^\.]+?) from',
        r'You are ([^\.]+?)\.',
        r'Play the role of ([^\.]+)',
        r'Roleplay as ([^\.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            name = match.group(1).strip()
            if ' from ' in name:
                name = name.split(' from ')[0]
            return name
    return None


# =============================================================================
# Qwen/HER 格式转换
# =============================================================================

def format_qwen_prompt(messages: List[Dict[str, str]]) -> str:
    """
    将 messages 格式转换为 Qwen/HER 格式
    
    格式:
    <|im_start|>system
    {system_content}<|im_end|>
    <|im_start|>user
    {user_content}<|im_end|>
    <|im_start|>assistant
    {assistant_content}<|im_end|>
    
    支持 assistant prefill：如果最后一条消息是 assistant 且是短标签（如 <system_think>），
    则作为 prefill 不闭合
    """
    # 常见的 prefill 标签
    PREFILL_TAGS = ['<system_thinking>', '<system_think>', '<think>', '<role_thinking>']
    
    prompt_parts = []
    
    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content']
        name = msg.get('name')
        is_last = (i == len(messages) - 1)
        
        if role == 'system':
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == 'assistant':
            # 检查是否是最后一条且是 prefill 标签
            is_prefill = is_last and content in PREFILL_TAGS
            if is_prefill:
                # Prefill: 不闭合，让模型续写
                prompt_parts.append(f"<|im_start|>assistant\n{content}")
            else:
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        else:  # user
            # 如果有名字，添加前缀
            if name and not content.startswith(f"{name}:"):
                content = f"{name}: {content}"
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
    
    # 只有在最后不是 assistant prefill 时才添加生成提示
    if messages and messages[-1]['role'] == 'assistant' and messages[-1]['content'] in PREFILL_TAGS:
        pass  # 已经添加了 prefill
    else:
        prompt_parts.append("<|im_start|>assistant\n")
    
    return "\n".join(prompt_parts)


# 别名：apply_qwen_chat_template
apply_qwen_chat_template = format_qwen_prompt


# =============================================================================
# Llama3/CoSER 格式转换
# =============================================================================

def format_llama3_prompt(messages: List[Dict[str, str]]) -> str:
    """
    将 messages 格式转换为 Llama3/CoSER 格式
    
    格式:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {assistant_content}<|eot_id|>
    """
    prompt_parts = []
    
    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content']
        name = msg.get('name')
        
        if i == 0:
            header = "<|begin_of_text|>"
        else:
            header = ""
        
        if role == 'system':
            prompt_parts.append(f"{header}<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
        elif role == 'assistant':
            prompt_parts.append(f"{header}<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
        else:  # user
            if name and not content.startswith(f"{name}:"):
                content = f"{name}: {content}"
            prompt_parts.append(f"{header}<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
    
    # 添加生成提示
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    
    return "".join(prompt_parts)


# =============================================================================
# 评测辅助函数
# =============================================================================

def calculate_bleu_rouge(reference: List[Dict], simulation: List[Dict]) -> tuple:
    """计算 BLEU 和 ROUGE-L 分数"""
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        from rouge import Rouge
        
        simulation_str = '\n\n'.join([m['content'].strip('\n') for m in simulation])
        reference_str = '\n\n'.join([f"{m['character']}: {m['message']}".strip('\n') for m in reference])
        
        reference_tokens = word_tokenize(reference_str.lower())
        simulation_tokens = word_tokenize(simulation_str.lower())
        
        bleu = sentence_bleu([reference_tokens], simulation_tokens)
        rouge_l = Rouge().get_scores(simulation_str, reference_str)[0]['rouge-l']['f']
        
        return bleu, rouge_l
    except Exception as e:
        logger.warning(f"Failed to calculate BLEU/ROUGE: {e}")
        return 0.0, 0.0


# =============================================================================
# 模型类型判断
# =============================================================================

def get_model_type(model_name: str) -> str:
    """
    根据模型名称判断模型类型
    
    Returns:
        "coser", "her", "qwen", "llama3", "api"
    """
    model_lower = model_name.lower()
    
    if 'coser' in model_lower:
        return 'coser'
    elif 'her' in model_lower:
        return 'her'
    elif 'qwen' in model_lower:
        return 'qwen'
    elif 'llama' in model_lower:
        return 'llama3'
    elif any(x in model_lower for x in ['gpt', 'claude', 'gemini']):
        return 'api'
    else:
        return 'her'  # 默认使用 HER 格式


def get_output_format_instruction(model_type: str) -> str:
    """
    根据模型类型获取输出格式指令
    
    Args:
        model_type: "coser", "her", "qwen", "llama3", "api"
    
    Returns:
        输出格式说明文本
    """
    if model_type == 'coser' or model_type == 'llama3':
        # CoSER 原生格式: [...] 表示内心想法, (...) 表示动作
        return """
Your output should include **thought**, **speech**, and **action**. 
- Use [your thought] for thoughts, which others can't see
- Use (your action) for actions, which others can see
(all thinking is invisible to others)
"""
    elif model_type == 'her':
        # HER 格式: <system_thinking> + <role_thinking> + <role_action>
        return """
=Requirements=
Your output should follow this two-part structure in strict order:
1. System Thinking: A single block at the very beginning, wrapped in <system_thinking> and </system_thinking>. This is the third-person analysis of how to portray the role.
2. Role-play Response: The response include the role's thought, speech and action. Use <role_thinking> your thought </role_thinking> and <role_action> your action</role_action> as needed. The role's thought should be invisible to others.
"""
    elif model_type == 'qwen':
        # Qwen 格式: <think> + <role_thinking> + <role_action>
        return """
=Requirements=
Your output should follow this two-part structure in strict order:
1. Thinking: A single block at the very beginning, wrapped in <think> and </think>. This is the third-person analysis of how to portray the role.
2. Role-play Response: The response include the role's thought, speech and action. Use <role_thinking> your thought </role_thinking> and <role_action> your action</role_action> as needed. The role's thought should be invisible to others.
"""
    else:
        # API 格式: 无特殊思考格式
        return """
## Output Format
- Output dialogue directly
- Stay in character and respond naturally
"""


def get_stop_tokens(model_type: str) -> List[str]:
    """
    根据模型类型获取停止 token
    """
    if model_type in ['qwen', 'her']:
        return ["<|im_end|>", "<|im_start|>"]
    elif model_type in ['coser', 'llama3']:
        return ["<|eot_id|>", "<|start_header_id|>"]
    else:
        return []


# =============================================================================
# 对话格式转换 (统一数据 → 目标模型格式)
# =============================================================================

def convert_dialogue_format(message: str, source_format: str = "coser", target_format: str = "her") -> str:
    """
    将对话内容从源格式转换为目标格式
    
    原始 CoSER 数据格式:
    - [内心想法] - 内心独白，他人不可见
    - (动作描述) - 动作，他人可见
    - 纯文本 - 对话
    
    转换目标:
    - coser: 保持原样 [思考] (动作)
    - her/qwen: <role_thinking>思考</role_thinking> <role_action>动作</role_action>
    - api: 移除所有标记，只保留动作和对话
    
    Args:
        message: 原始消息内容
        source_format: 源格式 ("coser")
        target_format: 目标格式 ("coser", "her", "api", "qwen", "llama3")
    
    Returns:
        转换后的消息
    """
    if source_format == target_format:
        return message
    
    if target_format == 'coser' or target_format == 'llama3':
        # CoSER/Llama3 格式保持原样: [思考] (动作)
        return message
    
    elif target_format in ['her', 'qwen']:
        # HER/Qwen 格式: <role_thinking>思考</role_thinking> <role_action>动作</role_action>
        result = message
        
        # [思考] → <role_thinking>思考</role_thinking>
        result = re.sub(r'\[([^\]]+)\]', r'<role_thinking>\1</role_thinking>', result)
        
        # (动作) → <role_action>动作</role_action>
        result = re.sub(r'\(([^)]+)\)', r'<role_action>\1</role_action>', result)
        
        return result
    
    elif target_format == 'api':
        # API 格式: 保持 [...] 和 (...) 原样 (与 CoSER 相同)
        return message
    
    else:
        return message


def convert_dialogue_history(
    dialogues: List[Dict[str, str]], 
    target_format: str = "her"
) -> List[Dict[str, str]]:
    """
    转换整个对话历史的格式
    
    Args:
        dialogues: 原始对话列表 [{"character": "...", "message": "..."}, ...]
        target_format: 目标格式
    
    Returns:
        转换后的对话列表
    """
    converted = []
    for d in dialogues:
        converted_msg = convert_dialogue_format(
            d.get("message", ""), 
            source_format="coser", 
            target_format=target_format
        )
        converted.append({
            "character": d.get("character", ""),
            "message": converted_msg
        })
    return converted


def build_initial_messages(
    sample: Dict[str, Any],
    character: str,
    system_prompt: str,
    model_type: str = "her",
    continue_from: int = 0
) -> List[Dict[str, str]]:
    """
    从统一数据构建初始 messages 列表
    
    根据 model_type 转换对话历史格式，然后构建 OpenAI 格式的 messages
    
    Args:
        sample: 统一格式的样本数据
        character: 当前角色名
        system_prompt: 系统提示词
        model_type: 模型类型
        continue_from: 从第几轮开始用模型生成
    
    Returns:
        OpenAI 格式的 messages 列表
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # 获取原始对话历史
    dialogues = sample.get("dialogues", [])
    
    # 转换对话格式
    converted_dialogues = convert_dialogue_history(dialogues[:continue_from], target_format=model_type)
    
    # 添加对话开始标记
    messages.append({"role": "user", "content": "===Conversation Start===\n\n"})
    
    # 添加历史对话
    for d in converted_dialogues:
        speaker = d["character"]
        content = d["message"]
        
        # 判断当前角色的视角
        if speaker == character:
            # 自己说的话 → assistant
            role = "assistant"
            # 自己可以看到自己的内心想法（保留）
        else:
            # 别人说的话 → user
            role = "user"
            # 移除他人的内心想法（对自己不可见）
            content = remove_inner_thoughts(content)
            # 添加说话者前缀
            if not content.startswith(f"{speaker}:"):
                content = f"{speaker}: {content}"
        
        messages.append({"role": role, "content": content, "name": speaker})
    
    return messages
