"""CoSER Benchmark - Given-Circumstance Acting Evaluation

完整实现 CoSER 评测流程：
1. 多角色对话模拟 (GCA Simulation)
2. LLM-as-Judge 四维度评估
3. 支持缓存和断点续传
4. 支持推理和评估分离

模型类型支持：
- coser: CoSER 原生格式 [...] + (...)
- her: HER 格式 <role_thinking> + <role_action>
- qwen: Qwen 格式 (ChatML)
- llama3: Llama3 格式
- api: 标准 API 格式 (GPT/Claude/Gemini)
"""

import json
import random
import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..base import (
    MultiTurnBenchmark,
    BenchmarkConfig,
    InferenceResult,
    EvaluationResult,
    CacheManager
)
from .prompts import (
    get_character_prompt,
    get_environment_prompt,
    get_nsp_prompt,
    CRITIC_PROMPTS
)
from .utils import (
    remove_system_thinking,
    parse_nsp_response,
    extract_last_speaker,
    remove_inner_thoughts,
    remove_role_thinking,
    remove_long_role_thinking,
    add_speaker_name,
    calculate_bleu_rouge,
    apply_qwen_chat_template,
    get_model_type,
    get_stop_tokens,
    convert_her_format,
    convert_coser_to_her_format,
    convert_to_coser_format
)

logger = logging.getLogger(__name__)

# 特殊角色
ENVIRONMENT = 'Environment'
NSP = "NSP"
SPECIAL_CHARACTERS = [NSP, ENVIRONMENT]

# 完整日志模式
FULL_LOG = os.environ.get('FULL_LOG', '0') == '1'

# 日志分隔线
LOG_SEP = "=" * 80
LOG_SEP_THIN = "-" * 60


def full_log_print(msg, prefix="📋"):
    """完整日志模式下打印详细信息"""
    if FULL_LOG:
        print(f"{prefix} {msg}")


def detailed_log(title: str, content: str, prefix: str = "📋", max_len: int = None):
    """打印详细的结构化日志"""
    if not FULL_LOG:
        return
    print(f"\n{LOG_SEP_THIN}")
    print(f"{prefix} {title}")
    print(LOG_SEP_THIN)
    if max_len and len(content) > max_len:
        print(f"{content[:max_len]}...\n[截断，共 {len(content)} 字符]")
    else:
        print(content)
    print(LOG_SEP_THIN)


def log_messages(messages: list, title: str = "Messages (ChatTemplate 之前)", truncate: bool = False):
    """打印 messages 列表（OpenAI 格式 JSON）
    
    Args:
        messages: OpenAI 格式的消息列表
        title: 日志标题
        truncate: 是否截断长内容（默认 False，打印完整内容）
    """
    if not FULL_LOG:
        return
    print(f"\n{LOG_SEP}")
    print(f"📨 {title} (共 {len(messages)} 条消息)")
    print(LOG_SEP)
    
    # 打印完整的 JSON 格式
    import json
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        name = msg.get('name', '')
        content = msg.get('content', '')
        
        # 构建显示用的消息对象
        display_msg = {"role": role}
        if name:
            display_msg["name"] = name
        
        # 根据 truncate 参数决定是否截断
        if truncate and len(content) > 500:
            display_msg["content"] = content[:500] + f"...[截断，共{len(content)}字符]"
        else:
            display_msg["content"] = content
        
        print(f"\n[{i}] " + json.dumps(display_msg, ensure_ascii=False, indent=2))
    
    print(f"\n{LOG_SEP}")


def log_messages_json(messages: list, agent_name: str, round_num: int):
    """以 JSON 格式打印完整的消息列表（用于调试多轮对话）"""
    if not FULL_LOG:
        return
    import json
    print(f"\n[LOG:MESSAGES] [{agent_name}] 第 {round_num + 1} 轮")
    print(f"{'🔷'*30}")
    print(f"🔷 [{agent_name}] 第 {round_num} 轮 - 完整消息队列 (共 {len(messages)} 条)")
    print(f"{'🔷'*30}")
    
    for i, msg in enumerate(messages):
        print(f"\n--- Message [{i}] ---")
        # 完整打印，不截断
        formatted = json.dumps(msg, ensure_ascii=False, indent=2)
        print(formatted)
    
    print(f"\n{'🔷'*30}\n")


@dataclass
class CoSERConfig(BenchmarkConfig):
    """CoSER 专属配置"""
    # 模型配置
    actor_model: str = ""  # 角色扮演模型
    env_model: str = ""    # 环境模型
    nsp_model: str = ""    # 下一说话者预测模型
    judge_model: str = ""  # 评估模型
    
    # 模拟配置
    max_rounds: int = 10
    continue_from: int = 0  # 从第几轮开始用模型生成（之前用 ground truth）
    wo_thought: bool = False  # 是否禁用内心想法
    retrieval: Optional[str] = None  # 检索增强类型
    nsp_mode: str = "model"  # NSP模式: "model"=使用模型预测, "random"=随机选择角色
    
    # 评估配置
    eval_workers: int = 100  # 评估并行数（默认100）
    eval_remove_role_thinking: bool = False  # 消融实验：评估时是否移除 role_thinking
    dimensions: List[str] = field(default_factory=lambda: [
        'Storyline Consistency',
        'Anthropomorphism', 
        'Character Fidelity',
        'Storyline Quality'
    ])
    
    # 模型类型: coser, her, qwen, llama3, api
    model_type: str = "her"


class CharacterAgent:
    """角色 Agent - 管理单个角色的对话历史
    
    支持不同模型类型的对话格式处理
    """
    
    def __init__(
        self,
        name: str,
        model: Any,
        system_prompt: str,
        model_type: str = "her"
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.model_type = model_type
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        
        full_log_print(f"角色: {name}", "👤")
        full_log_print(f"模型类型: {model_type}", "🤖")
        full_log_print(f"System Prompt:\n{'='*60}\n{system_prompt}\n{'='*60}", "📝")
    
    def update(self, role: str, content: str, name: Optional[str] = None):
        """更新对话历史"""
        msg = {"role": role, "content": content}
        if name:
            msg["name"] = name
        self.messages.append(msg)
        
        full_log_print(f"[{self.name}] 添加消息: role={role}, name={name}", "💬")
        full_log_print(f"内容: {content[:200]}..." if len(content) > 200 else f"内容: {content}", "📄")
    
    async def chat(self, max_tokens: int = 4096, temperature: float = 0.7, round_num: int = 0) -> str:
        """生成回复
        
        根据模型类型选择不同的调用方式：
        - her: 使用 completions API + Qwen chat template
        - 其他: 使用 chat completions API
        
        Args:
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            round_num: 当前轮次（用于日志）
        """
        try:
            # 📨 打印完整的 OpenAI 格式 JSON 消息队列（不截断）
            if FULL_LOG:
                log_messages_json(self.messages, self.name, round_num)
            
            if self.model_type == 'her':
                # HER 模型使用 completions API + Qwen chat template
                # 这样才能正确输出 <system_think> 标签
                prompt = apply_qwen_chat_template(self.messages)
                
                # 📤 打印 ChatTemplate 之后 (HER 格式)
                if FULL_LOG:
                    # 单行格式 (方便复制测试)
                    prompt_oneline = repr(prompt)
                    print(f"\n[LOG:TEMPLATE] [{self.name}] HER 格式")
                    print(f"{LOG_SEP}")
                    print(f"📤 [{self.name}] ChatTemplate 之后 (HER completions API) - 单行格式")
                    print(LOG_SEP)
                    print(prompt_oneline)
                    print(LOG_SEP)
                    # 多行格式 (方便阅读)
                    print(f"\n📤 [{self.name}] ChatTemplate 之后 (HER completions API) - 多行格式")
                    print(LOG_SEP)
                    print(prompt)
                    print(LOG_SEP)
                
                # HER 模型使用 complete 方法 - 优先用异步版本
                if hasattr(self.model, 'complete_async'):
                    response = await self.model.complete_async(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=get_stop_tokens('her')
                    )
                elif hasattr(self.model, 'complete'):
                    # 同步版本用 asyncio.to_thread 包装
                    response = await asyncio.to_thread(
                        self.model.complete,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=get_stop_tokens('her')
                    )
                else:
                    logger.warning(f"Model doesn't have complete method, falling back to chat")
                    response = await asyncio.to_thread(
                        self.model.chat,
                        messages=self.messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
            else:
                # 其他模型使用 chat API
                if FULL_LOG:
                    import json
                    print(f"\n{LOG_SEP}")
                    print(f"📤 [{self.name}] 发送给 API 的请求 (chat API)")
                    print(LOG_SEP)
                    print(f"模型类型: {self.model_type}")
                    print(f"消息数量: {len(self.messages)}")
                    print(f"\n完整 messages JSON:")
                    print(json.dumps(self.messages, ensure_ascii=False, indent=2))
                    print(LOG_SEP)
                
                # 标准 chat 接口（同步调用，用 asyncio.to_thread 包装）
                response = await asyncio.to_thread(
                    self.model.chat,
                    messages=self.messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            # 📥 打印 Raw Response - 完整打印
            if FULL_LOG:
                print(f"\n[LOG:RESPONSE] [{self.name}]")
                print(f"{LOG_SEP}")
                print(f"📥 [{self.name}] Raw Response (完整)")
                print(LOG_SEP)
                print(response or "(空响应)")
                print(LOG_SEP)
            
            return response or ""
        except Exception as e:
            logger.error(f"Agent {self.name} chat error: {e}")
            if FULL_LOG:
                import traceback
                print(f"❌ [{self.name}] 错误详情:")
                traceback.print_exc()
            return ""


class CoSERBenchmark(MultiTurnBenchmark):
    """CoSER Benchmark 实现
    
    Given-Circumstance Acting (GCA) 评测：
    - 多角色同时对话
    - 环境 Agent 提供场景反馈
    - NSP Agent 预测下一说话者
    - 四维度 LLM Judge 评估
    """
    
    name = "coser"
    
    def __init__(self, config: CoSERConfig):
        # 先设置配置，再调用父类初始化
        self.coser_config = config
        super().__init__(config)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载 CoSER 测试数据"""
        data_path = Path(self.config.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} CoSER test cases")
        return data
    
    def get_sample_id(self, sample: Dict[str, Any], index: int) -> str:
        """获取样本唯一标识"""
        book = sample.get('book', 'unknown')
        i_p = sample.get('plot', {}).get('i_p', index)
        i_c = sample.get('i_c', 0)
        return f"{book}-{i_p}-{i_c}"
    
    async def run_inference_single(
        self,
        sample: Dict[str, Any],
        model: Any,
        env_model: Optional[Any] = None,
        nsp_model: Optional[Any] = None,
        **kwargs
    ) -> InferenceResult:
        """
        对单个样本进行 GCA 模拟
        
        Args:
            sample: 测试样本
            model: Actor 模型（角色扮演）
            env_model: 环境模型（可选，默认使用同一模型）
            nsp_model: NSP 模型（可选，默认使用同一模型）
        """
        sample_id = self.get_sample_id(sample, 0)
        logger.info(f"Starting GCA simulation for {sample_id}")
        
        # 使用传入的模型或回退到主模型
        env_model = env_model or model
        nsp_model = nsp_model or model
        
        # 获取模型类型
        model_type = self.coser_config.model_type
        if not model_type:
            model_type = get_model_type(self.coser_config.actor_model)
        
        full_log_print(f"模型类型: {model_type}", "🔧")
        
        # 提取场景信息
        book_title = sample.get('book', 'Unknown')
        plot = sample.get('plot', {})
        character_profiles = sample.get('character_profiles', {})
        scenario = sample.get('scenario', '')
        speaking_characters = sample.get('speaking_characters_w_env', [])
        major_characters = sample.get('major_characters', [])
        
        # 确保 Environment 在角色列表中
        if ENVIRONMENT not in speaking_characters:
            speaking_characters.append(ENVIRONMENT)
        
        # 构建增强的角色档案
        involved_profiles = {}
        plot_characters = [c['name'] for c in plot.get('key_characters', [])]
        
        for character in speaking_characters:
            if character == ENVIRONMENT:
                continue
            
            profile = character_profiles.get(character, '')
            if character in plot_characters:
                char_info = next(
                    (c for c in plot.get('key_characters', []) if c.get('name') == character),
                    {}
                )
                if 'description' in char_info:
                    profile = char_info['description'].strip() + '\n\n' + profile.strip()
            
            if profile.strip():
                involved_profiles[character] = profile.strip()
        
        # 创建所有 Agent
        agents = {}
        
        for character in speaking_characters + [NSP]:
            if character == NSP:
                # NSP Agent - 使用标准格式
                system_prompt = get_nsp_prompt(speaking_characters, scenario)
                agent_model = nsp_model
                agent_model_type = "her"  # NSP 总是用标准格式
            elif character == ENVIRONMENT:
                # Environment Agent - 使用标准格式
                system_prompt = get_environment_prompt(major_characters, scenario)
                agent_model = env_model
                agent_model_type = "her"  # 环境总是用标准格式
            else:
                # Character Agent - 使用指定的模型类型
                char_profile = involved_profiles.get(character, '')
                
                # 获取角色动机
                motivation = ''
                key_chars = sample.get('key_characters', [])
                for c in key_chars:
                    if c.get('name') == character:
                        motivation = c.get('motivation', '')
                        break
                
                # CoSER 模型不需要输出示例
                add_output_example = model_type not in ['coser']
                
                system_prompt = get_character_prompt(
                    book_name=book_title,
                    character=character,
                    character_profile=char_profile,
                    background=plot.get('summary', ''),
                    scenario=scenario,
                    motivation=motivation,
                    thoughtless=self.coser_config.wo_thought,
                    other_character_profiles=involved_profiles,
                    model_type=model_type,
                    add_output_example=add_output_example
                )
                agent_model = model
                agent_model_type = model_type
            
            agent = CharacterAgent(
                name=character,
                model=agent_model,
                system_prompt=system_prompt,
                model_type=agent_model_type
            )
            agent.update('user', "===Conversation Start===\n\n")
            agents[character] = agent
        
        # 开始对话模拟
        dialogue = []
        current_speaker = speaking_characters[0]
        ground_truth = sample.get('dialogues', [])
        max_rounds = self.coser_config.max_rounds
        continue_from = self.coser_config.continue_from
        
        for i_round in range(max_rounds):
            if current_speaker == "<END CHAT>":
                break
            
            # 🎭 轮次开始日志
            logger.info(f"===Round {i_round + 1}=== Speaker: {current_speaker}")
            if FULL_LOG:
                print(f"\n{'🎭'*20}")
                print(f"🎭 第 {i_round + 1} 轮 | 当前说话者: {current_speaker}")
                print(f"{'🎭'*20}")
            
            # 生成当前说话者的回复
            for actor in [current_speaker, NSP]:
                agent = agents[actor]
                
                # ⚡ 优化：NSP random模式直接跳过模型调用
                if actor == NSP and self.coser_config.nsp_mode == "random" and continue_from <= i_round:
                    response = None  # random模式不需要模型响应
                # 使用 ground truth 或模型生成
                elif continue_from > i_round:
                    if actor == current_speaker:
                        response = ground_truth[i_round]['message'] if i_round < len(ground_truth) else ""
                    else:  # NSP
                        next_char = ground_truth[i_round + 1]['character'] if i_round + 1 < len(ground_truth) else "<END CHAT>"
                        response = next_char
                else:
                    response = await agent.chat(
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        round_num=i_round + 1
                    )
                
                if actor == NSP:
                    # 获取 NSP 模式
                    nsp_mode = self.coser_config.nsp_mode
                    
                    # 🎲 NSP 模式: random - 直接随机选择，不调用模型
                    if nsp_mode == "random":
                        # 排除当前说话者、Environment、Narrator
                        excluded = {current_speaker, ENVIRONMENT, "Narrator", "narrator"}
                        available_chars = [c for c in speaking_characters if c not in excluded]
                        if not available_chars:
                            available_chars = [c for c in speaking_characters if c != current_speaker]
                        
                        if available_chars:
                            next_actor = random.choice(available_chars)
                            current_speaker = next_actor
                        else:
                            next_actor = "<END CHAT>"
                            current_speaker = "<END CHAT>"
                        
                        # 📊 NSP 详细日志 (Random 模式)
                        if FULL_LOG:
                            print(f"\n{LOG_SEP_THIN}")
                            print(f"🎲 NSP (Random 模式)")
                            print(LOG_SEP_THIN)
                            print(f"   候选角色: {available_chars}")
                            print(f"   排除角色: {excluded}")
                            print(f"   ✅ 最终选择: {next_actor}")
                            print(LOG_SEP_THIN)
                        
                        logger.info(f"Next speaker (random): {current_speaker}")
                        
                        dialogue.append({
                            "role": NSP,
                            "content": next_actor,
                            "round": i_round
                        })
                        agent.update('assistant', next_actor)
                    else:
                        # 🤖 NSP 模式: model - 使用模型预测
                        raw_nsp_response = response
                        next_actor = parse_nsp_response(response, speaking_characters)
                        
                        # 简化逻辑：只有在有效且非重复时才采用，否则一律随机
                        final_speaker = None
                        fallback_reason = None
                        
                        if next_actor == "<END CHAT>" and i_round >= 5:
                            final_speaker = "<END CHAT>"
                        elif next_actor in speaking_characters and next_actor != current_speaker:
                            # 有效角色且不重复，采用 NSP 结果
                            final_speaker = next_actor
                        else:
                            # 格式不对 / 重复上一轮角色 / 未知角色 -> 随机选择非上一轮角色
                            candidates = set(major_characters + [ENVIRONMENT]) - {current_speaker}
                            if not candidates:
                                candidates = set(speaking_characters) - {current_speaker}
                            old_speaker = current_speaker
                            final_speaker = random.choice(list(candidates)) if candidates else "<END CHAT>"
                            fallback_reason = f"NSP 选择了 '{next_actor}'，但无效/重复，从 {candidates} 随机选择"
                            logger.info(f"⚠️ NSP 仍选择了上一个说话者 {old_speaker}，response: {response[:100]}")
                        
                        current_speaker = final_speaker
                        
                        # 📊 NSP 详细日志 (Model 模式)
                        if FULL_LOG:
                            print(f"\n[LOG:NSP] Model 模式")
                            print(f"{LOG_SEP_THIN}")
                            print(f"🤖 NSP (Model 模式)")
                            print(LOG_SEP_THIN)
                            print(f"   Raw Response (完整):")
                            print(f"   {raw_nsp_response}")
                            print(f"   ---")
                            print(f"   解析结果: {next_actor}")
                            print(f"   有效角色列表: {speaking_characters}")
                            if fallback_reason:
                                print(f"   ⚠️ 回退原因: {fallback_reason}")
                            print(f"   ✅ 最终选择: {final_speaker}")
                            print(LOG_SEP_THIN)
                        
                        logger.info(f"Next speaker: {current_speaker} (Raw NSP: {next_actor})")
                        
                        dialogue.append({
                            "role": NSP,
                            "content": next_actor,
                            "round": i_round
                        })
                        agent.update('assistant', next_actor)
                else:
                    # 角色/环境回复
                    is_environment = (actor == ENVIRONMENT)
                    
                    # 📊 角色/环境回复详细日志
                    if FULL_LOG:
                        role_type = "🌍 Environment" if is_environment else f"👤 {actor}"
                        print(f"\n{LOG_SEP_THIN}")
                        print(f"{role_type} 回复处理 (第 {i_round + 1} 轮)")
                        print(LOG_SEP_THIN)
                        print(f"[1] Raw Response (完整):")
                        print(response)  # 完整打印，不截断
                    
                    # Environment 响应需要移除 <think> 标签
                    if is_environment:
                        response = remove_system_thinking(response)
                        if FULL_LOG:
                            print(f"\n[1.5] Environment 移除 <think> 后:")
                            print(response)
                    
                    dialogue_entry = {
                        "role": actor,
                        "content": response,
                        "round": i_round
                    }
                    dialogue.append(dialogue_entry)
                    
                    # 📊 打印添加到 dialogue 的 JSON
                    if FULL_LOG:
                        print(f"\n[📋 添加到 dialogue 的消息]")
                        print(json.dumps(dialogue_entry, ensure_ascii=False, indent=2))
                    
                    # 清理回复用于更新其他 agent 的历史
                    # 1. 先处理 long_role_thinking
                    response_clean = remove_long_role_thinking(response)
                    # 2. 移除 system_thinking (保留 role_thinking!)
                    response_clean = remove_system_thinking(response_clean)
                    
                    if FULL_LOG:
                        print(f"\n[2] 移除 system_thinking 后 (保留 role_thinking):")
                        print(response_clean)  # 完整打印
                    
                    # 普通模型需要加角色名前缀
                    response_clean = add_speaker_name(response_clean, actor)
                    if FULL_LOG:
                        print(f"\n[3] 添加说话者前缀后:")
                        print(response_clean)  # 完整打印
                    
                    # 更新所有 agent 的对话历史
                    if FULL_LOG:
                        print(f"\n[4] 更新各 Agent 的对话历史:")
                    
                    for other_actor, other_agent in agents.items():
                        if other_actor == actor:
                            # 给自己：保留完整内容（包括 system_thinking 和 role_thinking）
                            # 训练时模型能看到自己之前的完整输出，测试时也要保持一致
                            response_for_self = response
                            if FULL_LOG:
                                has_system_think = '<system_think>' in response_for_self or '<system_thinking>' in response_for_self
                                print(f"    👤 给自己 ({actor}): 保留完整内容 (含 system_think: {has_system_think})")
                                print(f"       内容前100字: {response_for_self[:100]}...")
                            other_agent.update('assistant', response_for_self, name=actor)
                        else:
                            # 给别人：移除所有内心想法（<role_thinking> 和 [...]），保留动作格式
                            response_for_others = remove_inner_thoughts(response_clean)
                            if FULL_LOG:
                                print(f"    👥 给 {other_actor}: 移除 role_thinking ->")
                                print(f"       {response_for_others}")  # 完整打印
                            other_agent.update('user', response_for_others, name=actor)
                    
                    if FULL_LOG:
                        print(LOG_SEP_THIN)
        
        # 打印完整对话记录
        if FULL_LOG:
            print(f"\n{'='*60}")
            print(f"📊 场景完整对话记录: {book_title} (共 {len(dialogue)} 条消息)")
            print(f"{'='*60}")
            for i, conv in enumerate(dialogue):
                print(f"\n[{i+1}] {conv['role']} (轮次: {conv.get('round', 'N/A')}):")
                print(conv['content'])  # 完整打印
            print(f"\n{'='*60}\n")
        
        return InferenceResult(
            sample_id=sample_id,
            model_name=self.coser_config.actor_model,
            input_data=sample,
            dialogue=dialogue,
            metadata={
                "book_title": book_title,
                "i_p": plot.get('i_p'),
                "i_c": sample.get('i_c'),
                "involved_character_profiles": involved_profiles,
                "total_rounds": len([d for d in dialogue if d['role'] != NSP]),
                "model_type": model_type
            },
            status="completed"
        )
    
    async def evaluate_single(
        self,
        inference_result: InferenceResult,
        judge_model: Optional[Any] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        评估单条推理结果
        
        使用 LLM-as-Judge 进行四维度评估：
        1. Storyline Consistency - 故事线一致性
        2. Anthropomorphism - 拟人化
        3. Character Fidelity - 角色忠实度
        4. Storyline Quality - 故事线质量
        """
        sample_id = inference_result.sample_id
        sample = inference_result.input_data
        simulation = inference_result.dialogue
        metadata = inference_result.metadata
        model_type = metadata.get('model_type', 'her')
        
        logger.info(f"Evaluating {sample_id}")
        
        # 过滤 NSP 消息，准备评估数据
        simulation_filtered = [m for m in simulation if m['role'] != NSP]
        reference = sample.get('dialogues', [])
        
        # 清理 simulation 并统一转换为 CoSER 格式用于评估
        # 1. 移除 system_thinking
        # 2. 将所有格式统一转换为 CoSER 格式:
        #    - HER: <role_thinking> -> [...], <role_action> -> (...)
        #    - CoSER: 保持不变
        simulation_for_eval = []
        for m in simulation_filtered:
            if m['role'] == ENVIRONMENT:
                simulation_for_eval.append(m)
            else:
                # 先移除 system_thinking
                cleaned_content = remove_system_thinking(m['content'])
                # 消融实验：先移除 role_thinking（必须在格式转换之前！）
                if self.coser_config.eval_remove_role_thinking:
                    cleaned_content = remove_role_thinking(cleaned_content)
                # 再统一转换为 CoSER 格式
                cleaned_content = convert_to_coser_format(cleaned_content, model_type)
                simulation_for_eval.append({**m, 'content': cleaned_content})
        
        # 日志记录消融配置
        ablation_mode = "移除role_thinking" if self.coser_config.eval_remove_role_thinking else "保留role_thinking"
        logger.info(f"📝 评估格式转换: {model_type} -> CoSER 格式 ({ablation_mode})")
        
        # 转换为字符串用于评估
        # 所有模型的 content 已经有角色名前缀了，不需要重复添加
        simulation_str = '\n\n'.join([
            m['content'].strip() 
            for m in simulation_for_eval
        ])
        
        # reference 也需要转换为 CoSER 格式，保持与 simulation 一致
        # test_set_her.json 中的 message 是 HER 格式 (<role_thinking>, <role_action>)
        reference_str = '\n\n'.join([
            f"{m['character']}: {convert_to_coser_format(m['message'], 'her')}".strip() 
            for m in reference
        ])
        
        logger.info(f"===Simulation of {sample_id}===\n{simulation_str[:500]}...\n")
        
        # 准备评估上下文
        book_title = metadata.get('book_title', 'Unknown')
        scenario = sample.get('scenario', '')
        plot_summary = sample.get('plot', {}).get('summary', '')
        involved_profiles = metadata.get('involved_character_profiles', {})
        major_characters = sample.get('major_characters', [])
        
        character_profile_str = '\n\n'.join([
            f"### {char}\n\n{profile.strip()}"
            for char, profile in involved_profiles.items()
        ])
        
        # 并发评估四个维度
        dimensions = self.coser_config.dimensions
        eval_results = {}
        
        # 计算非 Environment 的对话轮数（用于分数调整）
        actor_rounds = len([m for m in simulation_for_eval if m['role'] != ENVIRONMENT])
        
        def evaluate_dimension(dimension: str) -> Tuple[str, Dict]:
            """评估单个维度"""
            try:
                critic_prompt = CRITIC_PROMPTS['self-play-deduct-template'].format(
                    book=book_title,
                    plot_summary=plot_summary,
                    scenario=scenario,
                    character_profiles=character_profile_str,
                    original_conversation=reference_str,
                    major_characters=', '.join(major_characters),
                    additional_instructions='',
                    dimension_name=dimension,
                    dimension_brief=CRITIC_PROMPTS['dimension_details'][dimension]['dimension_brief'],
                    dimension_criteria=CRITIC_PROMPTS['dimension_details'][dimension]['dimension_criteria']
                )
                
                # 📊 评估日志 - 输入
                if FULL_LOG:
                    print(f"\n[LOG:EVAL] 维度: {dimension}")
                    print(f"{'⭐'*30}")
                    print(f"⭐ 评估维度: {dimension}")
                    print(f"{'⭐'*30}")
                    print(f"\n📋 Judge System Prompt (完整):")
                    print(critic_prompt)
                    print(f"\n📋 Judge User Input (对话内容):")
                    print(simulation_str)
                    print(f"{'⭐'*30}\n")
                
                # 同步调用 judge model
                print(f"[DEBUG] judge_model={judge_model}, type={type(judge_model)}")
                if judge_model:
                    if hasattr(judge_model, 'chat_sync'):
                        response = judge_model.chat_sync(
                            messages=[
                                {"role": "system", "content": critic_prompt},
                                {"role": "user", "content": simulation_str}
                            ],
                            max_tokens=2048,
                            temperature=0
                        )
                    elif hasattr(judge_model, 'chat'):
                        # 同步调用 chat 方法 (vLLM model 的 chat 是同步的)
                        response = judge_model.chat(
                            messages=[
                                {"role": "system", "content": critic_prompt},
                                {"role": "user", "content": simulation_str}
                            ],
                            max_tokens=2048,
                            temperature=0
                        )
                    else:
                        # 创建新的事件循环 (用于 ThreadPoolExecutor)
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            response = loop.run_until_complete(judge_model.achat(
                            messages=[
                                {"role": "system", "content": critic_prompt},
                                {"role": "user", "content": simulation_str}
                            ],
                            max_tokens=2048,
                            temperature=0
                        ))
                        finally:
                            loop.close()
                    
                    # 📊 评估日志 - 输出
                    if FULL_LOG:
                        print(f"\n{'✨'*30}")
                        print(f"✨ 维度 {dimension} - Judge Raw Response:")
                        print(f"{'✨'*30}")
                        print(response)
                        print(f"{'✨'*30}\n")
                    
                    # 解析 JSON 响应（带长度惩罚）
                    result = self._parse_eval_response(response, dimension, actor_rounds)
                    
                    # 📊 评估日志 - 解析结果
                    if FULL_LOG:
                        print(f"\n📊 维度 {dimension} - 解析结果:")
                        print(f"   分数: {result.get('score', 0):.2f}")
                        print(f"   缺陷数: {len(result.get('flaws', []))}")
                        for i, flaw in enumerate(result.get('flaws', [])[:5]):  # 最多显示5个
                            print(f"   [{i+1}] {flaw}")
                        print()
                else:
                    result = {"flaws": [], "score": 100}
                
                logger.info(f"✅ 维度 {dimension} 完成，分数: {result.get('score', 0):.2f}")
                return dimension, result
                
            except Exception as e:
                logger.error(f"Error evaluating {dimension}: {e}")
                return dimension, {"flaws": [], "score": 0, "error": str(e)}
        
        # 并发执行评估
        with ThreadPoolExecutor(max_workers=self.coser_config.eval_workers) as executor:
            futures = {
                executor.submit(evaluate_dimension, dim): dim 
                for dim in dimensions
            }
            
            for future in as_completed(futures):
                dim, result = future.result()
                eval_results[dim] = result
        
        logger.info(f"✅ 4个维度并发评估完成！")
        
        # 计算 BLEU 和 ROUGE
        bleu, rouge_l = calculate_bleu_rouge(
            reference[self.coser_config.continue_from:],
            simulation_for_eval[self.coser_config.continue_from:]
        )
        
        # 汇总分数
        scores = {
            dim: eval_results[dim].get('score', 0)
            for dim in dimensions
        }
        scores['bleu'] = bleu
        scores['rouge_l'] = rouge_l
        scores['avg'] = sum(scores[d] for d in dimensions) / len(dimensions)
        
        return EvaluationResult(
            sample_id=sample_id,
            inference_result=inference_result,
            scores=scores,
            details={
                "dimension_results": eval_results,
                "simulation_str": simulation_str,
                "reference_str": reference_str
            }
        )
    
    def _parse_eval_response(self, response: str, dimension: str, actor_rounds: int = 0) -> Dict:
        """解析评估模型的 JSON 响应
        
        分数计算公式（参考原始 CoSER 代码）:
        score = max(0, min(100 - (total_severity - 0.3 * actor_rounds) * 5, 100))
        
        长度惩罚：对话轮数越多，允许的缺陷容忍度越高
        """
        import re
        
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                if dimension in data:
                    result = data[dimension]
                    flaws = result.get('flaws', [])
                    
                    # 计算分数（扣分制，带长度惩罚）
                    # 原始公式: score = max(0, min(100 - (total_severity - 0.3 * actor_rounds) * 5, 100))
                    total_severity = sum(
                        f.get('severity', 1)
                        for f in flaws 
                        if isinstance(f.get('severity'), (int, float))
                    )
                    # 长度惩罚：对话轮数越多，允许的缺陷容忍度越高
                    adjusted_severity = total_severity - 0.3 * actor_rounds
                    score = max(0, min(100 - adjusted_severity * 5, 100))
                    result['score'] = score
                    
                    return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
        
        return {"flaws": [], "score": 50}  # 默认中等分数
    
    async def run_full_evaluation(
        self,
        actor_model: Any,
        model_name: str,
        env_model: Optional[Any] = None,
        nsp_model: Optional[Any] = None,
        judge_model: Optional[Any] = None,
        mode: str = "full",
        inference_dir: Optional[str] = None,
        limit: Optional[int] = None,
        skip_cache: bool = False,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行完整的 CoSER 评估流程
        
        Args:
            actor_model: 角色扮演模型
            model_name: 模型名称
            env_model: 环境模型
            nsp_model: NSP 模型
            judge_model: 评估模型
            mode: 运行模式 ("full", "inference", "evaluate")
            inference_dir: 评估模式下的推理结果目录
            limit: 限制样本数量
            skip_cache: 是否跳过缓存
            model_type: 强制指定模型类型 (coser, her, qwen, llama3, api)
        
        Returns:
            评估结果
        """
        # 更新配置
        self.coser_config.actor_model = model_name
        
        # 设置模型类型
        if model_type:
            self.coser_config.model_type = model_type
        elif not self.coser_config.model_type:
            self.coser_config.model_type = get_model_type(model_name)
        
        logger.info(f"Model type: {self.coser_config.model_type}")
        
        result = {
            "benchmark": self.name,
            "model": model_name,
            "model_type": self.coser_config.model_type,
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        }
        
        # 推理阶段
        if mode in ["full", "inference"]:
            inference_results = await self.run_inference(
                model=actor_model,
                model_name=model_name,
                limit=limit,
                skip_cache=skip_cache,
                env_model=env_model,
                nsp_model=nsp_model
            )
            result["inference"] = {
                "total": len(inference_results),
                "completed": len([r for r in inference_results if r.status == "completed"]),
                "failed": len([r for r in inference_results if r.status == "failed"])
            }
        else:
            inference_results = None
        
        # 评估阶段
        if mode in ["full", "evaluate"]:
            eval_results, summary = await self.run_evaluation(
                inference_results=inference_results,
                inference_dir=inference_dir,
                judge_model=judge_model,
                skip_cache=skip_cache
            )
            result["evaluation"] = summary
        
        return result
