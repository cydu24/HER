#!/usr/bin/env python3
"""
Chat Template 管理 (Jinja2)
支持不同模型的对话格式转换

功能:
1. Jinja Template - 用于发送给 vLLM API 的完整模板
2. 本地格式化 - 使用 Jinja2 渲染消息
3. 内容格式转换 - her/coser 之间的角色扮演标记转换
4. Stop Tokens - 各模板的停止词
"""

from typing import Dict, List, Any, Optional, Union

try:
    from jinja2 import Template
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False
    print("⚠️ jinja2 not installed, some features will be limited")


class ChatTemplateManager:
    """
    统一的 Chat Template 管理器 (基于 Jinja2)
    
    支持的模板类型:
    - qwen/her/chatml: Qwen系列 (<|im_start|>...<|im_end|>)
    - llama3/coser: Llama3系列 (<|start_header_id|>...<|eot_id|>)
    - api/default: 不转换，直接使用messages
    """
    
    # ==========================================
    # 预定义 Jinja Templates (发送给 vLLM API)
    # ==========================================
    JINJA_TEMPLATES = {
        # ChatML / Qwen 格式
        "chatml": """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}""",
        
        "qwen": """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|}
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}""",
        
        # Llama3 格式
        "llama3": """<|begin_of_text|>{% for message in messages %}{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}""",
    }
    
    # ==========================================
    # 模板配置 (用于本地格式化)
    # ==========================================
    TEMPLATES = {
        "qwen": {
            "system": "<|im_start|>system\n{content}<|im_end|>\n",
            "user": "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
            "generation_prefix": "<|im_start|>assistant\n",
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"]
        },
        
        "llama3": {
            "bos": "<|begin_of_text|>",
            "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
            "generation_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"]
        },
    }
    
    # ==========================================
    # 别名映射
    # ==========================================
    ALIASES = {
        # Qwen 系列
        "her": "qwen",
        "her_with_systhink": "qwen",
        "her_without_systhink": "qwen",
        "her_nosys": "qwen",
        "qwen3": "qwen",
        "chatml": "qwen",
        
        # Llama 系列
        "coser": "llama3",
        "llama": "llama3",
        "llama3.1": "llama3",
        
        # API 模式 (不转换)
        "api": None,
        "default": None,
        "openai": None,
    }
    
    # ==========================================
    # 角色扮演内容格式映射
    # ==========================================
    ROLEPLAY_FORMATS = {
        "her": {
            "thinking_start": "<role_thinking>",
            "thinking_end": "</role_thinking>",
            "action_start": "<role_action>",
            "action_end": "</role_action>",
        },
        "coser": {
            "thinking_start": "[",
            "thinking_end": "]",
            "action_start": "(",
            "action_end": ")",
        },
    }
    
    @classmethod
    def get_template_name(cls, name: str) -> Optional[str]:
        """解析模板名称（处理别名）"""
        name = name.lower()
        if name in cls.ALIASES:
            return cls.ALIASES[name]
        if name in cls.TEMPLATES:
            return name
        return None
    
    @classmethod
    def get_jinja_template(cls, template_name: str) -> Optional[str]:
        """
        获取 Jinja Template 字符串 (发送给 vLLM API)
        
        Args:
            template_name: 模板名称或自定义 Jinja 字符串
            
        Returns:
            Jinja template 字符串，用于 vLLM 的 chat_template 参数
        """
        # 检查是否是预定义模板
        if template_name in cls.JINJA_TEMPLATES:
            return cls.JINJA_TEMPLATES[template_name]
        
        # 解析别名
        resolved = cls.get_template_name(template_name)
        if resolved and resolved in cls.JINJA_TEMPLATES:
            return cls.JINJA_TEMPLATES[resolved]
        
        # 如果看起来像自定义 Jinja 模板 (包含 {% )
        if template_name and "{%" in template_name:
            return template_name
        
        return None
    
    @classmethod
    def apply(cls, 
              messages: List[Dict[str, str]], 
              template_name: str = "api",
              add_generation_prompt: bool = True,
              **kwargs) -> Union[List[Dict[str, str]], str]:
        """
        应用 chat template (使用 Jinja2 渲染)
        
        Args:
            messages: OpenAI格式的消息列表
            template_name: 模板名称 (qwen/llama3/api/...)
            add_generation_prompt: 是否添加 assistant 开始标记
            **kwargs: 传递给 Jinja2 模板的额外变量
            
        Returns:
            - API模式: 返回原始 messages
            - 其他模式: 返回格式化的字符串
        """
        resolved_name = cls.get_template_name(template_name)
        
        # API模式直接返回messages
        if resolved_name is None:
            return messages
        
        # 优先使用 Jinja2 渲染
        jinja_str = cls.JINJA_TEMPLATES.get(resolved_name)
        if jinja_str and HAS_JINJA:
            template = Template(jinja_str)
            return template.render(
                messages=messages, 
                add_generation_prompt=add_generation_prompt,
                **kwargs
            )
        
        # 回退到旧的格式化方式
        template = cls.TEMPLATES.get(resolved_name)
        if not template:
            return messages
        
        formatted = ""
        has_system = False
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted += template["system"].format(content=content)
                has_system = True
            elif role == "user":
                if not has_system and resolved_name == "llama3":
                    formatted = "<|begin_of_text|>" + formatted
                formatted += template["user"].format(content=content)
            elif role == "assistant":
                formatted += template["assistant"].format(content=content)
        
        # 添加生成前缀
        if add_generation_prompt and messages and messages[-1]["role"] != "assistant":
            formatted += template.get("generation_prefix", "")
        
        return formatted
    
    @classmethod
    def render(cls,
               messages: List[Dict[str, str]],
               template_name: str,
               add_generation_prompt: bool = True,
               **kwargs) -> str:
        """
        使用 Jinja2 渲染消息 (强制使用 Jinja)
        
        Args:
            messages: OpenAI格式的消息列表
            template_name: 模板名称或自定义 Jinja 字符串
            add_generation_prompt: 是否添加 assistant 开始标记
            **kwargs: 传递给模板的额外变量 (如 bot_name)
            
        Returns:
            渲染后的字符串
        """
        if not HAS_JINJA:
            raise ImportError("jinja2 is required for render(). Install: pip install jinja2")
        
        # 获取 Jinja 模板字符串
        jinja_str = cls.get_jinja_template(template_name)
        if not jinja_str:
            # 如果模板名本身就是 Jinja 字符串
            if "{%" in template_name:
                jinja_str = template_name
            else:
                raise ValueError(f"Unknown template: {template_name}")
        
        template = Template(jinja_str)
        return template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs
        )
    
    @classmethod
    def get_stop_tokens(cls, template_name: str = "api") -> List[str]:
        """获取停止token"""
        resolved_name = cls.get_template_name(template_name)
        if resolved_name and resolved_name in cls.TEMPLATES:
            return cls.TEMPLATES[resolved_name].get("stop_tokens", [])
        return []
    
    @classmethod
    def convert_roleplay_content(cls, 
                                  content: str, 
                                  source_format: str = "her",
                                  target_format: str = "her") -> str:
        """
        转换角色扮演内容格式
        
        支持的格式:
        - her: <role_thinking>...</role_thinking> <role_action>...</role_action>
        - coser: [...] (...)
        
        Args:
            content: 原始内容
            source_format: 源格式 (her/coser)
            target_format: 目标格式 (her/coser)
            
        Returns:
            转换后的内容
        """
        if source_format == target_format:
            return content
        
        source = cls.ROLEPLAY_FORMATS.get(source_format, cls.ROLEPLAY_FORMATS["her"])
        target = cls.ROLEPLAY_FORMATS.get(target_format, cls.ROLEPLAY_FORMATS["her"])
        
        # 替换 thinking 标记
        if source["thinking_start"]:
            content = content.replace(source["thinking_start"], target["thinking_start"])
            content = content.replace(source["thinking_end"], target["thinking_end"])
        
        # 替换 action 标记
        content = content.replace(source["action_start"], target["action_start"])
        content = content.replace(source["action_end"], target["action_end"])
        
        return content
    
    @classmethod
    def apply_roleplay_format(cls,
                               messages: List[Dict[str, str]],
                               target_format: str = "her",
                               source_format: str = "her") -> List[Dict[str, str]]:
        """
        对消息列表应用角色扮演格式转换
        
        Args:
            messages: OpenAI 格式消息列表
            target_format: 目标格式 (her/coser)
            source_format: 源格式
            
        Returns:
            格式转换后的消息列表
        """
        if target_format == source_format:
            return messages
        
        formatted_messages = []
        for msg in messages:
            new_msg = msg.copy()
            new_msg["content"] = cls.convert_roleplay_content(
                msg["content"], 
                source_format=source_format,
                target_format=target_format
            )
            formatted_messages.append(new_msg)
        
        return formatted_messages
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """列出所有支持的模板"""
        templates = list(cls.TEMPLATES.keys())
        templates.extend([k for k, v in cls.ALIASES.items() if v is None])
        return templates
    
    @classmethod
    def list_jinja_templates(cls) -> List[str]:
        """列出所有预定义的 Jinja 模板"""
        return list(cls.JINJA_TEMPLATES.keys())


# ==========================================
# 便捷函数
# ==========================================

def apply_chat_template(messages: List[Dict[str, str]], template: str = "api") -> Any:
    """应用chat template的便捷函数"""
    return ChatTemplateManager.apply(messages, template)


def get_stop_tokens(template: str = "api") -> List[str]:
    """获取停止token的便捷函数"""
    return ChatTemplateManager.get_stop_tokens(template)


def get_jinja_template(template: str) -> Optional[str]:
    """获取Jinja template的便捷函数"""
    return ChatTemplateManager.get_jinja_template(template)


def convert_roleplay(content: str, source: str = "her", target: str = "her") -> str:
    """转换角色扮演格式的便捷函数"""
    return ChatTemplateManager.convert_roleplay_content(content, source, target)


if __name__ == "__main__":
    # 测试
    print("🧪 测试 ChatTemplateManager\n")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! <role_action>waves hand</role_action>"}
    ]
    
    # 测试本地格式化
    print("1. 本地格式化测试:")
    for tmpl in ["api", "qwen", "llama3"]:
        result = ChatTemplateManager.apply(messages, tmpl)
        if isinstance(result, str):
            print(f"   {tmpl}: {result[:80]}...")
        else:
            print(f"   {tmpl}: [原始 messages]")
    
    # 测试 Jinja 模板
    print("\n2. Jinja Template 测试:")
    for tmpl in ["qwen", "llama3"]:
        jinja = ChatTemplateManager.get_jinja_template(tmpl)
        if jinja:
            print(f"   {tmpl}: {jinja[:60]}...")
    
    # 测试角色扮演格式转换
    print("\n3. 角色扮演格式转换:")
    test_content = "Hello! <role_thinking>thinking...</role_thinking> <role_action>waves</role_action>"
    for target in ["her", "coser"]:
        result = ChatTemplateManager.convert_roleplay_content(test_content, "her", target)
        print(f"   → {target}: {result}")
    
    print("\n✅ 测试完成!")
