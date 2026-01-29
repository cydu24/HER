#!/usr/bin/env python3
"""
vLLM 模型实现
支持本地部署的 vLLM 服务

功能:
1. 支持 ip_dir / base_url / ip_list 三种连接方式
2. 支持发送 Jinja chat_template 到 vLLM API
3. 支持角色扮演内容格式转换 (her/coser)
"""

import os
import glob
import random
import requests
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from .base import BaseModel
from .chat_templates import ChatTemplateManager


class vLLMModel(BaseModel):
    """vLLM 本地部署模型"""
    
    def __init__(self,
                 model_name: str = None,
                 base_url: str = None,
                 base_urls: List[str] = None,
                 ip_dir: str = None,
                 ip_list: List[str] = None,
                 port: int = 8000,
                 chat_template: str = "api",
                 jinja_template: str = None,
                 roleplay_format: str = None,
                 url_path_suffix: str = None,
                 **kwargs):
        """
        初始化 vLLM 模型

        Args:
            model_name: 模型名称 (用于 API 请求的 model 字段)
                       如果为 None，会自动从服务器获取
            base_url: 直接指定服务URL (优先级最高)
            base_urls: 多个服务URL列表 (负载均衡，随机选择)
            ip_dir: IP文件目录 (自动读取 ip_address_*.txt)
            ip_list: IP列表 (随机选择)
            port: 服务端口
            chat_template: chat template 类型 (qwen/llama3/api)
                          用于本地格式化或发送 Jinja 到 vLLM
            jinja_template: 显式指定 Jinja template (覆盖 chat_template)
                           可以是预定义名称 (qwen/llama3) 或自定义字符串
            roleplay_format: 角色扮演内容格式 (her/coser)
                            用于转换消息中的 <role_thinking>/<role_action> 标记
            url_path_suffix: URL 路径后缀 (如 /all)，用于分布式 vLLM 服务
                            如果指定，所有请求将发送到 base_url + url_path_suffix
                            而不是默认的 /v1/chat/completions
        """
        self.port = port
        self.chat_template = chat_template
        self.jinja_template = jinja_template
        self.roleplay_format = roleplay_format or self._infer_roleplay_format(chat_template)
        self.url_path_suffix = url_path_suffix

        # 确定服务URL (支持负载均衡)
        self.base_url = base_url
        self.base_urls = base_urls or []  # 多URL负载均衡
        self.ip_list = ip_list or []

        # 从IP目录读取
        if not self.base_url and ip_dir:
            self.ip_list = self._load_ips_from_dir(ip_dir)

        # 自动获取 model_name
        if not model_name:
            if self.base_url:
                model_name = self._fetch_model_name(self.base_url)
            elif self.base_urls:
                # 从第一个 URL 获取 model_name
                model_name = self._fetch_model_name(self.base_urls[0])

        super().__init__(model_name or "unknown", **kwargs)

        if not self.base_url and not self.base_urls and not self.ip_list:
            print(f"⚠️  Warning: No URL configured for {self.model_name}")
    
    def _fetch_model_name(self, base_url: str) -> Optional[str]:
        """从服务器获取模型名称"""
        try:
            url = f"{base_url.rstrip('/')}/v1/models"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["id"]
        except:
            pass
        return None
    
    def _infer_roleplay_format(self, chat_template: str) -> str:
        """从 chat_template 推断角色扮演格式"""
        template_lower = chat_template.lower()
        if template_lower in ["coser", "llama3", "llama"]:
            return "coser"
        else:
            return "her"  # 默认 her 格式
    
    def _load_ips_from_dir(self, ip_dir: str) -> List[str]:
        """从目录读取IP地址"""
        ips = []
        if os.path.exists(ip_dir):
            pattern = os.path.join(ip_dir, "ip_address_*.txt")
            for ip_file in glob.glob(pattern):
                try:
                    with open(ip_file, 'r') as f:
                        ip = f.read().strip()
                        if ip:
                            ips.append(ip)
                except Exception as e:
                    print(f"⚠️  读取IP文件失败 {ip_file}: {e}")
        return ips
    
    def _get_url(self, endpoint: str = "/v1/chat/completions") -> Optional[str]:
        """获取服务URL

        如果设置了 url_path_suffix，使用它作为endpoint；
        否则使用传入的 endpoint 参数
        """
        # 如果有自定义路径后缀，使用它
        if self.url_path_suffix:
            endpoint = self.url_path_suffix

        # 优先使用单个 base_url
        if self.base_url:
            base = self.base_url.rstrip('/')
            return f"{base}{endpoint}"

        # 多URL负载均衡：随机选择一个
        if self.base_urls:
            base = random.choice(self.base_urls).rstrip('/')
            return f"{base}{endpoint}"

        if self.ip_list:
            ip = random.choice(self.ip_list)
            return f"http://{ip}:{self.port}{endpoint}"

        return None
    
    def _get_effective_jinja_template(self) -> Optional[str]:
        """获取要发送给 vLLM 的 Jinja template"""
        # 优先使用显式指定的 jinja_template
        if self.jinja_template:
            return ChatTemplateManager.get_jinja_template(self.jinja_template)
        
        # 否则从 chat_template 获取
        if self.chat_template and self.chat_template.lower() not in ["api", "default"]:
            return ChatTemplateManager.get_jinja_template(self.chat_template)
        
        return None
    
    def _apply_roleplay_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        应用角色扮演内容格式转换
        
        将消息中的 <role_thinking>/<role_action> 标记转换为目标格式
        """
        if not self.roleplay_format or self.roleplay_format == "her":
            return messages  # her 是默认格式，不需要转换
        
        return ChatTemplateManager.apply_roleplay_format(
            messages, 
            target_format=self.roleplay_format,
            source_format="her"
        )
    
    def _remove_think_tags(self, content: str) -> str:
        """
        移除各种 thinking 标签:
        - <think>...</think> (Qwen3 thinking 模式)
        - <system_think>...</system_think> (Actor 模型可能生成)
        - <system_thinking>...</system_thinking> (Actor 模型可能生成)
        - <role_think>...</role_think> (Actor 模型格式异常，与 <role_thinking> 不同)
        
        注意：保留 <role_thinking> 和 <role_action> 标签，因为它们是角色扮演格式的一部分
        """
        import re
        # 移除 <think>...</think> 标签及其内容
        cleaned = re.sub(r'<think>[\s\S]*?</think>\s*', '', content)
        # 移除 <system_think>...</system_think> 标签及其内容
        cleaned = re.sub(r'<system_think>[\s\S]*?</system_think>\s*', '', cleaned)
        # 移除 <system_thinking>...</system_thinking> 标签及其内容
        cleaned = re.sub(r'<system_thinking>[\s\S]*?</system_thinking>\s*', '', cleaned)
        # 移除 <role_think>...</role_think> 标签及其内容 (注意: 不是 <role_thinking>)
        cleaned = re.sub(r'<role_think>[\s\S]*?</role_think>\s*', '', cleaned)
        return cleaned.strip()
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.7,
             max_tokens: int = 512,
             repetition_penalty: float = 1.15,
             **kwargs) -> Optional[str]:
        """
        发送聊天请求
        
        Args:
            messages: OpenAI 格式消息列表
            temperature: 温度参数
            max_tokens: 最大生成长度
            repetition_penalty: 重复惩罚 (默认1.15，防止重复)
            **kwargs: 其他参数
            
        Returns:
            模型响应文本，失败返回 None
        """
        url = self._get_url("/v1/chat/completions")
        if not url:
            print(f"❌ No URL available for {self.model_name}")
            return None
        
        # 1. 应用角色扮演格式转换
        formatted_messages = self._apply_roleplay_format(messages)
        
        # 2. 构建请求
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stream": False,
            **kwargs
        }
        
        # 3. 添加 Jinja chat_template (如果需要)
        # ⚠️ TEMP FIX: vLLM /v1/chat/completions 不支持运行时传入 chat_template 参数
        # jinja_tpl = self._get_effective_jinja_template()
        # if jinja_tpl:
        #     payload["chat_template"] = jinja_tpl
        
        # 4. 添加停止token
        stop_tokens = ChatTemplateManager.get_stop_tokens(self.chat_template)
        if stop_tokens and "stop" not in kwargs:
            payload["stop"] = stop_tokens
        
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            result = response.json()
            
            # 解析响应
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice:
                    msg = choice["message"]
                    # 优先返回 content，如果为空则尝试 reasoning_content
                    content = msg.get("content")
                    if content:
                        # 移除 Qwen thinking 标签 <think>...</think>
                        content = self._remove_think_tags(content)
                        return content
                    # 处理 reasoning/thinking 模式的响应
                    reasoning = msg.get("reasoning_content")
                    if reasoning:
                        return reasoning
                    return None
                elif "text" in choice:
                    text = choice["text"]
                    # 移除 Qwen thinking 标签
                    text = self._remove_think_tags(text)
                    return text
            
            print(f"❌ No valid response: {result}")
            return None
            
        except requests.exceptions.Timeout:
            print(f"❌ Request timeout (180s) for {self.model_name}")
            return None
        except requests.exceptions.HTTPError as e:
            # 打印详细错误信息
            print(f"❌ vLLM API Error [{self.model_name}]: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Response: {e.response.text[:500]}")
            return None
        except Exception as e:
            print(f"❌ vLLM API Error [{self.model_name}]: {e}")
            return None
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> Optional[str]:
        """
        使用 text completions API
        
        Args:
            prompt: 完整的 prompt 字符串（已经格式化好的）
            temperature: 温度参数
            max_tokens: 最大生成长度
            **kwargs: 其他参数
            
        Returns:
            模型响应文本，失败返回 None
        """
        url = self._get_url("/v1/completions")
        if not url:
            print(f"❌ No URL available for {self.model_name}")
            return None
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs
        }
        
        # 添加停止 tokens
        stop_tokens = ChatTemplateManager.get_stop_tokens(self.chat_template)
        if stop_tokens and "stop" not in kwargs:
            payload["stop"] = stop_tokens
        
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            result = response.json()
            
            # 解析响应
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0].get("text", "")
            
            print(f"❌ No valid response: {result}")
            return None
            
        except requests.exceptions.Timeout:
            print(f"❌ Request timeout (180s) for {self.model_name}")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"❌ vLLM Completions API Error [{self.model_name}]: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Response: {e.response.text[:500]}")
            return None
        except Exception as e:
            print(f"❌ vLLM Completions API Error [{self.model_name}]: {e}")
            return None
    
    async def complete_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        session: aiohttp.ClientSession = None,
        **kwargs
    ) -> Optional[str]:
        """
        异步版本的 text completions API（真正的并发）
        
        Args:
            prompt: 完整的 prompt 字符串
            temperature: 温度参数
            max_tokens: 最大生成长度
            session: 可选的 aiohttp session（复用连接）
            **kwargs: 其他参数
            
        Returns:
            模型响应文本，失败返回 None
        """
        url = self._get_url("/v1/completions")
        if not url:
            return None
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs
        }
        
        # 添加停止 tokens
        stop_tokens = ChatTemplateManager.get_stop_tokens(self.chat_template)
        if stop_tokens and "stop" not in kwargs:
            payload["stop"] = stop_tokens
        
        close_session = False
        if session is None:
            connector = aiohttp.TCPConnector(limit=0)  # 无连接数限制
            session = aiohttp.ClientSession(connector=connector)
            close_session = True
        
        try:
            async with session.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    print(f"❌ vLLM Async Error [{self.model_name}]: {response.status} - {text[:200]}")
                    return None
                
                result = await response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("text", "")
                
                return None
                
        except asyncio.TimeoutError:
            print(f"❌ Async request timeout for {self.model_name}", flush=True)
            raise  # 重新抛出，让上层重试
        except Exception as e:
            print(f"❌ vLLM Async Error [{self.model_name}]: {e}", flush=True)
            raise  # 重新抛出，让上层重试
        finally:
            if close_session:
                await session.close()
    
    def health_check(self) -> bool:
        """检查服务是否可用"""
        url = self._get_url("/v1/models")
        if not url:
            return False
        
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self) -> Optional[Dict]:
        """获取模型信息"""
        url = self._get_url("/v1/models")
        if not url:
            return None
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_actual_model_name(self) -> Optional[str]:
        """从服务器获取实际模型名称"""
        info = self.get_model_info()
        if info and "data" in info and len(info["data"]) > 0:
            return info["data"][0]["id"]
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """获取适配器信息"""
        return {
            "type": "vllm",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "ip_count": len(self.ip_list),
            "port": self.port,
            "chat_template": self.chat_template,
            "jinja_template": (self.jinja_template[:50] + "...") 
                             if self.jinja_template and len(self.jinja_template) > 50 
                             else self.jinja_template,
            "roleplay_format": self.roleplay_format,
        }


class VLLMClient(vLLMModel):
    """
    VLLMClient 别名类
    
    提供与 unified_benchmark/vllm_client.py 相同的接口
    方便迁移现有代码
    """
    
    def __init__(self, 
                 ip_dir: str = None,
                 base_url: str = None,
                 model_name: str = None,
                 chat_template: str = "default",
                 jinja_template: str = None,
                 port: int = 8000,
                 **kwargs):
        """
        兼容 vllm_client.py 的初始化接口
        
        Args:
            ip_dir: IP文件目录
            base_url: 服务URL
            model_name: 模型名称 (如果为 None，将自动获取)
            chat_template: chat template 类型
            jinja_template: Jinja template (预定义名称或自定义字符串)
            port: 服务端口
        """
        # 如果没有指定 model_name，尝试自动获取
        if not model_name and base_url:
            model_name = self._fetch_model_name(base_url)
        
        super().__init__(
            model_name=model_name or "unknown",
            base_url=base_url,
            ip_dir=ip_dir,
            port=port,
            chat_template=chat_template,
            jinja_template=jinja_template,
            **kwargs
        )
        
        # 如果还是没有 model_name，尝试从服务器获取
        if not model_name and self._get_url():
            actual_name = self.get_actual_model_name()
            if actual_name:
                self.model_name = actual_name
    
    @staticmethod
    def _fetch_model_name(base_url: str) -> Optional[str]:
        """从服务器获取模型名称"""
        try:
            url = f"{base_url.rstrip('/')}/v1/models"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["id"]
        except:
            pass
        return None
    
    def chat_completion(self, 
                        messages: List[Dict[str, str]], 
                        temperature: float = 0.0,
                        max_tokens: int = 512,
                        **kwargs) -> Optional[str]:
        """
        兼容 vllm_client.py 的接口
        """
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)


if __name__ == "__main__":
    print("🧪 测试 vLLMModel\n")
    
    # 测试1: 创建模型（不连接）
    model = vLLMModel(
        model_name="test-model",
        base_url="http://localhost:8000",
        chat_template="qwen",
        roleplay_format="coser"
    )
    print(f"Model info: {model.get_info()}")
    
    # 测试2: 角色扮演格式转换
    test_messages = [
        {"role": "user", "content": "Hello! <role_action>waves hand</role_action>"}
    ]
    formatted = model._apply_roleplay_format(test_messages)
    print(f"\n角色扮演格式转换:")
    print(f"  原始: {test_messages[0]['content']}")
    print(f"  转换后: {formatted[0]['content']}")
    
    # 测试3: Jinja template
    print(f"\nJinja template: {model._get_effective_jinja_template()[:60] if model._get_effective_jinja_template() else 'None'}...")
    
    print("\n✅ 测试完成!")
