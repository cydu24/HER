#!/usr/bin/env python3
"""
模型基类定义
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaseModel(ABC):
    """所有模型的基类"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.7,
             max_tokens: int = 512,
             **kwargs) -> Optional[str]:
        """
        发送对话请求
        
        Args:
            messages: OpenAI格式的消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大生成token数
            
        Returns:
            模型生成的文本，失败返回None
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
    
    @property
    def model_type(self) -> str:
        """返回模型类型标识"""
        return self.__class__.__name__.replace("Model", "").lower()

