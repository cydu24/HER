#!/usr/bin/env python3
"""
Models Module

Provides model adapters for various backends:
- OpenAI API
- Anthropic Claude API  
- vLLM (local deployment)
- Any OpenAI-compatible endpoint
"""

from .base import BaseModel
from .api_models import OpenAIModel, AnthropicModel, OpenAICompatibleModel
from .vllm_models import vLLMModel, VLLMClient
from .factory import ModelFactory, load_models, get_model
from .chat_templates import ChatTemplateManager

__all__ = [
    # Base
    "BaseModel",
    # API Models
    "OpenAIModel",
    "AnthropicModel", 
    "OpenAICompatibleModel",
    # vLLM Models
    "vLLMModel",
    "VLLMClient",
    # Factory
    "ModelFactory",
    "load_models",
    "get_model",
    # Templates
    "ChatTemplateManager",
]
