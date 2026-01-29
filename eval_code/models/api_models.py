#!/usr/bin/env python3
"""
API Models - OpenAI Compatible Interface

Supports:
- OpenAI API
- Anthropic Claude API
- Any OpenAI-compatible endpoint (vLLM, etc.)
"""

import os
import requests
import time
import random
from functools import wraps
from typing import Dict, List, Optional, Any
from .base import BaseModel


# ============================================
# Retry Mechanism
# ============================================

def retry_on_error(max_retries: int = 5, 
                   base_delay: float = 1.0, 
                   max_delay: float = 30.0,
                   retryable_status: tuple = (429, 500, 502, 503, 504)):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        retryable_status: HTTP status codes to retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    last_exception = e
                    status_code = e.response.status_code if e.response else 0
                    
                    if status_code == 400:
                        print(f"❌ Bad request (400), not retrying: {e}")
                        return None
                    
                    if status_code in retryable_status:
                        if attempt < max_retries:
                            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                            print(f"⚠️  HTTP {status_code}, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})...")
                            time.sleep(delay)
                            continue
                    
                    print(f"❌ HTTP error: {e}")
                    return None
                    
                except requests.exceptions.Timeout as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"⚠️  Timeout, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    print(f"❌ Timeout after {max_retries} retries")
                    return None
                    
                except requests.exceptions.ConnectionError as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"⚠️  Connection error, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    print(f"❌ Connection error: {e}")
                    return None
                    
                except Exception as e:
                    print(f"❌ Unknown error: {e}")
                    return None
            
            return None
        return wrapper
    return decorator


# Pre-configured API models (examples)
API_CONFIGS = {
    "gpt-4": {
        "model_name": "gpt-4",
        "type": "openai"
    },
    "gpt-4-turbo": {
        "model_name": "gpt-4-turbo",
        "type": "openai"
    },
    "claude-3-opus": {
        "model_name": "claude-3-opus-20240229",
        "type": "anthropic"
    },
    "claude-3-sonnet": {
        "model_name": "claude-3-sonnet-20240229",
        "type": "anthropic"
    },
}


class OpenAIModel(BaseModel):
    """OpenAI API Model"""
    
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.7,
             max_tokens: int = 512,
             **kwargs) -> Optional[str]:
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ OpenAI API Error [{self.model_name}]: {e}")
            return None


class AnthropicModel(BaseModel):
    """Anthropic Claude API Model"""
    
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.7,
             max_tokens: int = 512,
             **kwargs) -> Optional[str]:
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Separate system message
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        
        payload = {
            "model": self.model_name,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if system_msg:
            payload["system"] = system_msg
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
        except Exception as e:
            print(f"❌ Anthropic API Error [{self.model_name}]: {e}")
            return None


class OpenAICompatibleModel(BaseModel):
    """
    Generic OpenAI-compatible API Model
    
    Works with any endpoint that implements the OpenAI chat completions API,
    including vLLM, LocalAI, Ollama, etc.
    
    Usage:
        model = OpenAICompatibleModel(
            model_name="your-model",
            base_url="http://localhost:8000/v1",
            api_key="your-api-key"  # Optional for local models
        )
        response = model.chat([{"role": "user", "content": "Hello!"}])
    """
    
    def __init__(self,
                 model_name: str,
                 base_url: str,
                 api_key: str = None,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv("API_KEY", "")
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.7,
             max_tokens: int = 512,
             **kwargs) -> Optional[str]:
        
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle OpenAI format
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0].get("message", {}).get("content")
            
            return None
            
        except Exception as e:
            print(f"❌ API Error [{self.model_name}]: {e}")
            return None

