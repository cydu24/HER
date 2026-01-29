#!/usr/bin/env python3
"""
Model Factory
Create model instances from configuration files

Usage:
    # From YAML configuration
    model = ModelFactory.get_model("configs/models.yaml", "my-model")
    
    # Create vLLM model directly
    model = ModelFactory.create_vllm(base_url="http://localhost:8000")
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from .base import BaseModel
from .api_models import OpenAIModel, AnthropicModel, OpenAICompatibleModel, API_CONFIGS
from .vllm_models import vLLMModel


class ModelFactory:
    """Model Factory Class"""

    # Model type mapping
    MODEL_CLASSES = {
        "openai": OpenAIModel,
        "anthropic": AnthropicModel,
        "openai_compatible": OpenAICompatibleModel,
        "vllm": vLLMModel,
    }
    
    @classmethod
    def list_api_models(cls) -> List[str]:
        """List all pre-configured API models"""
        return list(API_CONFIGS.keys())
    
    @classmethod
    def create_vllm(cls,
                    base_url: str = None,
                    base_urls: List[str] = None,
                    model_name: str = None,
                    chat_template: str = "api",
                    jinja_template: str = None,
                    port: int = None,
                    url_path_suffix: str = None,
                    **kwargs) -> vLLMModel:
        """
        Create vLLM model
        
        Args:
            base_url: Service URL
            base_urls: Multiple service URLs (for load balancing)
            model_name: Model name (optional, will auto-detect from server)
            chat_template: Chat template type (qwen/llama3/api)
            jinja_template: Jinja template (predefined name or custom string)
            port: Service port
            url_path_suffix: URL path suffix (e.g., /all) for distributed vLLM
            
        Returns:
            vLLMModel instance
            
        Example:
            model = ModelFactory.create_vllm(base_url="http://localhost:8000")
            response = model.chat([{"role": "user", "content": "Hello!"}])
        """
        return vLLMModel(
            model_name=model_name,
            base_url=base_url,
            base_urls=base_urls,
            port=port,
            chat_template=chat_template,
            jinja_template=jinja_template,
            url_path_suffix=url_path_suffix,
            **kwargs
        )
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseModel:
        """
        Create model from configuration dictionary
        
        Args:
            config: Model configuration dict, must contain "type"
                   - API models need "model_name"
                   - vLLM models support "base_url", "model_name" is optional
            
        Returns:
            Model instance
        """
        model_type = config.get("type", "openai")
        
        # vLLM model
        if model_type == "vllm":
            return cls.create_vllm(
                base_url=config.get("base_url"),
                base_urls=config.get("base_urls"),
                model_name=config.get("model_name"),
                chat_template=config.get("chat_template", "api"),
                jinja_template=config.get("jinja_template"),
                port=config.get("port"),
            )
        
        model_name = config.get("model_name")
        
        # API models require model_name
        if not model_name:
            raise ValueError("model_name is required for non-vLLM models")
        
        if model_type not in cls.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Supported: {list(cls.MODEL_CLASSES.keys())}")
        
        model_class = cls.MODEL_CLASSES[model_type]
        
        # Remove type field, pass rest as parameters
        params = {k: v for k, v in config.items() if k != "type"}
        
        return model_class(**params)
    
    @classmethod
    def from_yaml(cls, 
                  yaml_path: str, 
                  model_key: str = None) -> Dict[str, BaseModel]:
        """
        Load models from YAML configuration file
        
        Args:
            yaml_path: Path to YAML configuration file
            model_key: Specific model name to load, None loads all
            
        Returns:
            Model dictionary {model_key: model_instance}
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        models = {}
        models_config = config.get("models", config)
        
        for key, model_config in models_config.items():
            if model_key and key != model_key:
                continue
            
            try:
                model_type = model_config.get("type", "api")
                if "model_name" not in model_config and model_type != "vllm":
                    model_config["model_name"] = key
                
                models[key] = cls.create(model_config)
                print(f"✅ Loaded model: {key}")
            except Exception as e:
                print(f"❌ Failed to load model {key}: {e}")
        
        return models
    
    @classmethod
    def get_model(cls, 
                  yaml_path: str, 
                  model_key: str,
                  chat_template: str = None,
                  jinja_template: str = None) -> Optional[BaseModel]:
        """
        【Unified model creation entry】Get single model from config file
        
        Supported model_key formats:
        1. Model name in config file (e.g., "my-roleplay-model")
        2. Pre-configured API model name (gpt-4, claude-3-opus, etc.)
        3. vllm: prefix (e.g., "vllm:http://localhost:8000")
        
        Args:
            yaml_path: Path to YAML configuration file
            model_key: Model name
            chat_template: Chat template override (priority over config file)
            jinja_template: Jinja template override
            
        Returns:
            Model instance
        """
        # 0. Handle vllm: prefix
        if model_key.startswith("vllm:"):
            base_url = model_key[5:]
            print(f"🔧 vLLM explicit prefix: {base_url} (template: {chat_template or 'api'})")
            return cls.create_vllm(
                base_url=base_url,
                chat_template=chat_template or "api",
                jinja_template=jinja_template
            )
        
        # 1. Try to load from config file
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            models_config = config.get("models", config)
            model_config = models_config.get(model_key)
            
            if model_config:
                model_type = model_config.get("type", "api")
                
                if model_type == "vllm":
                    effective_template = chat_template or model_config.get("chat_template", "api")
                    effective_jinja = jinja_template or model_config.get("jinja_template")
                    url_suffix = model_config.get("endpoint") or model_config.get("url_path_suffix")

                    base_urls = model_config.get("base_urls")
                    if base_urls:
                        print(f"🔧 Loading vLLM (load balanced): {model_key} ({len(base_urls)} URLs, template: {effective_template})")
                    else:
                        print(f"🔧 Loading vLLM: {model_key} (template: {effective_template})")
                    return cls.create_vllm(
                        base_url=model_config.get("base_url"),
                        base_urls=base_urls,
                        model_name=model_config.get("model_name"),
                        chat_template=effective_template,
                        jinja_template=effective_jinja,
                        port=model_config.get("port"),
                        url_path_suffix=url_suffix,
                    )
                else:
                    # API model
                    print(f"🔧 Loading API model: {model_key}")
                    if "model_name" not in model_config:
                        model_config["model_name"] = model_key
                    return cls.create(model_config)
                    
        except FileNotFoundError:
            pass  # Config file not found, try other methods
        except Exception as e:
            print(f"⚠️ Cannot load {model_key} from config: {e}")
        
        # 2. Check if it's a pre-configured API model
        if model_key in API_CONFIGS:
            print(f"🔧 Using pre-configured API model: {model_key}")
            config = API_CONFIGS[model_key].copy()
            config["type"] = config.get("type", "openai")
            return cls.create(config)
        
        print(f"❌ Model configuration not found: {model_key}")
        return None


# Convenience functions
def load_models(yaml_path: str) -> Dict[str, BaseModel]:
    """Load all models from config file"""
    return ModelFactory.from_yaml(yaml_path)


def get_model(yaml_path: str, model_key: str) -> Optional[BaseModel]:
    """Get single model"""
    return ModelFactory.get_model(yaml_path, model_key)
