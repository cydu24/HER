"""CoSER Benchmark - Given-Circumstance Acting Evaluation"""

from .benchmark import CoSERBenchmark
from .prompts import get_character_prompt, get_environment_prompt, get_nsp_prompt
from .utils import (
    remove_system_thinking,
    parse_nsp_response,
    remove_inner_thoughts,
    remove_role_thinking,
    normalize_action_format
)

__all__ = [
    "CoSERBenchmark",
    "get_character_prompt",
    "get_environment_prompt", 
    "get_nsp_prompt",
    "remove_system_thinking",
    "parse_nsp_response",
    "remove_inner_thoughts",
    "remove_role_thinking",
    "normalize_action_format"
]

