from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_api_keys_from_file(file_path: str = "api_keys.json") -> Dict[str, str]:
    """Load API keys from a JSON file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"API keys file {file_path} not found")
            return {}
    except Exception as e:
        logger.error(f"Error loading API keys from {file_path}: {str(e)}")
        return {}

@dataclass
class GeneralConfig:
    """General configuration parameters that are method-independent"""
    provider_model_map: Dict[str, List[str]] = field(default_factory=lambda: {
        # Anthropic
        "anthropic": [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-5-haiku-latest",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
            "claude-3.7-opus-preview"
        ],
        # OpenAI
        "openai": [
            "gpt-5-codex-low",
            "gpt-5-codex-medium",
            "gpt-5-codex-high",
            "gpt-5-minimal",
            "gpt-5-low",
            "gpt-5-medium",
            "gpt-5-high",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "chatgpt-4o-latest"
        ],
        # Google Gemini
        "google": [
            "gemini-2.0-ultra",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-pro-exp-02-05",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-2.0-flash-thinking-exp"
        ],
        # Together AI / Meta / Mixtral
        "together": [
            "meta-llama/Llama-4-405B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            "deepseek-ai/DeepSeek-V3"
        ],
        # Packycode Codex (OpenAI compatible wire API)
        "packycode": [
            "gpt-5-codex",
            "gpt-5-codex-low",
            "gpt-5-codex-medium",
            "gpt-5-codex-high"
        ],
        # DeepSeek
        "deepseek": [
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-math",
            "deepseek-coder"
        ],
        # Qwen
        "qwen": [
            "qwen-max",
            "qwen-max-2025-01-25",
            "qwen-plus",
            "qwen-plus-2025-01-25",
            "qwen-turbo",
            "qwen-turbo-2024-11-01",
            "qwen2.5-72b-instruct",
            "qwen2.5-32b-instruct",
            "qwq-plus"
        ],
        # xAI Grok
        "grok": [
            "grok-2",
            "grok-2-latest",
            "grok-2-mini"
        ]
    })
    provider_display_names: Dict[str, str] = field(default_factory=lambda: {
        "anthropic": "Anthropic",
        "openai": "OpenAI",
        "google": "Google Gemini",
        "together": "Together AI",
        "packycode": "Packycode Codex",
        "deepseek": "DeepSeek",
        "qwen": "Qwen",
        "grok": "Grok"
    })
    max_tokens: int = 4096
    chars_per_line: int = 48
    max_lines: int = 12
    available_models: List[str] = field(init=False, repr=False)
    model_providers: Dict[str, str] = field(init=False, repr=False)
    providers: List[str] = field(init=False, repr=False)

    def __post_init__(self):
        """Derived mappings and load API keys"""
        self.providers = list(self.provider_model_map.keys())
        self.available_models = []
        self.model_providers = {}
        for provider, models in self.provider_model_map.items():
            for model in models:
                self.available_models.append(model)
                self.model_providers[model] = provider

        self.provider_api_keys = load_api_keys_from_file()

    def get_default_api_key(self, provider: str) -> str:
        """Get default API key for specific provider"""
        return self.provider_api_keys.get(provider, "")

@dataclass
class PlainTextConfig:
    """Configuration specific to Plain Text method (no visualization)"""
    name: str = "Plain Text"
    prompt_format: str = '''{question}'''
    example_question: str = "Who are you?"

@dataclass
class ChainOfThoughtsConfig:
    """Configuration specific to Chain of Thoughts method"""
    name: str = "Chain of Thoughts"
    prompt_format: str = '''Please answer the question using the following format by Chain-of-Thoughts, with each step clearly marked:

Question: {question}

Let's solve this step by step:
<step number="1">
[First step of reasoning]
</step>
... (add more steps as needed)
<answer>
[Final answer]
</answer>'''
    example_question: str = "猫娘昨天为了给主人的新项目优化代码，一口气修复了48个bug喵！但是呢，今天因为主人一直在旁边捣乱，一会儿摸摸耳朵一会儿挠挠下巴，让猫娘完全没办法专心嘛……所以只修复了昨天一半数量的bug喵。 那么算一算，猫娘这两天加起来，一共解决了多少个bug呀？"

@dataclass
class TreeOfThoughtsConfig:
    """Configuration specific to Tree of Thoughts method"""
    name: str = "Tree of Thoughts"
    prompt_format: str = '''Please answer the question using Tree of Thoughts reasoning. Consider multiple possible approaches and explore their consequences. Feel free to create as many branches and sub-branches as needed for thorough exploration. Use the following format:

Question: {question}

Let's explore different paths of reasoning:
<node id="root">
[Initial analysis of the problem]
</node>

[Add main approaches with unique IDs (approach1, approach2, etc.)]
<node id="approach1" parent="root">
[First main approach to solve the problem]
</node>

[For each approach, add as many sub-branches as needed using parent references]
<node id="approach1.1" parent="approach1">
[Exploration of a sub-path]
</node>

[Continue adding nodes and exploring paths as needed. You can create deeper levels by extending the ID pattern (e.g., approach1.1.1)]

<answer>
Based on exploring all paths:
- [Explain which path(s) led to the best solution and why]
- [State the final answer]
</answer>'''
    example_question: str = "Using the numbers 3, 3, 8, and 8, find a way to make exactly 24 using basic arithmetic operations (addition, subtraction, multiplication, division). Each number must be used exactly once, and you can use parentheses to control the order of operations."

@dataclass
class LeastToMostConfig:
    """Configuration specific to Least-to-Most method"""
    name: str = "Least to Most"
    prompt_format: str = '''Please solve this question using the Least-to-Most approach. First break down the complex question into simpler sub-questions, then solve them in order from simplest to most complex.

Question: {question}

Let's solve this step by step:
<step number="1">
<question>[First sub-question - should be the simplest]</question>
<reasoning>[Reasoning process for this sub-question]</reasoning>
<answer>[Answer to this sub-question]</answer>
</step>
... (add more steps as needed)
<final_answer>
[Final answer that combines the insights from all steps]
</final_answer>'''
    example_question: str = "How to create a personal website?"

@dataclass
class SelfRefineConfig:
    """Configuration specific to Self-Refine method"""
    name: str = "Self-Refine"
    prompt_format: str = '''Please solve this question step by step, then check your work and revise any mistakes. Use the following format:

Question: {question}

Let's solve this step by step:
<step number="1">
[First step of reasoning]
</step>
... (add more steps as needed)
<answer>
[Initial answer]
</answer>

Now, let's check our work:
<revision_check>
[Examine each step for errors or improvements]
</revision_check>

[If any revisions are needed, add revised steps:]
<revised_step number="[new_step_number]" revises="[original_step_number]">
[Corrected reasoning]
</revised_step>
... (add more revised steps if needed)

[If the answer changes, add the revised answer:]
<revised_answer>
[Updated final answer]
</revised_answer>'''
    example_question: str = "Write a one sentence fiction and then improve it after refine."

@dataclass
class SelfConsistencyConfig:
    """Configuration specific to Self-consistency method"""
    name: str = "Self-consistency"
    prompt_format: str = '''Please solve the question using multiple independent reasoning paths. Generate 3 different Chain-of-Thought solutions and provide the final answer based on majority voting.

Question: {question}

Path 1:
<step number="1">
[First step of reasoning]
</step>
... (add more steps as needed)
<answer>
[Path 1's answer]
</answer>

Path 2:
... (repeat the same format for all 3 paths)

Note: Each path should be independent and may arrive at different answers. The final answer will be determined by majority voting.'''
    example_question: str = "How many r are there in strawberrrrrrrrry?"

@dataclass
class BeamSearchConfig:
    """Configuration specific to Beam Search method"""
    name: str = "Beam Search"
    prompt_format: str = '''Please solve this question using Beam Search reasoning. For each step:
1. Explore multiple paths fully regardless of intermediate scores
2. Assign a score between 0 and 1 to each node based on how promising that step is
3. Calculate path_score for each result by summing scores along the path from root to result
4. The final choice will be based on the highest cumulative path score

Question: {question}

<node id="root" score="[score]">
[Initial analysis - Break down the key aspects of the problem]
</node>

# First approach branch
<node id="approach1" parent="root" score="[score]">
[First approach - Outline the general strategy]
</node>

<node id="impl1.1" parent="approach1" score="[score]">
[Implementation 1.1 - Detail the specific steps and methods]
</node>

<node id="result1.1" parent="impl1.1" score="[score]" path_score="[sum of scores from root to here]">
[Result 1.1 - Describe concrete outcome and effectiveness]
</node>

<node id="impl1.2" parent="approach1" score="[score]">
[Implementation 1.2 - Detail alternative steps and methods]
</node>

<node id="result1.2" parent="impl1.2" score="[score]" path_score="[sum of scores from root to here]">
[Result 1.2 - Describe concrete outcome and effectiveness]
</node>

# Second approach branch
<node id="approach2" parent="root" score="[score]">
[Second approach - Outline an alternative general strategy]
</node>

<node id="impl2.1" parent="approach2" score="[score]">
[Implementation 2.1 - Detail the specific steps and methods]
</node>

<node id="result2.1" parent="impl2.1" score="[score]" path_score="[sum of scores from root to here]">
[Result 2.1 - Describe concrete outcome and effectiveness]
</node>

<node id="impl2.2" parent="approach2" score="[score]">
[Implementation 2.2 - Detail alternative steps and methods]
</node>

<node id="result2.2" parent="impl2.2" score="[score]" path_score="[sum of scores from root to here]">
[Result 2.2 - Describe concrete outcome and effectiveness]
</node>

<answer>
Best path (path_score: [highest_path_score]):
[Identify the path with the highest cumulative score]
[Explain why this path is most effective]
[Provide the final synthesized solution]
</answer>'''
    example_question: str = "Give me two suggestions for transitioning from a journalist to a book editor?"

class ReasoningConfig:
    """Main configuration class that manages both general and method-specific configs"""
    def __init__(self):
        self.general = GeneralConfig()
        self.methods = {
            "cot": ChainOfThoughtsConfig(),
            "tot": TreeOfThoughtsConfig(),
            "scr": SelfConsistencyConfig(),
            "srf": SelfRefineConfig(),
            "l2m": LeastToMostConfig(),
            "bs": BeamSearchConfig(),
            "plain": PlainTextConfig(),
        }
    
    def get_method_config(self, method_id: str) -> Optional[dict]:
        """Get configuration for specific method"""
        method = self.methods.get(method_id)
        if method:
            return {
                "name": method.name,
                "prompt_format": method.prompt_format,
                "example_question": method.example_question
            }
        return None

    def get_initial_values(self) -> dict:
        """Get initial values for UI"""
        # 按提供商聚合模型列表
        provider_models = {}
        for model in self.general.available_models:
            provider_id = self.general.model_providers.get(model)
            if not provider_id:
                # 没有提供商映射的模型直接跳过，避免污染前端列表
                continue
            provider_models.setdefault(provider_id, []).append(model)

        api_providers = []
        for provider_id in self.general.providers:
            models = provider_models.get(provider_id, [])
            api_providers.append({
                "id": provider_id,
                "name": self.general.provider_display_names.get(provider_id, provider_id.title()),
                "models": models,
                "default_api_key": self.general.get_default_api_key(provider_id),
                "requires_api_key": True
            })

        # 如果存在未在 providers 列表中的模型，补充到配置末尾
        extra_providers = set(provider_models.keys()) - set(self.general.providers)
        for provider_id in extra_providers:
            models = provider_models.get(provider_id, [])
            api_providers.append({
                "id": provider_id,
                "name": self.general.provider_display_names.get(provider_id, provider_id.title()),
                "models": models,
                "default_api_key": self.general.get_default_api_key(provider_id),
                "requires_api_key": True
            })

        reasoning_methods = []
        for method_id, method_config in self.methods.items():
            reasoning_methods.append({
                "id": method_id,
                "name": method_config.name,
                "prompt_format": method_config.prompt_format,
                "example_question": method_config.example_question
            })

        default_provider_id = ""
        default_model = ""
        if api_providers:
            # 优先选择第一个存在模型的提供商
            provider_with_models = next((p for p in api_providers if p["models"]), api_providers[0])
            default_provider_id = provider_with_models["id"]
            default_model = provider_with_models["models"][0] if provider_with_models["models"] else ""

        default_method_id = ""
        if reasoning_methods:
            default_method_id = next((m["id"] for m in reasoning_methods if m["id"] == "cot"), reasoning_methods[0]["id"])

        return {
            "api_providers": api_providers,
            "reasoning_methods": reasoning_methods,
            "default_values": {
                "provider": default_provider_id,
                "model": default_model,
                "reasoning_method": default_method_id,
                "max_tokens": self.general.max_tokens,
                "chars_per_line": self.general.chars_per_line,
                "max_lines": self.general.max_lines
            }
        }
    
    def add_method(self, method_id: str, config: Any) -> None:
        """Add a new reasoning method configuration"""
        if method_id not in self.methods:
            self.methods[method_id] = config
        else:
            raise ValueError(f"Method {method_id} already exists")

# Create global config instance
config = ReasoningConfig()


def get_initial_values() -> dict:
    """Module-level helper used by视图函数，返回初始配置"""
    return config.get_initial_values()


def get_method_config(method_id: str) -> Optional[dict]:
    """模块级便捷方法，供外部调用获取指定推理方法配置"""
    return config.get_method_config(method_id)
