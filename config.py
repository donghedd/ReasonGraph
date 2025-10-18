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


def save_api_keys_to_file(api_keys: Dict[str, str], file_path: str = "api_keys.json") -> None:
    """Persist API keys to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(api_keys, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error saving API keys to %s: %s", file_path, exc)
        raise

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

    def update_provider_api_keys(self, api_keys: Dict[str, str]) -> None:
        """Update provider API keys and persist to storage"""
        sanitized: Dict[str, str] = {}
        for provider, key in api_keys.items():
            if not isinstance(provider, str):
                continue
            provider_id = provider.strip().lower()
            if not provider_id:
                continue
            key_value = (key or "").strip()
            if key_value:
                sanitized[provider_id] = key_value

        self.provider_api_keys = sanitized
        save_api_keys_to_file(self.provider_api_keys)

@dataclass
class PlainTextConfig:
    """Configuration specific to Plain Text method (no visualization)"""
    name: str = "纯文本"
    prompt_format: str = '''请用中文直接回答以下问题：
{question}

请确保推理与最终答案均采用自然流畅的中文表达。'''
    example_question: str = "请介绍一下你自己？"

@dataclass
class ChainOfThoughtsConfig:
    """Configuration specific to Chain of Thoughts method"""
    name: str = "逐步思考"
    prompt_format: str = '''请作为严谨的中文推理助手，以“逐步思考”(Chain-of-Thoughts) 的格式回答下列问题，所有输出必须为中文：

问题：{question}

请按以下结构输出：
<step number="1">
[中文推理第 1 步]
</step>
<step number="2">
[中文推理第 2 步]
</step>
...(如需更多步骤请继续添加)
<answer>
[中文最终答案]
</answer>'''
    example_question: str = "昨天一口气修复了48个bug，但是今天只修复了昨天一半数量的bug。 那么算一算，这两天加起来，一共解决了多少个bug？"

@dataclass
class TreeOfThoughtsConfig:
    """Configuration specific to Tree of Thoughts method"""
    name: str = "思维树"
    prompt_format: str = '''请使用 Tree of Thoughts（思维树）进行中文推理，全面探索不同思路并输出结构化结果：

问题：{question}

请按以下 XML 结构输出：
<node id="root">
[中文初步分析]
</node>

<node id="approach1" parent="root">
[中文描述第一种主要思路]
</node>

<node id="approach1.1" parent="approach1">
[中文描述该思路的进一步推演]
</node>

...(可继续添加更多分支，ID 需唯一并正确标注父节点)

<answer>
基于上述全部分支的中文综合结论：
- [说明最优路径及原因]
- [给出中文最终答案]
</answer>'''
    example_question: str = "使用数字 3、3、8、8 以及加减乘除各一次，如何通过合理括号运算得到 24？"

@dataclass
class LeastToMostConfig:
    """Configuration specific to Least-to-Most method"""
    name: str = "由易到难"
    prompt_format: str = '''请使用 “由易到难”(Least-to-Most) 的中文方法解决下列问题：先拆解为若干子问题，再按难度逐步求解。

问题：{question}

请按照以下结构输出：
<step number="1">
<question>[中文描述最简单的子问题]</question>
<reasoning>[中文推理过程]</reasoning>
<answer>[该子问题的中文答案]</answer>
</step>
...(如需更多步骤请继续添加)
<final_answer>
[综合所有步骤所得的中文最终答案]
</final_answer>'''
    example_question: str = "如何创建一个个人网站？"

@dataclass
class SelfRefineConfig:
    """Configuration specific to Self-Refine method"""
    name: str = "自我修订"
    prompt_format: str = '''请分两个阶段完成中文推理与自我修订：

问题：{question}

第一阶段（初稿）：
<step number="1">
[中文推理第 1 步]
</step>
...(如需更多步骤请继续添加)
<answer>
[中文初始答案]
</answer>

第二阶段（复核与优化）：
<revision_check>
- [逐条审查潜在问题或改进点，使用中文描述]
- [给出中文修订意见与理由]
</revision_check>

<revised_step number="[新步骤编号]" revises="[被修订步骤编号]">
[中文修订后的推理内容]
</revised_step>
...(如需更多修订步骤请继续添加)

<revised_answer>
[若答案更新，请在此给出最终中文答案；否则说明保持不变]
</revised_answer>'''
    example_question: str = "写一句科幻短句，并在自我修订后给出更精彩的版本。"

@dataclass
class SelfConsistencyConfig:
    """Configuration specific to Self-consistency method"""
    name: str = "多路径自洽"
    prompt_format: str = '''请使用 Self-Consistency（自洽投票）策略，生成 3 条彼此独立的中文推理路径，并通过多数投票确定最终答案。

问题：{question}

路径 1：
<step number="1">
[中文推理步骤]
</step>
...(如需更多步骤请继续添加)
<answer>
[路径 1 的中文结论]
</answer>

路径 2：
...(结构同上)

路径 3：
...(结构同上)

请确保三条路径相互独立，可给出不同结论；最终答案由中文多数票决定，并在总结部分明确写出。'''
    example_question: str = "单词 strawberrrrrrrrry 中包含多少个字母 r？"

@dataclass
class BeamSearchConfig:
    """Configuration specific to Beam Search method"""
    name: str = "束搜索"
    prompt_format: str = '''请使用 Beam Search（束搜索）策略完成中文推理。请为每个节点提供 0~1 的评分，并计算 path_score（从根节点到当前节点的累积得分）。

问题：{question}

<node id="root" score="[评分]">
[中文初步分析，拆解核心问题]
</node>

# 分支示例
<node id="approach1" parent="root" score="[评分]">
[中文描述第一种主要策略]
</node>

<node id="impl1.1" parent="approach1" score="[评分]">
[中文阐述该策略的具体实施步骤]
</node>

<node id="result1.1" parent="impl1.1" score="[评分]" path_score="[累积得分]">
[中文说明该路径的结果与影响]
</node>

...(根据需要继续添加新的分支与节点，确保正确维护 parent、score 与 path_score 属性)

<answer>
最佳路径（path_score: [最高累积得分]）：
- [指出得分最高的路径编号]
- [中文说明该路径为何最优]
- [输出最终的中文解决方案]
</answer>'''
    example_question: str = "从记者转型为图书编辑有哪些高可行性的路径？请结合评分给出最佳建议。"

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
            # 优先选择 DeepSeek，如未配置则退回到第一个存在模型的提供商
            preferred_provider = next((p for p in api_providers if p["id"] == "deepseek" and p["models"]), None)
            provider_with_models = preferred_provider or next((p for p in api_providers if p["models"]), api_providers[0])
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
