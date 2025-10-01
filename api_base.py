"""
Base API module for handling different API providers.
This module provides a unified interface for interacting with various API providers
like Anthropic, OpenAI, Google Gemini and Together AI.
"""

import json
import os
from abc import ABC, abstractmethod
import logging
import requests
from openai import OpenAI
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standardized API response structure"""
    text: str
    raw_response: Any
    usage: Dict[str, int]
    model: str

class APIError(Exception):
    """Custom exception for API-related errors"""
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} API Error: {message} (Status: {status_code})")

class BaseAPI(ABC):
    """Abstract base class for API interactions"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.provider_name = "base"  # Override in subclasses
        
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the API"""
        pass

    def _format_prompt(self, question: str, prompt_format: Optional[str] = None) -> str:
        """Format the prompt using custom format if provided"""
        if prompt_format:
            return prompt_format.format(question=question)
        
        # Default format if none provided
        return f"""Please answer the question using the following format, with each step clearly marked:

Question: {question}

Let's solve this step by step:
<step number="1">
[First step of reasoning]
</step>
<step number="2">
[Second step of reasoning]
</step>
<step number="3">
[Third step of reasoning]
</step>
... (add more steps as needed)
<answer>
[Final answer]
</answer>

Note:
1. Each step must be wrapped in XML tags <step>
2. Each step must have a number attribute
3. The final answer must be wrapped in <answer> tags
"""

    def _handle_error(self, error: Exception, context: str = "") -> None:
        """Standardized error handling"""
        error_msg = f"{self.provider_name} API error in {context}: {str(error)}"
        logger.error(error_msg)
        raise APIError(str(error), self.provider_name)

class AnthropicAPI(BaseAPI):
    """Class to handle interactions with the Anthropic API"""
    
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        super().__init__(api_key, model)
        self.provider_name = "Anthropic"
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the Anthropic API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": formatted_prompt}],
                "max_tokens": max_tokens
            }
            
            logger.info(f"Sending request to Anthropic API with model {self.model}")
            response = requests.post(self.base_url, headers=self.headers, json=data)
            response.raise_for_status()
            
            response_data = response.json()
            return response_data["content"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            self._handle_error(e, "request")
        except (KeyError, IndexError) as e:
            self._handle_error(e, "response parsing")
        except Exception as e:
            self._handle_error(e, "unexpected")

class OpenAIAPI(BaseAPI):
    """Class to handle interactions with the OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "gpt-5-medium"):
        super().__init__(api_key, model)
        self.provider_name = "OpenAI"
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the OpenAI API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(f"Sending request to OpenAI API with model {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self._handle_error(e, "request or response processing")

class GeminiAPI(BaseAPI):
    """Class to handle interactions with the Google Gemini API"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-ultra"):
        super().__init__(api_key, model)
        self.provider_name = "Gemini"
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the Gemini API"""
        try:
            from google.genai import types
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(f"Sending request to Gemini API with model {self.model}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=[formatted_prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7
                )
            )
            
            if not response.text:
                raise APIError("Empty response from Gemini API", self.provider_name)
                
            return response.text
            
        except Exception as e:
            self._handle_error(e, "request or response processing")

class TogetherAPI(BaseAPI):
    """Class to handle interactions with the Together AI API"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-4-405B-Instruct"):
        super().__init__(api_key, model)
        self.provider_name = "Together"
        try:
            from together import Together
            self.client = Together(api_key=api_key)
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the Together AI API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(f"Sending request to Together AI API with model {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=max_tokens
            )
            
            # Robust response extraction
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            elif hasattr(response, 'text'):
                return response.text
            else:
                # If response doesn't match expected structures
                raise APIError("Unexpected response format from Together AI", self.provider_name)
            
        except Exception as e:
            self._handle_error(e, "request or response processing")


class PackycodeAPI(BaseAPI):
    """OpenAI-compatible Packycode Codex API client."""

    def __init__(self, api_key: str, model: str = "gpt-5-codex-high"):
        super().__init__(api_key, model)
        self.provider_name = "Packycode"
        base_url = os.getenv('PACKYCODE_BASE_URL', 'https://codex-api.packycode.com/v1')
        wire_api = os.getenv('PACKYCODE_WIRE_API', 'responses').strip().lower()
        self.wire_api = wire_api if wire_api in {"responses", "chat"} else "responses"

        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.base_model, self.default_effort = self._normalize_model(model)

        default_headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': os.getenv('PACKYCODE_USER_AGENT', 'ReasonGraph/1.0 (+https://packycode.com)')
        }

        extra_headers = os.getenv('PACKYCODE_HEADERS')
        if extra_headers:
            try:
                default_headers.update(json.loads(extra_headers))
            except json.JSONDecodeError:
                logger.warning('PACKYCODE_HEADERS 环境变量解析失败，忽略额外头信息')

        self.session.headers.update(default_headers)

        cookies = os.getenv('PACKYCODE_COOKIES')
        if cookies:
            try:
                cookie_dict = json.loads(cookies)
                self.session.cookies.update(cookie_dict)
            except json.JSONDecodeError:
                logger.warning('PACKYCODE_COOKIES 环境变量解析失败，忽略自定义 Cookie')

        proxy = os.getenv('PACKYCODE_PROXY')
        if proxy:
            self.session.proxies.update({'http': proxy, 'https': proxy})

    MODEL_ALIASES = {
        'gpt-5-codex': ('gpt-5-codex', None),
        'gpt-5-codex-low': ('gpt-5-codex', 'low'),
        'gpt-5-codex-medium': ('gpt-5-codex', 'medium'),
        'gpt-5-codex-high': ('gpt-5-codex', 'high')
    }

    def _normalize_model(self, model_name: str) -> tuple[str, Optional[str]]:
        normalized = self.MODEL_ALIASES.get(model_name)
        if normalized:
            return normalized
        return model_name, None

    def generate_response(self, prompt: str, max_tokens: int = 2048,
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using Packycode Codex endpoint."""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            logger.info(f"Sending request to Packycode API with model {self.model}")

            effort = self.default_effort or os.getenv('PACKYCODE_REASONING_EFFORT')

            if self.wire_api == "responses":
                url = f"{self.base_url}/responses"
                payload = {
                    "model": self.base_model,
                    "input": formatted_prompt,
                    "stream": True,
                    "max_output_tokens": max_tokens,
                    "response_format": {"type": "text"}
                }
                if effort:
                    payload['reasoning'] = {"effort": effort}
                response = self.session.post(url, json=payload, stream=True, timeout=120)
                return self._consume_responses_stream(response)

            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.base_model,
                "messages": [
                    {"role": "system", "content": "You are Codex, an advanced reasoning assistant."},
                    {"role": "user", "content": formatted_prompt}
                ],
                "max_tokens": max_tokens,
                "stream": True
            }
            if effort:
                payload['reasoning'] = {"effort": effort}
            response = self.session.post(url, json=payload, stream=True, timeout=120)
            return self._consume_chat_stream(response)

        except Exception as e:
            self._handle_error(e, "request or response processing")

    @staticmethod
    def _ensure_response_ok(response: requests.Response, provider_name: str) -> None:
        if response.status_code == 403:
            raise APIError(
                "Packycode API 被 Cloudflare 拒绝访问，请配置 PACKYCODE_HEADERS/COOKIES 或使用代理以通过安全校验。",
                provider_name,
                status_code=403
            )
        if response.status_code >= 400:
            try:
                error_json = response.json()
            except ValueError:
                error_json = response.text
            raise APIError(str(error_json), provider_name, response.status_code)

    def _consume_responses_stream(self, response: requests.Response) -> str:
        self._ensure_response_ok(response, self.provider_name)

        chunks: List[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith(':'):
                continue
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if not payload or payload == '[DONE]':
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            event_type = event.get('type', '')
            if event_type == 'response.error':
                message = event.get('error', {}).get('message') or event.get('message') or 'Packycode stream error'
                raise APIError(message, self.provider_name)

            if event_type.startswith('response.output_text'):
                delta = event.get('delta') or event.get('text')
                if isinstance(delta, dict):
                    delta = delta.get('text') or delta.get('content', '')
                if delta:
                    chunks.append(delta)

        text = ''.join(chunks).strip()
        if not text:
            raise APIError('Packycode 响应内容为空', self.provider_name)
        return text

    def _consume_chat_stream(self, response: requests.Response) -> str:
        self._ensure_response_ok(response, self.provider_name)

        chunks: List[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line or line.startswith(':'):
                continue
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = event.get('choices', [])
            if not choices:
                continue
            delta = choices[0].get('delta', {})
            if 'content' in delta and delta['content']:
                chunks.append(delta['content'])
            if delta.get('role') == 'assistant' and not delta.get('content'):
                continue
            if event.get('error'):
                message = event['error'].get('message', 'Packycode stream error')
                raise APIError(message, self.provider_name)

        text = ''.join(chunks).strip()
        if not text:
            raise APIError('Packycode 响应内容为空', self.provider_name)
        return text


class DeepSeekAPI(BaseAPI):
    """Class to handle interactions with the DeepSeek API"""
    
    def __init__(self, api_key: str, model: str = "deepseek-reasoner"):
        super().__init__(api_key, model)
        self.provider_name = "DeepSeek"
        try:
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the DeepSeek API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(f"Sending request to DeepSeek API with model {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=max_tokens
            )
            
            # Check if this is the reasoning model response
            if self.model == "deepseek-reasoner" and hasattr(response.choices[0].message, "reasoning_content"):
                # Include both reasoning and answer
                reasoning = response.choices[0].message.reasoning_content
                answer = response.choices[0].message.content
                return f"Reasoning:\n{reasoning}\n\nAnswer:\n{answer}"
            else:
                # Regular model response
                return response.choices[0].message.content
            
        except Exception as e:
            self._handle_error(e, "request or response processing")

class QwenAPI(BaseAPI):
    """Class to handle interactions with the Qwen API"""
    
    def __init__(self, api_key: str, model: str = "qwen2.5-72b-instruct"):
        super().__init__(api_key, model)
        self.provider_name = "Qwen"
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the Qwen API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(f"Sending request to Qwen API with model {self.model}")
            
            # Check if this is the reasoning model (qwq-plus)
            if self.model == "qwq-plus":
                # For qwq-plus model, we need to use streaming
                reasoning_content = ""
                answer_content = ""
                is_answering = False
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ],
                    max_tokens=max_tokens,
                    stream=True  # qwq-plus only supports streaming output
                )
                
                for chunk in response:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    # Collect reasoning process
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        reasoning_content += delta.reasoning_content
                    # Collect answer content
                    elif hasattr(delta, 'content') and delta.content is not None:
                        answer_content += delta.content
                        is_answering = True
                
                # Return combined reasoning and answer
                return f"Reasoning:\n{reasoning_content}\n\nAnswer:\n{answer_content}"
            else:
                # Regular model response (non-streaming)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ],
                    max_tokens=max_tokens
                )
                
                return response.choices[0].message.content
            
        except Exception as e:
            self._handle_error(e, "request or response processing")

class GrokAPI(BaseAPI):
    """Class to handle interactions with the Grok API"""
    
    def __init__(self, api_key: str, model: str = "grok-2-latest"):
        super().__init__(api_key, model)
        self.provider_name = "Grok"
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the Grok API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(f"Sending request to Grok API with model {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self._handle_error(e, "request or response processing")

class APIFactory:
    """Factory class for creating API instances"""
    
    _providers = {
        "anthropic": {
            "class": AnthropicAPI,
            "default_model": "claude-3-7-sonnet-20250219"
        },
        "openai": {
            "class": OpenAIAPI,
            "default_model": "gpt-5-medium"
        },
        "google": {
            "class": GeminiAPI,
            "default_model": "gemini-2.0-ultra"
        },
        "together": {
            "class": TogetherAPI,
            "default_model": "meta-llama/Llama-4-405B-Instruct"
        },
        "packycode": {
            "class": PackycodeAPI,
            "default_model": "gpt-5-codex-high"
        },
        "deepseek": {
            "class": DeepSeekAPI,
            "default_model": "deepseek-reasoner"
        },
        "qwen": {
            "class": QwenAPI,
            "default_model": "qwen2.5-72b-instruct"
        },
        "grok": {
            "class": GrokAPI,
            "default_model": "grok-2-latest"
        }
    }
    
    @classmethod
    def supported_providers(cls) -> List[str]:
        """Get list of supported providers"""
        return list(cls._providers.keys())
    
    @classmethod
    def create_api(cls, provider: str, api_key: str, model: Optional[str] = None) -> BaseAPI:
        """Factory method to create appropriate API instance"""
        provider = provider.lower()
        if provider not in cls._providers:
            raise ValueError(f"Unsupported provider: {provider}. "
                           f"Supported providers are: {', '.join(cls.supported_providers())}")
        
        provider_info = cls._providers[provider]
        api_class = provider_info["class"]
        model = model or provider_info["default_model"]
        
        logger.info(f"Creating API instance for provider: {provider}, model: {model}")
        return api_class(api_key=api_key, model=model)

def create_api(provider: str, api_key: str, model: Optional[str] = None) -> BaseAPI:
    """Convenience function to create API instance"""
    return APIFactory.create_api(provider, api_key, model)

# Example usage:
if __name__ == "__main__":
    # Example with Anthropic
    anthropic_api = create_api("anthropic", "your-api-key")
    
    # Example with OpenAI
    openai_api = create_api("openai", "your-api-key", "gpt-4")
    
    # Example with Gemini
    gemini_api = create_api("gemini", "your-api-key", "gemini-2.0-flash")
    
    # Example with Together AI
    together_api = create_api("together", "your-api-key")
    
    # Get supported providers
    providers = APIFactory.supported_providers()
    print(f"Supported providers: {providers}")
