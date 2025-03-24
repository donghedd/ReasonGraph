"""
Base API module for handling different API providers.
This module provides a unified interface for interacting with various API providers
like Anthropic, OpenAI, Google Gemini and Together AI.
"""

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
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
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
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
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
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
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
    
    def __init__(self, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
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

class DeepSeekAPI(BaseAPI):
    """Class to handle interactions with the DeepSeek API"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
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
    
    def __init__(self, api_key: str, model: str = "qwen-plus"):
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
            "default_model": "gpt-4-turbo-preview"
        },
        "google": {
            "class": GeminiAPI,
            "default_model": "gemini-2.0-flash"
        },
        "together": {
            "class": TogetherAPI,
            "default_model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        },
        "deepseek": {
            "class": DeepSeekAPI,
            "default_model": "deepseek-chat"
        },
        "qwen": {
            "class": QwenAPI,
            "default_model": "qwen-plus"
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