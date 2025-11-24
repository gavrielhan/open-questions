#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ================================================================
# CONFIGURATION
# ================================================================

@dataclass
class OpenAIConfig:
    api_key: str
    api_base_url: str
    model: str
    max_tokens: int = 400
    temperature: float = 0.0
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    batch_size: int = 3           # smaller = fewer 429s and shorter responses (reduced from 5 for 9 topics)
    parallel_workers: int = 2     # lower concurrency for LiteLLM
    request_delay_seconds: float = 0.5  # delay between API requests to avoid rate limits
    # DeepSeek config for YAML repair (Azure endpoint)
    repair_api_key: str = ""
    repair_model: str = "DeepSeek-V3.1-gavriel"
    repair_endpoint: str = "https://sni-ai-foundry.services.ai.azure.com/openai/v1/"
    repair_api_version: str = "2025-04-01-preview"
    # Azure AI Foundry config
    api_version: str = "2024-12-01-preview"  # Azure API version
    
    def is_azure(self) -> bool:
        """Check if this is an Azure AI Foundry endpoint."""
        return "azure" in self.api_base_url.lower() or "cognitiveservices" in self.api_base_url.lower()
    
    def get_chat_completions_url(self) -> str:
        """Get the correct chat completions endpoint URL based on provider."""
        base = self.api_base_url.rstrip("/")
        if self.is_azure():
            # Azure OpenAI format - try with /openai prefix
            # Some Azure endpoints need /openai/deployments, others just /deployments
            return f"{base}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"
        else:
            # LiteLLM or standard OpenAI format
            return f"{base}/v1/chat/completions"

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY or API_KEY in environment.")

        api_base_url = (
            os.getenv("OPENAI_API_BASE_URL")
            or os.getenv("API_BASE_URL")
            or "https://api.openai.com"
        )
        model = os.getenv("MODEL") or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

        # DeepSeek config for YAML repair (can use same API key or different one)
        repair_key = os.getenv("REPAIR_API_KEY") or api_key  # Use main API key if not specified
        repair_model = os.getenv("REPAIR_MODEL", "DeepSeek-V3.1-gavriel")
        repair_endpoint = os.getenv("REPAIR_ENDPOINT", "https://sni-ai-foundry.services.ai.azure.com/openai/v1/")
        repair_api_version = os.getenv("REPAIR_API_VERSION", "2025-04-01-preview")
        api_version = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

        return cls(
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "400")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "5")),
            retry_backoff_seconds=float(os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", "2")),
            batch_size=int(os.getenv("OPENAI_BATCH_SIZE", "5")),
            parallel_workers=int(os.getenv("OPENAI_PARALLEL_WORKERS", "2")),
            request_delay_seconds=float(os.getenv("OPENAI_REQUEST_DELAY_SECONDS", "0.5")),
            repair_api_key=repair_key,
            repair_model=repair_model,
            repair_endpoint=repair_endpoint,
            repair_api_version=repair_api_version,
            api_version=api_version,
        )


# ================================================================
# LANGCHAIN CHAIN SETUP
# ================================================================

def create_classification_chain(config: OpenAIConfig, topic_columns: Sequence[str], question: str):
    """
    Create a unified LangChain chain that does classification and validation in sequence.
    This chain processes batches of texts.
    """
    from langchain_core.runnables import RunnableLambda
    
    # Create LLM with Azure AI Foundry or LiteLLM base URL
    # Increase max_tokens significantly for batch processing with multiple topics
    classification_max_tokens = max(config.max_tokens, 3000)  # Increased from 1000 to handle 9 topics in batch
    validation_max_tokens = max(config.max_tokens, 800)
    
    # Configure LangChain ChatOpenAI for Azure AI Foundry or LiteLLM
    # Azure AI Foundry is OpenAI-compatible, but uses a different endpoint structure
    # Try using base_url - if Azure AI Foundry supports /v1/chat/completions directly, this will work
    # Otherwise, we'll need to handle it in the direct API calls
    if config.is_azure():
        # For Azure OpenAI, we need to add /openai to the base URL
        # LangChain will append /v1/chat/completions, but Azure needs /openai/deployments/...
        # So we construct a special base_url that when combined with LangChain's path gives us the right endpoint
        # We want: {base}/openai/deployments/{model}/chat/completions
        # LangChain adds: /v1/chat/completions
        # So we can't use LangChain's standard path - we need to use Azure's native client or custom base_url
        # For now, use the base URL with /openai prefix
        azure_base = config.api_base_url.rstrip("/") + "/openai"
        llm_kwargs = {
            "model": config.model,  # Deployment name in Azure
            "base_url": azure_base,  # Add /openai prefix for Azure
            "api_key": config.api_key,
            # Skip temperature for Azure - some models only support default (1)
            "max_completion_tokens": classification_max_tokens,  # Azure uses max_completion_tokens
            # Skip response_format for Azure - it may cause 400 errors
        }
    else:
        # LiteLLM or standard OpenAI: use base_url and model
        llm_kwargs = {
            "model": config.model,
            "base_url": config.api_base_url,
            "api_key": config.api_key,
            "temperature": config.temperature,
            "max_tokens": classification_max_tokens,
            "model_kwargs": {"response_format": {"type": "json_object"}},
        }
    
    # Main classification LLM - with logprobs for confidence extraction
    # We need to use the underlying OpenAI client to get logprobs properly
    # LangChain's ChatOpenAI doesn't directly support logprobs, so we'll extract them from raw API calls
    llm_classify = ChatOpenAI(**llm_kwargs)
    
    # Validation and re-evaluation LLMs with same config but different max_tokens
    llm_kwargs_validate = llm_kwargs.copy()
    if config.is_azure():
        llm_kwargs_validate["max_completion_tokens"] = validation_max_tokens
    else:
        llm_kwargs_validate["max_tokens"] = validation_max_tokens
    
    llm_validate = ChatOpenAI(**llm_kwargs_validate)
    
    # Re-evaluation LLM for low confidence classifications - use DeepSeek
    # NOTE: Currently disabled due to LangChain compatibility issues with Azure AI Foundry endpoints
    # The DeepSeek endpoint format is not fully compatible with LangChain's ChatOpenAI wrapper
    llm_reevaluate_kwargs = {
        "model": config.repair_model,  # DeepSeek-V3.1-gavriel
        "base_url": config.repair_endpoint,
        "api_key": config.repair_api_key,
        "temperature": 0.7,  # Slightly higher for re-evaluation
        "max_tokens": validation_max_tokens,
    }
    llm_reevaluate = ChatOpenAI(**llm_reevaluate_kwargs)

    topics_text = ", ".join(topic_columns)
    sample_pairs = ", ".join(f'"{topic_columns[i]}": 0' for i in range(min(3, len(topic_columns))))
    if len(topic_columns) > 3:
        sample_pairs += ", ..."
    
    # Classification prompt
    classify_system = (
        "You are an expert annotator for academic media research analyzing Hebrew text. "
        "For each numbered main text below (written in Hebrew), decide whether every topic is explicitly mentioned or clearly discussed.\n\n"
        "Guidelines:\n"
        "- The main texts are in Hebrew. The topic names are also in Hebrew.\n"
        "- If a text is empty or very short and with no real information, return 0 for all topics.\n"
        "- Respond with JSON only; no markdown fences or commentary.\n"
        "- Use integers 0 or 1 only. 1 means the topic is clearly mentioned, 0 otherwise.\n"
        "- If uncertain, choose 0.\n"
        "- Include every topic exactly once per row and use the topic names exactly as provided.\n"
        "- Ensure all JSON strings are properly escaped and the output is valid JSON."
    )
    
    example_line = f'  "1": {{{sample_pairs}}}'
    example_line_escaped = example_line.replace("{", "{{").replace("}", "}}")
    
    classify_human = (
        "Return strictly valid JSON with the structure:\n"
        "{{\n"
        + example_line_escaped + ",\n"
        + "  \"2\": {{...}},\n"
        + "  ...\n"
        + "}}\n\n"
        "Topics: {topics}\n\n"
        "Main texts (Hebrew):\n"
        "{texts}"
    )
    
    classify_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(classify_system),
        HumanMessagePromptTemplate.from_template(classify_human),
    ])
    
    # Validation prompt
    validate_system = (
        "You are a quality assurance validator for academic media research classifications. "
        "Review the given question, text, and initial topic classifications. "
        "Verify that each topic classification (0 or 1) is correct based on the text content.\n\n"
        "Guidelines:\n"
        "- The question and text are in Hebrew. The topic names are also in Hebrew.\n"
        "- If a text is empty or very short and with no real information, return 0 for all topics.\n"
        "- Respond with JSON only; no markdown fences or commentary.\n"
        "- Return the corrected classifications even if no changes are needed.\n"
        "- Use integers 0 or 1 only. 1 means the topic is clearly mentioned, 0 otherwise.\n"
        "- Include every topic exactly once with the topic names exactly as provided.\n"
        "- Ensure all JSON strings are properly escaped and the output is valid JSON."
    )
    
    example_line_validate = f'{{{sample_pairs}}}'
    example_line_validate_escaped = example_line_validate.replace("{", "{{").replace("}", "}}")
    
    validate_human = (
        "Review and correct the topic classifications for the following entry.\n\n"
        "Question: {question}\n"
        "Text: {text}\n\n"
        "Initial Classifications:\n"
        "{classifications}\n\n"
        "Return the corrected classifications as JSON with this structure:\n"
        "{{\n"
        + example_line_validate_escaped + "\n"
        + "}}\n\n"
        "Topics to include: {topics}\n\n"
        "If the classifications are already correct, return them unchanged. "
        "If corrections are needed, return the corrected values."
    )
    
    validate_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(validate_system),
        HumanMessagePromptTemplate.from_template(validate_human),
    ])
    
    # Re-evaluation prompt for low confidence classifications
    reevaluate_system = (
        "You are a senior expert annotator reviewing classifications that had low confidence scores. "
        "Re-analyze the given question, text, and topic to make a more confident decision.\n\n"
        "Guidelines:\n"
        "- The question and text are in Hebrew. The topic name is also in Hebrew.\n"
        "- Take extra care to verify if the topic is truly mentioned or discussed.\n"
        "- If a text is empty or very short and with no real information, return 0 for all topics.\n"
        "- Respond with JSON only; no markdown fences or commentary.\n"
        "- Use integers 0 or 1 only. 1 means the topic is clearly mentioned, 0 otherwise.\n"
        "- Be more decisive than the initial classification.\n"
        "- Ensure all JSON strings are properly escaped and the output is valid JSON."
    )
    
    reevaluate_human = (
        "Re-evaluate this low-confidence classification with extra scrutiny.\n\n"
        "Question: {question}\n"
        "Text: {text}\n\n"
        "Topic: {topic}\n"
        "Initial Classification: {initial_value}\n"
        "Confidence: {confidence:.2f}\n\n"
        "Return the corrected classification as JSON:\n"
        "{{\n"
        "  \"{topic}\": 0 or 1\n"
        "}}\n\n"
        "Make a confident decision based on whether the topic is clearly present in the text."
    )
    
    reevaluate_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(reevaluate_system),
        HumanMessagePromptTemplate.from_template(reevaluate_human),
    ])
    
    parser = JsonOutputParser()
    
    # Unified chain: classify -> validate (will be used per text in classify_batch)
    classify_chain = classify_prompt | llm_classify
    validate_chain = validate_prompt | llm_validate | parser
    reevaluate_chain = reevaluate_prompt | llm_reevaluate | parser
    
    return classify_chain, validate_chain, reevaluate_chain, topics_text


# ================================================================
# API CALL WITH BACKOFF & REPAIR
# ================================================================

def _classify_with_logprobs(inputs: dict, topics: Sequence[str], config: OpenAIConfig) -> Tuple[dict, float]:
    """
    Make a direct API call to get classification with logprobs for confidence calculation.
    This bypasses LangChain to access logprobs directly.
    """
    import requests
    
    # Build the prompt
    topics_text = inputs["topics"]
    texts = inputs["texts"]
    
    sample_topics_block = "\n".join(f"    {topics[i]}: 0" for i in range(min(3, len(topics))))
    if len(topics) > 3:
        sample_topics_block += "\n    ..."
    
    system_prompt = (
        "You are an expert annotator for academic media research analyzing Hebrew text. "
        "For each numbered main text below (written in Hebrew), decide whether every topic is explicitly mentioned or clearly discussed.\n\n"
        "Guidelines:\n"
        "- The main texts are in Hebrew. The topic names are also in Hebrew.\n"
        "- If a text is empty or very short and with no real information, return 0 for all topics.\n"
        "- Respond with YAML only; no markdown fences or commentary.\n"
        "- Use integers 0 or 1 only. 1 means the topic is clearly mentioned, 0 otherwise.\n"
        "- If uncertain, choose 0.\n"
        "- Include every topic exactly once per row and use the topic names exactly as provided.\n"
        "- Ensure the YAML is strictly valid and properly indented."
    )
    
    user_prompt = (
        f"Return strictly valid YAML with the structure:\n"
        f"1:\n"
        f"{sample_topics_block}\n"
        f"2:\n"
        f"    ...\n\n"
        f"Topics: {topics_text}\n\n"
        f"Main texts (Hebrew):\n"
        f"{texts}"
    )
    
    # Use the correct endpoint URL based on provider (Azure or LiteLLM)
    url = config.get_chat_completions_url()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }
    
    classification_max_tokens = max(config.max_tokens, 3000)
    
    # Build payload - Azure doesn't need model in payload if it's in the URL
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    }
    
    # Azure newer models use max_completion_tokens instead of max_tokens
    # Also, some Azure models don't support temperature=0, so we skip it
    if config.is_azure():
        payload["max_completion_tokens"] = classification_max_tokens
        # Skip temperature for Azure - some models only support default (1)
    else:
        payload["max_tokens"] = classification_max_tokens
        payload["temperature"] = config.temperature
    
    # For Azure, model is in the URL path, not in payload
    # For LiteLLM/OpenAI, include model in payload and additional features
    if not config.is_azure():
        payload["model"] = config.model
        payload["response_format"] = {"type": "json_object"}
        payload["logprobs"] = True
        payload["top_logprobs"] = 5
    else:
        # Azure OpenAI may not support response_format or logprobs
        # Only add them if the API version supports them
        # For now, skip these features for Azure to avoid 400 errors
        pass
    
    error_log: List[str] = []

    for attempt in range(1, config.max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            
            # Log the error details if we get a bad response
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    print(f"Azure API Error: {error_data}", file=sys.stderr)
                except:
                    print(f"Azure API Error: {response.text}", file=sys.stderr)
            
            response.raise_for_status()
            data = response.json()
            
            # Extract logprobs and calculate confidence
            choice = data["choices"][0]
            message_content = choice["message"]["content"]
            
            # Extract logprobs if available
            logprobs = choice.get("logprobs")
            confidence = 1.0  # Default
            
            if logprobs and "content" in logprobs:
                import math
                content_logprobs = logprobs["content"]
                probs = []
                for token_data in content_logprobs:
                    if isinstance(token_data, dict) and "logprob" in token_data:
                        prob = math.exp(token_data["logprob"])
                        probs.append(prob)
                if probs:
                    confidence = sum(probs) / len(probs)
            
            # Parse YAML response
            try:
                initial_results = _parse_yaml_content(message_content)
            except ValueError as yaml_error:
                error_msg = str(yaml_error)
                print(f"⚠️  Classification YAML error: {error_msg}", file=sys.stderr)
                error_log.append(f"YAML parse attempt {attempt}: {error_msg}")
                repaired = _repair_yaml_auto(
                    message_content,
                    topics,
                    config,
                    error_message=error_msg,
                    error_log=error_log,
                )
                if repaired is None:
                    raise
                initial_results = repaired
            
            return initial_results, confidence
            
        except Exception as e:
            error_str = str(e)
            error_log.append(f"Attempt {attempt}: {error_str}")
            
            # Provide detailed error information
            if "404" in error_str:
                print(f"❌ 404 Error from classification API (attempt {attempt}/{config.max_retries})", file=sys.stderr)
                print(f"   URL: {url}", file=sys.stderr)
                print(f"   Error: {error_str}", file=sys.stderr)
                if response:
                    try:
                        print(f"   Response body: {response.text[:500]}", file=sys.stderr)
                    except:
                        pass
            elif "401" in error_str or "403" in error_str:
                print(f"🔒 Authentication Error (attempt {attempt}/{config.max_retries}): {error_str}", file=sys.stderr)
                print(f"   Check your API_KEY in .env file", file=sys.stderr)
            elif "429" in error_str or "Rate limit" in error_str:
                print(f"⏱️  Rate limit hit (attempt {attempt}/{config.max_retries})", file=sys.stderr)
            elif "500" in error_str or "502" in error_str or "503" in error_str:
                print(f"🔄 Server Error (attempt {attempt}/{config.max_retries}): {error_str}", file=sys.stderr)
            else:
                print(f"⚠️  Classification API Error (attempt {attempt}/{config.max_retries}): {error_str}", file=sys.stderr)
            
            if attempt == config.max_retries:
                # All retries failed - raise error
                print(f"❌ All {config.max_retries} attempts failed for classification API", file=sys.stderr)
                raise RuntimeError(f"Classification failed after {config.max_retries} attempts: {error_str}")
            
            wait = config.retry_backoff_seconds * attempt
            print(f"   ⏳ Retrying in {wait:.1f}s...", file=sys.stderr)
            time.sleep(wait)
    
    # Should not reach here, but just in case
    return {}, 1.0


def extract_confidence_from_logprobs(response) -> Dict[str, float]:
    """
    Extract confidence scores from logprobs in the LLM response.
    Returns a dict mapping token positions to confidence scores.
    """
    import math
    
    confidences = {}
    
    try:
        # Check if response has logprobs in response_metadata
        if hasattr(response, 'response_metadata'):
            logprobs_data = response.response_metadata.get('logprobs')
            if logprobs_data:
                # Handle different logprobs structures
                if isinstance(logprobs_data, dict):
                    if 'content' in logprobs_data:
                        content_logprobs = logprobs_data['content']
                        # Calculate average confidence from logprobs
                        for idx, token_data in enumerate(content_logprobs):
                            if isinstance(token_data, dict) and 'logprob' in token_data:
                                # Convert log probability to probability (0-1)
                                prob = math.exp(token_data['logprob'])
                                confidences[idx] = prob
                    elif 'token_logprobs' in logprobs_data:
                        # Alternative structure
                        token_logprobs = logprobs_data['token_logprobs']
                        for idx, logprob in enumerate(token_logprobs):
                            if logprob is not None:
                                prob = math.exp(logprob)
                                confidences[idx] = prob
        
        # Also check if logprobs are directly on the response
        if hasattr(response, 'logprobs') and response.logprobs:
            logprobs_data = response.logprobs
            if isinstance(logprobs_data, list):
                for idx, logprob in enumerate(logprobs_data):
                    if logprob is not None:
                        prob = math.exp(logprob) if isinstance(logprob, (int, float)) else 1.0
                        confidences[idx] = prob
        
        # Return average confidence if available
        if confidences:
            avg_confidence = sum(confidences.values()) / len(confidences)
            return {'average': avg_confidence}
        
    except Exception as e:
        # Silently fail - logprobs may not be supported
        pass
    
    # Default to high confidence if we can't extract logprobs (feature may not be supported)
    return {'average': 1.0}


def _strip_code_fences(content: str) -> str:
    """
    Remove Markdown code fences (```yaml, ```json, etc.) from the response string.
    """
    content = content.strip()
    if content.startswith("```"):
        # Remove starting fence and optional language tag
        content = content[3:]
        newline_idx = content.find("\n")
        if newline_idx != -1:
            # Drop language hint (e.g., "yaml" or "json")
            content = content[newline_idx + 1 :]
        # Remove trailing fence
        if content.endswith("```"):
            content = content[:-3]
    return content.strip()


def _parse_yaml_content(content: str) -> Dict[str, Any]:
    """
    Parse YAML (or JSON-as-YAML) content into a Python dict with string keys.
    """
    cleaned = _strip_code_fences(content)
    if not cleaned:
        raise ValueError("Empty YAML content.")

    try:
        data = yaml.safe_load(cleaned)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc

    if not isinstance(data, Mapping):
        raise ValueError("YAML root element must be a mapping/object.")

    # Convert top-level keys to strings (YAML may load numbers as ints)
    normalized = {}
    for key, value in data.items():
        normalized[str(key)] = value
    return normalized


def _cell_has_content(value: Any) -> bool:
    """Return True if a cell contains non-empty content."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return not pd.isna(value)


def _trim_trailing_empty_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Trim only the trailing rows that are completely empty.
    We check ONLY the first column (answer column) - if it's empty, the row is empty,
    regardless of what's in the topic columns (which might have pre-filled 0s).
    """
    original_rows = len(df)
    if original_rows == 0:
        return df, 0

    # Check only the first column for content
    first_col = df.iloc[:, 0]
    non_empty_mask = first_col.apply(_cell_has_content)
    
    if non_empty_mask.any():
        non_empty_positions = non_empty_mask.to_numpy().nonzero()[0]
        last_pos = int(non_empty_positions[-1])
        trimmed_df = df.iloc[: last_pos + 1].copy()
    else:
        trimmed_df = df.iloc[:1].copy()

    rows_removed = original_rows - len(trimmed_df)
    return trimmed_df, rows_removed


def _cell_has_content(value: Any) -> bool:
    """Return True if a cell contains any meaningful content."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return not pd.isna(value)


def _validate_with_direct_api(
    question: str,
    text: str,
    initial_classifications: dict,
    topics: Sequence[str],
    config: OpenAIConfig,
    error_log: Optional[List[str]] = None,
    text_index: Optional[int] = None,
) -> dict:
    """
    Validate classifications using direct API calls.
    Uses GPT-5.1 to double-check the initial classifications.
    Returns: dict of {topic: 0 or 1}
    """
    # Build URL based on provider
    if config.is_azure():
        url = config.get_chat_completions_url()
    else:
        url = f"{config.api_base_url.rstrip('/')}/v1/chat/completions"
    
    classifications_str = ", ".join(f'"{topic}": {value}' for topic, value in initial_classifications.items())
    topics_text = ", ".join(topics)
    
    system_prompt = (
        "You are a quality control expert reviewing topic classifications for Hebrew text analysis. "
        "Your job is to verify and correct the initial classifications.\n\n"
        "Guidelines:\n"
        "- The question and text are in Hebrew. The topic names are also in Hebrew.\n"
        "- Return 1 only if the topic is clearly discussed in the text.\n"
        "- Return 0 if uncertain or if the topic is not clearly present.\n"
        "- Respond with valid YAML only (no prose, no JSON, no code fences)."
    )
    
    user_prompt = (
        f"Review and correct these classifications if needed.\n\n"
        f"Question: {question}\n"
        f"Text: {text}\n\n"
        f"Initial Classifications: {{{classifications_str}}}\n\n"
        f"Topics: {topics_text}\n\n"
        f"Return corrected classifications as YAML with all topics:\n"
        f"{topics[0]}: 0 or 1\n"
        f"..."
    )
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    }
    
    # Set max_tokens based on provider
    if config.is_azure():
        payload["max_completion_tokens"] = 2000
    else:
        payload["max_tokens"] = 2000
        payload["model"] = config.model
        payload["temperature"] = 0
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Extract and parse YAML
        try:
            result = _parse_yaml_content(content)
        except ValueError as yaml_error:
            # Try repair with detailed context
            error_msg = str(yaml_error)
            print(f"⚠️  Validation YAML error: {error_msg}", file=sys.stderr)
            if error_log is not None:
                entry_prefix = f"Text #{text_index + 1}: " if text_index is not None else ""
                error_log.append(f"{entry_prefix}{error_msg}")
            repaired = _repair_yaml_auto(
                content,
                topics,
                config,
                error_message=error_msg,
                error_log=error_log,
            )
            if repaired:
                result = repaired
                print(f"✅ YAML repaired successfully", file=sys.stderr)
            else:
                print(f"❌ Repair failed, using initial classifications", file=sys.stderr)
                return initial_classifications
        
        # Ensure all topics are present
        validated = {}
        for topic in topics:
            value = result.get(topic, initial_classifications.get(topic, 0))
            validated[topic] = int(value in (1, "1", True))
        
        return validated
        
    except Exception as exc:
        print(f"⚠️  Validation failed: {exc}, using initial classifications", file=sys.stderr)
        return initial_classifications


def _reevaluate_with_deepseek(
    question: str,
    text: str,
    topic: str,
    initial_value: int,
    confidence: float,
    config: OpenAIConfig,
    error_log: Optional[List[str]] = None,
) -> Tuple[int, bool]:
    """
    Re-evaluate a single topic classification using DeepSeek when confidence is low.
    Uses direct API calls instead of LangChain.
    
    Returns: (value, success_flag)
        - value: 0 or 1
        - success_flag: True if DeepSeek succeeded, False if we fell back to initial value
    """
    if not config.repair_api_key:
        print(f"No repair API key, keeping initial value for {topic}", file=sys.stderr)
        return initial_value, False
    
    if error_log is None:
        error_log = []
    
    url = f"{config.repair_endpoint.rstrip('/')}/chat/completions"
    print(f"  🔄 Re-evaluating '{topic}' with DeepSeek at: {url}", file=sys.stderr)
    
    system_prompt = (
        "You are a senior expert annotator reviewing classifications that had low confidence scores. "
        "Re-analyze the given question, text, and topic to make a more confident decision.\n\n"
        "Guidelines:\n"
        "- The question and text are in Hebrew. The topic name is also in Hebrew.\n"
        "- Return 1 if the topic is clearly discussed or mentioned in the text.\n"
        "- Return 0 if the topic is not mentioned or only tangentially related.\n"
        "- If uncertain, choose 0.\n"
        "- Respond with valid YAML only (no prose, no JSON)."
    )
    
    user_prompt = (
        f"Re-evaluate this low-confidence classification with extra scrutiny.\n\n"
        f"Question: {question}\n"
        f"Text: {text}\n\n"
        f"Topic: {topic}\n"
        f"Initial Classification: {initial_value}\n"
        f"Confidence: {confidence:.2f}\n\n"
        f"Return the corrected classification as YAML:\n"
        f"{topic}: 0 or 1"
    )
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.repair_api_key}"
    }
    
    payload = {
        "model": config.repair_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Extract and parse YAML
        try:
            result = _parse_yaml_content(content)
        except ValueError as yaml_error:
            # YAML parsing failed - try repair
            error_msg = str(yaml_error)
            print(f"⚠️  Re-evaluation YAML error for '{topic}': {error_msg}", file=sys.stderr)
            error_log.append(f"Re-evaluation YAML error for '{topic}': {error_msg}")
            
            # Try the full repair pipeline (GPT-5.1 self-repair + 5 DeepSeek attempts)
            repaired = _repair_yaml_auto(
                content,
                [topic],  # Only this topic
                config,
                error_message=error_msg,
                error_log=error_log,
            )
            
            if repaired is None:
                # All repair attempts failed - use initial low-confidence value
                print(f"⚠️  All repair attempts failed for '{topic}', using initial low-confidence value: {initial_value}", file=sys.stderr)
                return initial_value, False
            
            result = repaired
        
        # Get the value for this topic
        value = result.get(topic, initial_value)
        return int(value in (1, "1", True)), True
        
    except Exception as exc:
        error_log.append(f"DeepSeek re-evaluation failed for '{topic}': {exc}")
        print(f"❌ DeepSeek re-evaluation failed for {topic}: {exc}, using initial value", file=sys.stderr)
        return initial_value, False


def _repair_yaml_with_deepseek(
    broken_yaml: str,
    topic_columns: Sequence[str],
    config: OpenAIConfig,
    error_message: str = None,
    error_log: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """
    Use DeepSeek (Azure) to repair malformed YAML produced by the model.
    Includes the specific parsing error and prior attempts to give the model context.
    """
    if not config.repair_api_key:
        print("No repair API key configured; skipping repair.", file=sys.stderr)
        return None

    # DeepSeek endpoint
    # Format: https://sni-ai-foundry.services.ai.azure.com/openai/v1/chat/completions
    url = f"{config.repair_endpoint.rstrip('/')}/chat/completions"
    
    # Build error context if provided
    error_context = ""
    if error_message:
        error_context += f"\n⚠️  PARSING ERROR: {error_message}\n"
    if error_log:
        error_context += "\n🧾 ERROR HISTORY:\n" + "\n".join(f"- {entry}" for entry in error_log) + "\n"
    
    prompt = (
        "You are a YAML repair expert. The following YAML should map row numbers to topic classifications "
        "(in Hebrew), but it contains formatting errors.\n\n"
        "🎯 YOUR TASK: Fix the YAML so it becomes 100% valid and strictly follows the required schema.\n\n"
        f"{error_context}"
        "✅ REQUIRED YAML FORMAT:\n"
        "1:\n"
        "  <topic name in Hebrew>: 0 or 1\n"
        "  <topic name in Hebrew>: 0 or 1\n"
        "2:\n"
        "  ...\n\n"
        "🔍 CRITICAL RULES:\n"
        "1. Use YAML mappings only (no lists, no prose, no JSON).\n"
        "2. Topic names must appear exactly as provided; do not translate or rename them.\n"
        "3. Values must be integers 0 or 1 (not booleans or strings).\n"
        "4. Keep the row numbering identical (\"1\", \"2\", ...).\n"
        "5. Do not wrap the output in markdown fences or add commentary.\n"
        "6. Preserve all rows and topics present in the input.\n\n"
        f"📋 Expected topics: {', '.join(topic_columns)}\n\n"
        "🔧 BROKEN YAML TO FIX:\n"
        f"{broken_yaml}"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.repair_api_key}"
    }

    payload = {
        "model": config.repair_model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_yaml_content(content)
    except Exception as exc:
        print(f"DeepSeek YAML repair failed: {exc}", file=sys.stderr)
        return None


def _repair_yaml_with_gpt(
    broken_yaml: str,
    topic_columns: Sequence[str],
    config: OpenAIConfig,
    error_message: str = None,
    error_log: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """
    Fallback YAML repair using the primary GPT model (same deployment as classification).
    """
    url = config.get_chat_completions_url()

    error_context = ""
    if error_message:
        error_context += f"\n⚠️  PARSING ERROR: {error_message}\n"
    if error_log:
        error_context += "\n🧾 ERROR HISTORY:\n" + "\n".join(f"- {entry}" for entry in error_log) + "\n"

    prompt = (
        "You previously generated YAML for topic classifications, but it failed to parse.\n"
        "Fix ONLY the formatting issues so the YAML becomes valid and uses the exact same topics and structure.\n\n"
        "🎯 REQUIREMENTS:\n"
        "1. Preserve all existing topic names in Hebrew exactly.\n"
        "2. Keep the same row numbering (\"1\", \"2\", ...).\n"
        "3. Values must be integers 0 or 1 (not booleans or strings).\n"
        "4. Output must be pure YAML mappings (no JSON, no prose, no code fences).\n"
        "5. Return ONLY the corrected YAML block with no commentary.\n"
        f"{error_context}\n"
        "Broken YAML:\n"
        f"{broken_yaml}"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    payload = {
        "messages": [
            {"role": "system", "content": "Return valid YAML only."},
            {"role": "user", "content": prompt},
        ],
    }

    classification_max_tokens = max(config.max_tokens, 2000)
    if config.is_azure():
        payload["max_completion_tokens"] = classification_max_tokens
    else:
        payload["max_tokens"] = classification_max_tokens
        payload["model"] = config.model
        payload["temperature"] = 0

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_yaml_content(content)
    except Exception as exc:
        print(f"GPT YAML repair failed: {exc}", file=sys.stderr)
        return None


def _repair_yaml_auto(
    broken_yaml: str,
    topic_columns: Sequence[str],
    config: OpenAIConfig,
    error_message: str = None,
    error_log: Optional[List[str]] = None,
) -> Optional[dict]:
    """
    Two-stage YAML repair process:
    1. GPT-5.1 self-repair (1 attempt) - same model that generated it
    2. DeepSeek repair (5 attempts with accumulated error log) - external repair model
    
    Returns: Repaired dict or None if all attempts fail
    """
    if error_log is None:
        error_log = []
    
    # Stage 1: GPT-5.1 self-repair (1 attempt)
    print("🔧 Stage 1: Attempting GPT-5.1 self-repair...", file=sys.stderr)
    repaired = _repair_yaml_with_gpt(
        broken_yaml, topic_columns, config, error_message=error_message, error_log=error_log
    )
    if repaired is not None:
        print("✅ YAML repaired successfully with GPT-5.1 self-repair", file=sys.stderr)
        return repaired
    
    # Stage 2: DeepSeek repair (5 attempts with error log)
    print("⚠️  GPT-5.1 self-repair failed. Stage 2: Attempting DeepSeek repair (up to 5 attempts)...", file=sys.stderr)
    
    for attempt in range(1, 6):  # 5 attempts
        error_log.append(f"DeepSeek repair attempt {attempt}/5")
        repaired = _repair_yaml_with_deepseek(
            broken_yaml, topic_columns, config, error_message=error_message, error_log=error_log
        )
        if repaired is not None:
            print(f"✅ YAML repaired successfully with DeepSeek (attempt {attempt}/5)", file=sys.stderr)
            return repaired
        else:
            print(f"❌ DeepSeek repair attempt {attempt}/5 failed", file=sys.stderr)
            if attempt < 5:
                time.sleep(1 * attempt)  # Brief delay between attempts
    
    print("❌ All repair attempts failed (1 GPT-5.1 + 5 DeepSeek)", file=sys.stderr)
    return None


def invoke_chain_with_retry(chain, inputs: dict, topic_columns: Sequence[str], config: OpenAIConfig, return_raw: bool = False) -> dict:
    """
    Invoke the LangChain chain with retry logic and YAML repair fallback.
    
    Args:
        return_raw: If True, returns the raw response (for extracting logprobs), otherwise returns parsed YAML
    """
    error_log: List[str] = []

    for attempt in range(1, config.max_retries + 1):
        try:
            result = chain.invoke(inputs)
            
            # If raw response is requested (for logprobs extraction)
            if return_raw:
                return result
            
            # Parse the response content
            if hasattr(result, 'content'):
                content = result.content
                try:
                    return _parse_yaml_content(content)
                except ValueError as parse_error:
                    repaired = _repair_yaml_auto(
                        content,
                        topic_columns,
                        config,
                        error_message=str(parse_error),
                        error_log=error_log,
                    )
                    if repaired is not None:
                        return repaired
                    raise
            
            # Validate result structure
            if isinstance(result, dict):
                return result
            # If result is a string, try to parse it
            if isinstance(result, str):
                try:
                    return _parse_yaml_content(result)
                except ValueError as parse_error:
                    repaired = _repair_yaml_auto(
                        result,
                        topic_columns,
                        config,
                        error_message=str(parse_error),
                        error_log=error_log,
                    )
                    if repaired is not None:
                        return repaired
                    raise
            raise ValueError(f"Unexpected result type: {type(result)}")
        except Exception as e:
            error_str = str(e)
            error_log.append(f"Attempt {attempt}: {error_str}")
            # For YAML errors, try to show what went wrong
            if isinstance(e, ValueError) and "YAML" in str(e):
                error_msg = f"YAML Parse Error: {e}"
                # Try to show the problematic part if available
                if hasattr(e, 'doc') and e.doc:
                    doc_preview = e.doc[:500] if len(e.doc) > 500 else e.doc
                    error_msg += f"\n  Problematic YAML (first 500 chars): {doc_preview}"
                print(f"⚠️  {error_msg}", file=sys.stderr)
                
                # Try DeepSeek repair for YAML errors
                try:
                    if hasattr(e, 'doc') and e.doc:
                        repaired_dict = _repair_yaml_auto(e.doc, topic_columns, config, error_log=error_log)
                        if repaired_dict is not None:
                            print("✅ YAML repaired successfully with DeepSeek", file=sys.stderr)
                            return repaired_dict
                except Exception as repair_error:
                    print(f"  DeepSeek repair also failed: {repair_error}", file=sys.stderr)
            
            elif "YAML" in str(e):
                # Generic YAML error
                print(f"⚠️  YAML-related error: {e}", file=sys.stderr)
                try:
                    # Try to extract content from error message
                    raw_content = str(e)
                    if raw_content:
                        repaired_dict = _repair_yaml_auto(raw_content, topic_columns, config, error_log=error_log)
                        if repaired_dict is not None:
                            print("✅ YAML repaired successfully with DeepSeek", file=sys.stderr)
                            return repaired_dict
                except Exception:
                    pass
            
            if attempt == config.max_retries:
                raise RuntimeError(f"Failed after {attempt} retries: {e}")
            
            wait = config.retry_backoff_seconds * attempt
            if "429" in str(e) or "Rate limit" in str(e):
                print(f"⏱️  Rate limited. Sleeping {wait:.1f}s...", file=sys.stderr)
            elif "500" in str(e) or "Server" in str(e):
                print(f"🔄 Server error. Retry in {wait:.1f}s...", file=sys.stderr)
            elif "404" in str(e):
                print(f"❌ 404 Not Found error: {e}. Retry in {wait:.1f}s...", file=sys.stderr)
            else:
                print(f"⚠️  Error (attempt {attempt}/{config.max_retries}): {e}. Retrying in {wait:.1f}s...", file=sys.stderr)
            time.sleep(wait)


# ================================================================
# CLASSIFICATION
# ================================================================

def classify_batch(texts: Sequence[str], topics: Sequence[str], question: str, config: OpenAIConfig) -> List[Dict[str, int]]:
    """
    Classify a batch of texts using the unified LangChain chain (classification + validation + re-evaluation).
    Skips texts that are empty or have less than 2 characters (returns all 0s).
    Re-evaluates classifications with confidence < 0.6 using DeepSeek model.
    """
    # Filter out short texts - return all 0s for them
    results: List[Optional[Dict[str, int]]] = []
    texts_to_process: List[Tuple[int, str]] = []
    
    for idx, text in enumerate(texts):
        if not text or len(text) < 2:
            # Skip short texts - set all topics to 0
            results.append({t: 0 for t in topics})
        else:
            texts_to_process.append((idx, text))
            results.append(None)  # Placeholder, will be filled later
    
    # If all texts were too short, return early
    if not texts_to_process:
        return results
    
    # No longer using LangChain - all API calls are direct
    topics_text = ", ".join(topics)
    
    # Step 1: Batch classification with logprobs (only for texts that passed the filter)
    process_texts = [text for _, text in texts_to_process]
    enumerated_texts = "\n\n".join(f"{i+1}. {question}: {text}" for i, text in enumerate(process_texts))
    classify_inputs = {
        "topics": topics_text,
        "texts": enumerated_texts,
        "question": question,
    }
    
    # Use direct API call to get logprobs for confidence calculation
    # This bypasses LangChain to access logprobs directly from the API
    initial_results, overall_confidence = _classify_with_logprobs(classify_inputs, topics, config)
    
    # Rate limiting: add delay after classification before starting validation
    if config.request_delay_seconds > 0:
        time.sleep(config.request_delay_seconds)
    
    # Parse initial results and track confidence per topic
    initial_classifications: List[Dict[str, int]] = []
    for i in range(1, len(process_texts) + 1):
        item = initial_results.get(str(i), {}) if isinstance(initial_results, dict) else {}
        initial_classifications.append({t: int(item.get(t, 0) in (1, "1", True)) for t in topics})
    
    # Step 2: Validate each classification using direct API calls
    validated_classifications: List[Dict[str, int]] = []
    validation_error_log: List[str] = []
    
    print("  ℹ️  Validating classifications with direct API calls", file=sys.stderr)
    
    for idx, (text, initial_class) in enumerate(zip(process_texts, initial_classifications)):
        # Skip validation for short texts
        if not text or len(text) < 2:
            validated_classifications.append({t: 0 for t in topics})
            continue
        
        try:
            # Use direct API call for validation
            validated_class = _validate_with_direct_api(
                question=question,
                text=text,
                initial_classifications=initial_class,
                topics=topics,
                config=config,
                error_log=validation_error_log,
                text_index=idx,
            )
            
            # Step 3: Re-evaluate low confidence classifications (confidence < 0.6)
            # Uses DeepSeek model for re-evaluation via direct API calls
            if overall_confidence < 0.6:
                print(f"  ⚠️  Low confidence ({overall_confidence:.2f}) detected, re-evaluating with DeepSeek...", file=sys.stderr)
                
                # Re-evaluate each topic individually for low confidence
                reevaluation_error_log: List[str] = []
                final_class = {}
                any_reevaluation_failed = False
                
                for topic in topics:
                    # Use direct API call to DeepSeek (bypasses LangChain issues)
                    new_value, success = _reevaluate_with_deepseek(
                        question=question,
                        text=text,
                        topic=topic,
                        initial_value=validated_class[topic],
                        confidence=overall_confidence,
                        config=config,
                        error_log=reevaluation_error_log,
                    )
                    final_class[topic] = new_value
                    
                    if not success:
                        any_reevaluation_failed = True
                    
                    # Small delay between topic re-evaluations
                    if config.request_delay_seconds > 0:
                        time.sleep(config.request_delay_seconds * 0.5)
                
                validated_classifications.append(final_class)
                
                # If any re-evaluation failed, mark this text for low-confidence warning
                if any_reevaluation_failed:
                    # Store metadata about this failure (we'll track it globally later)
                    if not hasattr(classify_batch, 'low_confidence_warnings'):
                        classify_batch.low_confidence_warnings = []
                    classify_batch.low_confidence_warnings.append({
                        'text_index': idx,
                        'confidence': overall_confidence,
                        'errors': reevaluation_error_log
                    })
            else:
                # High confidence (>= 0.6), use validated classification as-is
                validated_classifications.append(validated_class)
                
        except Exception as e:
            print(f"Validation failed, using initial classification: {e}", file=sys.stderr)
            validated_classifications.append(initial_class)
        
            # Rate limiting: add delay between validation requests (except for the last one)
            if idx < len(process_texts) - 1 and config.request_delay_seconds > 0:
                time.sleep(config.request_delay_seconds)
    
    # Merge validated results back into original order
    validated_idx = 0
    result_idx = 0
    for idx, text in enumerate(texts):
        if not text or len(text) < 2:
            # Already filled with all 0s, skip
            result_idx += 1
        else:
            # Replace placeholder with validated result
            results[result_idx] = validated_classifications[validated_idx]
            validated_idx += 1
            result_idx += 1
    
    # All results are now filled (no None values)
    return [r for r in results if r is not None]


def update_topics(
    df: pd.DataFrame,
    main_column: str,
    topic_cols: Sequence[str],
    config: OpenAIConfig,
    limit: Optional[int] = None,
    skip_existing: bool = False,
) -> pd.DataFrame:
    indices = list(df.index)
    if limit is not None:
        indices = indices[:limit]

    rows_to_process: List[Tuple[Any, str]] = []
    for idx in indices:
        if skip_existing:
            existing = df.loc[idx, topic_cols]
            if existing.notna().all():
                continue

        text_raw = df.at[idx, main_column]
        # Strip newlines and normalize whitespace to avoid formatting issues
        text = "" if pd.isna(text_raw) else str(text_raw).replace("\n", " ").replace("\r", " ").strip()

        # Skip empty or very short texts (less than 2 characters) - set all topics to 0
        if not text or len(text) < 2:
            for topic in topic_cols:
                df.at[idx, topic] = 0
            continue

        rows_to_process.append((idx, text))

    if not rows_to_process:
        print("No rows require classification (either empty or already filled).")
        return df

    batches = [
        rows_to_process[i : i + config.batch_size]
        for i in range(0, len(rows_to_process), config.batch_size)
    ]

    print(
        f"Processing {len(rows_to_process)} rows in {len(batches)} batches "
        f"(batch_size={config.batch_size}, workers={config.parallel_workers})..."
    )

    failed_batches: List[Tuple[int, List[Any], Exception]] = []
    low_confidence_batches: List[Tuple[int, List[Any], List[dict]]] = []

    with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures: Dict[Any, Tuple[int, List[Tuple[Any, str]]]] = {}

        for batch_idx, batch in enumerate(batches, start=1):
            texts = [text for _, text in batch]
            future = executor.submit(classify_batch, texts, topic_cols, main_column, config)
            futures[future] = (batch_idx, batch)

        for future in as_completed(futures):
            batch_idx, batch = futures[future]
            try:
                results = future.result()
                for (row_idx, text), topics_result in zip(batch, results):
                    for topic, value in topics_result.items():
                        df.at[row_idx, topic] = value
                print(f"✔ Batch {batch_idx}/{len(batches)} processed by model")
                
                # Check if any low-confidence warnings were generated for this batch
                if hasattr(classify_batch, 'low_confidence_warnings') and classify_batch.low_confidence_warnings:
                    row_indices = [row_idx for row_idx, _ in batch]
                    low_confidence_batches.append((batch_idx, row_indices, classify_batch.low_confidence_warnings[:]))
                    # Clear warnings for next batch
                    classify_batch.low_confidence_warnings.clear()
                    
            except Exception as exc:
                row_indices = [row_idx for row_idx, _ in batch]
                print(
                    f"✖ Batch {batch_idx} failed completely (rows {row_indices}): {exc}",
                    file=sys.stderr,
                )
                failed_batches.append((batch_idx, row_indices, exc))

    if failed_batches:
        print("\n" + "="*70, file=sys.stderr)
        print("❌ SUMMARY OF COMPLETELY FAILED BATCHES:", file=sys.stderr)
        print("="*70, file=sys.stderr)
        for batch_idx, row_indices, exc in failed_batches:
            print(
                f"   • Batch {batch_idx}: Rows {row_indices}\n"
                f"     Error: {exc}",
                file=sys.stderr,
            )
        print("="*70 + "\n", file=sys.stderr)
    
    if low_confidence_batches:
        print("\n" + "="*70, file=sys.stderr)
        print("⚠️  SUMMARY OF LOW-CONFIDENCE CLASSIFICATIONS:", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print("The following batches had low confidence (<0.6) and DeepSeek re-evaluation", file=sys.stderr)
        print("encountered YAML repair failures. Initial classifications were used.", file=sys.stderr)
        print("-"*70, file=sys.stderr)
        for batch_idx, row_indices, warnings in low_confidence_batches:
            print(
                f"   • Batch {batch_idx}: Rows {row_indices}\n"
                f"     {len(warnings)} text(s) with low-confidence fallback",
                file=sys.stderr,
            )
        print("="*70 + "\n", file=sys.stderr)

    return df


# ================================================================
# CLI
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Efficient LLM-based binary topic classification.")
    p.add_argument("--input", type=Path, default=Path("open question - steps to pull Med students back to Israel.csv"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OpenAIConfig.from_env()

    # Detect file type and read accordingly
    if args.input.suffix.lower() == '.csv':
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input)
    
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    
    # Remove trailing empty rows (preserve in-file blanks, drop only tail)
    original_rows = len(df)
    df, rows_removed = _trim_trailing_empty_rows(df)
    if rows_removed > 0:
        print(f"📝 Removed {rows_removed} trailing empty rows (from {original_rows} to {len(df)} rows)")
    
    # Determine column indices based on file type
    if args.input.suffix.lower() == '.csv':
        # CSV format: answer at index 0, topics at indices 1-9
        if len(df.columns) < 10:
            raise ValueError(f"Expected ≥10 columns for CSV, got {len(df.columns)}")
        main_column = df.columns[0]
        topics = df.columns[1:10].tolist()  # Indices 1-9 (inclusive)
        print(f"CSV file detected")
        print(f"Main text column (index 0): {main_column}")
        print(f"Topics (indices 1-9): {topics}")
    else:
        # Excel format: answer at index 8, topics from index 9 onwards
        if len(df.columns) <= 10:
            raise ValueError(f"Expected ≥11 columns for Excel, got {len(df.columns)}")
        main_column = df.columns[8]
        topics = df.columns[9:].tolist()
        print(f"Excel file detected")
        print(f"Main text column (index 8): {main_column}")
        print(f"Topics (from index 9): {topics}")

    print(f"Model: {cfg.model}, Base URL: {cfg.api_base_url}")

    df = update_topics(
        df,
        main_column,
        topics,
        cfg,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )

    # Save with appropriate extension
    if args.output:
        out = args.output
    else:
        if args.input.suffix.lower() == '.csv':
            out = args.input.with_name(f"{args.input.stem}_classified.csv")
        else:
            out = args.input.with_name(f"{args.input.stem}_classified.xlsx")
    
    if out.suffix.lower() == '.csv':
        df.to_csv(out, index=False)
    else:
        df.to_excel(out, index=False)
    print(f"✅ Done. Saved to {out}")


if __name__ == "__main__":
    main()
