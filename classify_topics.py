#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv


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
    batch_size: int = 5           # number of texts to process per batch
    parallel_workers: int = 2     # lower concurrency for LiteLLM
    request_delay_seconds: float = 0.5  # delay between API requests to avoid rate limits
    # DeepSeek config for YAML repair (Azure endpoint)
    repair_api_key: str = ""
    repair_model: str = "DeepSeek-V3.1-gavriel"
    repair_endpoint: str = "https://sni-ai-foundry.services.ai.azure.com/openai/v1/"
    repair_api_version: str = "2025-04-01-preview"
    # Azure AI Foundry config
    api_version: str = "2025-04-01-preview"  # Azure API version (latest for GPT-5.1)
    
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
        api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")

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
# API CALL WITH BACKOFF & REPAIR
# ================================================================

def _convert_topic_codes_to_names(results: dict, topic_mapping: dict) -> dict:
    """
    Convert topic codes (T1, T2, etc.) back to actual topic names in the results.
    
    Args:
        results: Dict like {"1": {"T1": 0, "T2": 1}, "2": {"T1": 1, "T2": 0}}
        topic_mapping: Dict like {"T1": "actual topic name 1", "T2": "actual topic name 2"}
    
    Returns:
        Dict with actual topic names: {"1": {"actual topic 1": 0, "actual topic 2": 1}, ...}
    """
    converted = {}
    for text_id, topic_values in results.items():
        if not isinstance(topic_values, dict):
            converted[text_id] = topic_values
            continue
        
        converted_topics = {}
        for key, value in topic_values.items():
            # Check if key is a topic code (T1, T2, etc.)
            if key in topic_mapping:
                actual_topic = topic_mapping[key]
                converted_topics[actual_topic] = value
            else:
                # Key might already be the actual topic name
                converted_topics[key] = value
        converted[text_id] = converted_topics
    
    return converted


def _classify_with_logprobs(inputs: dict, topics: Sequence[str], config: OpenAIConfig) -> Tuple[dict, Dict[str, float]]:
    """
    Make a direct API call to get classification with logprobs for confidence calculation.
    Returns: (results_dict, per_text_confidence_dict)
    where per_text_confidence_dict maps text number ("1", "2", ...) to confidence score.
    """
    import requests
    import math
    
    # Build the prompt with numbered topic references to avoid YAML issues with long Hebrew strings
    texts = inputs["texts"]
    question = inputs.get("question", "")
    topic_context = inputs.get("topic_context", "")
    
    # Create numbered topic mapping for compact output
    topic_mapping = {f"T{i+1}": topic for i, topic in enumerate(topics)}
    topics_with_numbers = "\n".join([f"T{i+1}: {topic}" for i, topic in enumerate(topics)])
    topic_keys = ", ".join([f"T{i+1}" for i in range(len(topics))])
    
    system_prompt = (
        "Act like an expert annotator for academic media research specializing in Hebrew-language content analysis.\n\n"
        "Your goal is to classify each Hebrew text against each topic and output a compact YAML structure.\n\n"
        "Task:\n"
        "For every (text, topic) pair, output 1 if the topic is mentioned, discussed, or semantically related to the text content; otherwise output 0.\n\n"
        "IMPORTANT - Interpreting Topics:\n"
        "- Topic labels may be short or concise. Interpret each topic as representing the FULL semantic concept, not just exact keyword matches.\n"
        "- Consider synonyms, related terms, paraphrases, and conceptually equivalent expressions in Hebrew.\n"
        "- Use the detailed topic explanations provided below to understand what each topic truly represents.\n"
        "- Think broadly about what each topic represents as a concept or theme.\n\n"
        "Rules:\n"
        "- If text is empty or lacks substantive information, set all topics to 0.\n"
        "- Set 1 when the topic's concept is present in the text (directly or through related terms).\n"
        "- Consider the semantic meaning, not just literal word matching.\n\n"
        "Output format (YAML only):\n"
        "- Use topic codes (T1, T2, etc.) as keys, NOT the full topic text.\n"
        "- Format: {text_number: {T1: 0, T2: 1, ...}, ...}\n"
        "- Example for 2 texts and 3 topics:\n"
        "  1: {T1: 0, T2: 1, T3: 0}\n"
        "  2: {T1: 1, T2: 0, T3: 1}\n\n"
        "Constraints:\n"
        "- Output ONLY valid YAML, no markdown fences, no explanations.\n"
        "- Use only integers 0 or 1.\n"
        "- Include every topic code for every text."
    )
    
    # Build user prompt with topic context if available
    topic_context_section = ""
    if topic_context:
        topic_context_section = f"\n📚 Detailed Topic Understanding:\n{topic_context}\n"
    
    user_prompt = (
        f"Survey/Question context: {question}\n\n"
        f"Topics (use these codes in output):\n{topics_with_numbers}\n"
        f"{topic_context_section}\n"
        f"Main texts (Hebrew):\n{texts}\n\n"
        f"Output classification for each text using topic codes: {topic_keys}"
    )
    
    # Use the correct endpoint URL based on provider (Azure or LiteLLM)
    url = config.get_chat_completions_url()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }
    
    # With compact topic codes (T1, T2, etc.), we need less tokens
    # But still ensure enough for many texts × many topics
    classification_max_tokens = max(config.max_tokens, 4000)
    
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
            
            # Extract logprobs and calculate per-text confidence
            choice = data["choices"][0]
            message_content = choice["message"]["content"]
            
            # Parse YAML response first
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
            
            # Convert topic codes (T1, T2, etc.) back to actual topic names
            converted_results = _convert_topic_codes_to_names(initial_results, topic_mapping)
            
            # Note: Azure doesn't support logprobs, so we return default confidence
            # Confidence is now determined by model agreement (GPT vs DeepSeek)
            text_keys = list(converted_results.keys())
            per_text_confidence = {key: 1.0 for key in text_keys}
            
            return converted_results, per_text_confidence
            
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
    Handles two formats:
    1. Old format: { "1": {topic: value}, "2": {topic: value} }
    2. New format: [ {id: 1, topic: value}, {id: 2, topic: value} ]
    """
    cleaned = _strip_code_fences(content)
    if not cleaned:
        raise ValueError("Empty YAML content.")

    try:
        data = yaml.safe_load(cleaned)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc

    # Handle new format: list with id keys
    if isinstance(data, list):
        normalized = {}
        for item in data:
            if not isinstance(item, Mapping):
                continue
            # Extract id (could be 'id' key or numeric key)
            text_id = None
            if 'id' in item:
                text_id = str(item['id'])
            elif isinstance(item, dict) and len(item) > 0:
                # Fallback: try to find numeric key or use first key
                for key in item.keys():
                    if key != 'id' and str(key).isdigit():
                        text_id = str(key)
                        break
                if text_id is None:
                    # No id found, skip this item
                    continue
            
            if text_id:
                # Create dict without 'id' key
                text_data = {k: v for k, v in item.items() if k != 'id'}
                normalized[text_id] = text_data
        return normalized
    
    # Handle old format: mapping with numeric/string keys
    if not isinstance(data, Mapping):
        raise ValueError("YAML root element must be a mapping/object or list.")

    # Convert top-level keys to strings (YAML may load numbers as ints)
    normalized = {}
    for key, value in data.items():
        normalized[str(key)] = value
    return normalized


def _classify_with_deepseek(inputs: dict, topics: Sequence[str], config: OpenAIConfig) -> dict:
    """
    Classify texts using DeepSeek model (for parallel comparison with GPT-5.1).
    Returns: results_dict mapping text number to topic classifications.
    """
    # Build the prompt with numbered topic references (same format as GPT-5.1)
    texts = inputs["texts"]
    question = inputs.get("question", "")
    topic_context = inputs.get("topic_context", "")
    
    # Create numbered topic mapping for compact output
    topic_mapping = {f"T{i+1}": topic for i, topic in enumerate(topics)}
    topics_with_numbers = "\n".join([f"T{i+1}: {topic}" for i, topic in enumerate(topics)])
    topic_keys = ", ".join([f"T{i+1}" for i in range(len(topics))])
    
    system_prompt = (
        "Act like an expert annotator for academic media research specializing in Hebrew-language content analysis.\n\n"
        "Your goal is to classify each Hebrew text against each topic and output a compact YAML structure.\n\n"
        "Task:\n"
        "For every (text, topic) pair, output 1 if the topic is mentioned, discussed, or semantically related to the text content; otherwise output 0.\n\n"
        "IMPORTANT - Interpreting Topics:\n"
        "- Topic labels may be short or concise. Interpret each topic as representing the FULL semantic concept, not just exact keyword matches.\n"
        "- Consider synonyms, related terms, paraphrases, and conceptually equivalent expressions in Hebrew.\n"
        "- Use the detailed topic explanations provided below to understand what each topic truly represents.\n"
        "- Think broadly about what each topic represents as a concept or theme.\n\n"
        "Rules:\n"
        "- If text is empty or lacks substantive information, set all topics to 0.\n"
        "- Set 1 when the topic's concept is present in the text (directly or through related terms).\n"
        "- Consider the semantic meaning, not just literal word matching.\n\n"
        "Output format (YAML only):\n"
        "- Use topic codes (T1, T2, etc.) as keys, NOT the full topic text.\n"
        "- Format: {text_number: {T1: 0, T2: 1, ...}, ...}\n"
        "- Example for 2 texts and 3 topics:\n"
        "  1: {T1: 0, T2: 1, T3: 0}\n"
        "  2: {T1: 1, T2: 0, T3: 1}\n\n"
        "Constraints:\n"
        "- Output ONLY valid YAML, no markdown fences, no explanations.\n"
        "- Use only integers 0 or 1.\n"
        "- Include every topic code for every text."
    )
    
    # Build user prompt with topic context if available
    topic_context_section = ""
    if topic_context:
        topic_context_section = f"\n📚 Detailed Topic Understanding:\n{topic_context}\n"
    
    user_prompt = (
        f"Survey/Question context: {question}\n\n"
        f"Topics (use these codes in output):\n{topics_with_numbers}\n"
        f"{topic_context_section}\n"
        f"Main texts (Hebrew):\n{texts}\n\n"
        f"Output classification for each text using topic codes: {topic_keys}"
    )
    
    # Use DeepSeek endpoint
    base = config.repair_endpoint.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.repair_api_key or config.api_key}"
    }
    
    payload = {
        "model": config.repair_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 4000,  # Increased for compact topic codes format
    }
    
    error_log: List[str] = []
    
    for attempt in range(1, config.max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            message_content = data["choices"][0]["message"]["content"]
            
            # Parse YAML response
            try:
                results = _parse_yaml_content(message_content)
                # Convert topic codes (T1, T2, etc.) back to actual topic names
                return _convert_topic_codes_to_names(results, topic_mapping)
            except ValueError as yaml_error:
                error_msg = str(yaml_error)
                print(f"⚠️  DeepSeek YAML error: {error_msg}", file=sys.stderr)
                error_log.append(f"YAML parse attempt {attempt}: {error_msg}")
                
                # Try to repair the YAML
                repaired = _repair_yaml_auto(
                    message_content,
                    topics,
                    config,
                    error_message=error_msg,
                    error_log=error_log,
                )
                if repaired is not None:
                    # Convert repaired results too
                    return _convert_topic_codes_to_names(repaired, topic_mapping)
                raise
                
        except Exception as e:
            error_str = str(e)
            error_log.append(f"DeepSeek attempt {attempt}: {error_str}")
            print(f"⚠️  DeepSeek classification error (attempt {attempt}/{config.max_retries}): {error_str}", file=sys.stderr)
            
            if attempt == config.max_retries:
                raise RuntimeError(f"DeepSeek classification failed after {config.max_retries} attempts: {error_str}")
            
            wait = config.retry_backoff_seconds * attempt
            time.sleep(wait)
    
    return {}


def _resolve_conflict_with_judge(
    text: str,
    question: str,
    topic: str,
    gpt_value: int,
    deepseek_value: int,
    config: OpenAIConfig,
) -> int:
    """
    Use GPT-5.1 as a judge to resolve a conflict between GPT-5.1 and DeepSeek classifications.
    
    Returns: The resolved value (0 or 1)
    """
    system_prompt = (
        "Act like an expert linguist and logical analyst specializing in Hebrew text analysis.\n\n"
        "Your goal is to resolve a disagreement between two independent AI classifiers about whether a given Hebrew text clearly mentions or discusses a specific topic, and to output a single definitive binary decision (0 or 1).\n\n"
        "Task:\n\n"
        "1) Carefully read the full Hebrew text and understand its meaning in context.\n\n"
        "2) Interpret the topic phrase precisely in Hebrew, including natural paraphrases and close formulations.\n\n"
        "3) Independently decide if the topic is clearly and explicitly mentioned or discussed:\n\n"
        "   - Output 1 only if the topic is clearly present, explicitly referenced, or directly discussed in the text.\n\n"
        "   - Output 0 if the topic is absent, only implied, vague, tangential, or requires external/world knowledge to connect.\n\n"
        "4) Use the classifier decisions only as background signals; do not average them or follow them blindly. Your judgment is final.\n\n"
        "Constraints:\n\n"
        "- Respond with ONLY a single character: 0 or 1.\n\n"
        "- No spaces, no newlines before or after, no explanations, no punctuation, no extra text.\n\n"
        "- Think rigorously about the relationship between the topic and the text before deciding."
    )
    
    user_prompt = (
        f"Question context: {question}\n\n"
        f"Hebrew text to analyze:\n\n"
        f"{text}\n\n"
        f"Topic in question (Hebrew):\n\n"
        f"{topic}\n\n"
        f"Classifier A decision: {gpt_value} (1 = topic IS present, 0 = topic is NOT present)\n"
        f"Classifier B decision: {deepseek_value} (1 = topic IS present, 0 = topic is NOT present)\n\n"
        f"After carefully analyzing the Hebrew text and applying the system instructions, decide whether the topic is clearly mentioned or directly discussed in the text. Do not rely on keywords alone or on the classifier outputs; base your answer on the actual content of the text.\n\n"
        f"Respond with ONLY: 0 or 1\n\n"
        f"Take a deep breath and work on this problem step-by-step."
    )
    
    url = config.get_chat_completions_url()
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
    
    if config.is_azure():
        payload["max_completion_tokens"] = 50  # Give enough room for the response
    else:
        payload["model"] = config.model
        payload["max_tokens"] = 50
        payload["temperature"] = 0
    
    for attempt in range(1, 3):  # 2 retries for judge
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            # Debug: check the response
            if response.status_code != 200:
                print(f"⚠️  Judge API error {response.status_code}: {response.text[:200]}", file=sys.stderr)
                response.raise_for_status()
            
            data = response.json()
            
            # Get the content
            content = ""
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "").strip()
            
            # Extract 0 or 1 from response
            # Check for "1" first since it's more specific
            if content == "1" or content.startswith("1"):
                return 1
            elif content == "0" or content.startswith("0"):
                return 0
            elif "1" in content and "0" not in content:
                return 1
            elif "0" in content and "1" not in content:
                return 0
            elif not content:
                # Empty response - retry with longer max tokens
                print(f"⚠️  Judge returned empty response (attempt {attempt})", file=sys.stderr)
                if attempt < 2:
                    payload["max_completion_tokens"] = 100  # Try with more tokens
                    continue
                # Default to GPT's classification on empty response
                print(f"    → Defaulting to GPT-5.1's answer: {gpt_value}", file=sys.stderr)
                return gpt_value
            else:
                # Ambiguous response - default to conservative (0)
                print(f"⚠️  Judge returned ambiguous response: '{content}', defaulting to 0", file=sys.stderr)
                return 0
                
        except Exception as e:
            print(f"⚠️  Judge error (attempt {attempt}/2): {e}", file=sys.stderr)
            if attempt == 2:
                # On failure, default to GPT-5.1's classification
                print(f"  → Defaulting to GPT-5.1's answer: {gpt_value}", file=sys.stderr)
                return gpt_value
            time.sleep(1)
    
    return gpt_value


def _compare_and_resolve_classifications(
    gpt_results: dict,
    deepseek_results: dict,
    texts: List[str],
    question: str,
    topics: Sequence[str],
    config: OpenAIConfig,
) -> Tuple[List[Dict[str, Union[int, str]]], List[Dict]]:
    """
    Compare GPT-5.1 and DeepSeek classifications, resolve conflicts using GPT-5.1 as judge.
    
    Returns:
        - final_classifications: List of resolved classifications (values are int for agreement, str like "1*" or "0*" for judge-resolved)
        - conflict_report: List of conflicts that were resolved (for reporting)
    """
    final_classifications: List[Dict[str, Union[int, str]]] = []
    conflict_report: List[Dict] = []
    
    for i in range(1, len(texts) + 1):
        text_key = str(i)
        text = texts[i - 1]
        
        # Safely get results, handling cases where YAML returned wrong type
        gpt_raw = gpt_results.get(text_key, {}) if isinstance(gpt_results, dict) else {}
        deepseek_raw = deepseek_results.get(text_key, {}) if isinstance(deepseek_results, dict) else {}
        
        # Ensure we have dicts, not strings
        gpt_class = gpt_raw if isinstance(gpt_raw, dict) else {}
        deepseek_class = deepseek_raw if isinstance(deepseek_raw, dict) else {}
        
        final_class: Dict[str, Union[int, str]] = {}
        text_conflicts: List[Dict] = []
        
        for topic in topics:
            # Safely extract values, handling malformed responses
            gpt_val_raw = gpt_class.get(topic, 0) if isinstance(gpt_class, dict) else 0
            deepseek_val_raw = deepseek_class.get(topic, 0) if isinstance(deepseek_class, dict) else 0
            gpt_val = int(gpt_val_raw in (1, "1", True))
            deepseek_val = int(deepseek_val_raw in (1, "1", True))
            
            if gpt_val == deepseek_val:
                # Agreement - use the agreed value
                final_class[topic] = gpt_val
            else:
                # Conflict - use judge to resolve
                print(f"  ⚖️  Conflict on text #{i}, topic '{topic[:30]}...': GPT={gpt_val}, DeepSeek={deepseek_val}", file=sys.stderr)
                
                resolved = _resolve_conflict_with_judge(
                    text=text,
                    question=question,
                    topic=topic,
                    gpt_value=gpt_val,
                    deepseek_value=deepseek_val,
                    config=config,
                )
                
                # Mark judge-resolved decisions with "*"
                final_class[topic] = f"{resolved}*"
                text_conflicts.append({
                    "text_index": i,
                    "topic": topic,
                    "gpt_value": gpt_val,
                    "deepseek_value": deepseek_val,
                    "resolved_value": resolved,
                })
                
                print(f"    → Judge resolved: {resolved}", file=sys.stderr)
                
                # Small delay between judge calls
                if config.request_delay_seconds > 0:
                    time.sleep(config.request_delay_seconds * 0.3)
        
        final_classifications.append(final_class)
        if text_conflicts:
            conflict_report.extend(text_conflicts)
    
    return final_classifications, conflict_report


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
    
    Optimized: Works backwards from the end to find the last non-empty row quickly.
    """
    original_rows = len(df)
    if original_rows == 0:
        return df, 0

    # Check only the first column for content
    # Work backwards from the end to find the last non-empty row
    first_col = df.iloc[:, 0]
    last_non_empty_idx = -1
    
    # Start from the end and work backwards
    for i in range(len(df) - 1, -1, -1):
        if _cell_has_content(first_col.iloc[i]):
            last_non_empty_idx = i
            break
    
    if last_non_empty_idx == -1:
        # All rows are empty, keep just the first row
        trimmed_df = df.iloc[:1].copy()
    else:
        # Keep everything up to and including the last non-empty row
        trimmed_df = df.iloc[:last_non_empty_idx + 1].copy()

    rows_removed = original_rows - len(trimmed_df)
    return trimmed_df, rows_removed


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
    
    # Create topic code mapping for reference
    topic_codes = [f"T{i+1}" for i in range(len(topic_columns))]
    
    prompt = (
        "You are a YAML repair expert. The following YAML should map row numbers to topic classifications "
        "using topic codes (T1, T2, etc.), but it contains formatting errors.\n\n"
        "🎯 YOUR TASK: Fix the YAML so it becomes 100% valid and strictly follows the required schema.\n\n"
        f"{error_context}"
        "✅ REQUIRED YAML FORMAT:\n"
        "1: {T1: 0, T2: 1, T3: 0, ...}\n"
        "2: {T1: 1, T2: 0, T3: 1, ...}\n"
        "...\n\n"
        "🔍 CRITICAL RULES:\n"
        "1. Use YAML mappings with inline format: {T1: 0, T2: 1, ...}\n"
        "2. Topic keys must be T1, T2, T3, etc. (codes, not full names).\n"
        "3. Values must be integers 0 or 1 (not booleans or strings).\n"
        "4. Keep the row numbering identical (1, 2, ...).\n"
        "5. Do not wrap the output in markdown fences or add commentary.\n"
        "6. Preserve all rows and topics present in the input.\n\n"
        f"📋 Expected topic codes: {', '.join(topic_codes)}\n\n"
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
        error_msg = str(exc)
        print(f"DeepSeek YAML repair failed: {error_msg}", file=sys.stderr)
        # Add error to log for next attempt
        if error_log is not None:
            error_log.append(f"DeepSeek repair error: {error_msg}")
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

    # Create topic code mapping for reference
    topic_codes = [f"T{i+1}" for i in range(len(topic_columns))]
    
    prompt = (
        "You previously generated YAML for topic classifications, but it failed to parse.\n"
        "Fix ONLY the formatting issues so the YAML becomes valid.\n\n"
        "🎯 REQUIRED FORMAT:\n"
        "1: {T1: 0, T2: 1, T3: 0, ...}\n"
        "2: {T1: 1, T2: 0, T3: 1, ...}\n\n"
        "🔍 RULES:\n"
        "1. Use topic codes (T1, T2, etc.) as keys, NOT full topic names.\n"
        "2. Keep the same row numbering (1, 2, ...).\n"
        "3. Values must be integers 0 or 1.\n"
        "4. Output must be pure YAML (no code fences, no commentary).\n"
        f"5. Expected topic codes: {', '.join(topic_codes)}\n"
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
        error_msg = str(exc)
        print(f"GPT YAML repair failed: {error_msg}", file=sys.stderr)
        # Add error to log for next attempt
        if error_log is not None:
            error_log.append(f"GPT repair error: {error_msg}")
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
    2. Alternating DeepSeek and GPT repair (5 attempts total: 3 DeepSeek, 2 GPT)
       - Starts with DeepSeek, then alternates
       - Always includes error messages as context
    
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
    
    # Stage 2: Alternating DeepSeek and GPT repair (5 attempts: 3 DeepSeek, 2 GPT)
    print("⚠️  GPT-5.1 self-repair failed. Stage 2: Attempting alternating repair (DeepSeek → GPT → DeepSeek → GPT → DeepSeek)...", file=sys.stderr)
    
    # Pattern: DeepSeek, GPT, DeepSeek, GPT, DeepSeek (3 DeepSeek, 2 GPT)
    repair_sequence = [
        ("DeepSeek", _repair_yaml_with_deepseek),
        ("GPT", _repair_yaml_with_gpt),
        ("DeepSeek", _repair_yaml_with_deepseek),
        ("GPT", _repair_yaml_with_gpt),
        ("DeepSeek", _repair_yaml_with_deepseek),
    ]
    
    for attempt, (model_name, repair_func) in enumerate(repair_sequence, start=1):
        error_log.append(f"{model_name} repair attempt {attempt}/5")
        repaired = repair_func(
            broken_yaml, topic_columns, config, error_message=error_message, error_log=error_log
        )
        if repaired is not None:
            print(f"✅ YAML repaired successfully with {model_name} (attempt {attempt}/5)", file=sys.stderr)
            return repaired
        else:
            print(f"❌ {model_name} repair attempt {attempt}/5 failed", file=sys.stderr)
            if attempt < 5:
                time.sleep(1 * attempt)  # Brief delay between attempts
    
    print("❌ All repair attempts failed (1 GPT-5.1 self-repair + 5 alternating repairs)", file=sys.stderr)
    return None


# ================================================================
# TOPIC CONTEXT GENERATION
# ================================================================

def _generate_topic_context(
    topics: Sequence[str], 
    answer_column_name: str, 
    config: OpenAIConfig
) -> str:
    """
    Generate a detailed context explaining what each topic means.
    This is called once at the start of classification to give the model
    full understanding of the topics before processing texts.
    
    Args:
        topics: List of topic column names/labels
        answer_column_name: Name of the answer column (may provide context)
        config: API configuration
    
    Returns:
        A detailed context string explaining each topic
    """
    print("📚 Generating topic context for better classification...")
    
    topics_list = "\n".join([f"- T{i+1}: {topic}" for i, topic in enumerate(topics)])
    
    system_prompt = (
        "You are an expert in Hebrew language and academic research analysis.\n\n"
        "Your task is to create a detailed understanding of each topic that will be used for text classification.\n"
        "For each topic, provide a clear explanation (max 15 words) of what it represents, including:\n"
        "- The core concept/theme\n"
        "- Related terms, synonyms, or expressions in Hebrew that would indicate this topic\n"
        "- What kind of content would match this topic\n\n"
        "Output format:\n"
        "T1: [brief explanation of topic 1 and related terms]\n"
        "T2: [brief explanation of topic 2 and related terms]\n"
        "...\n\n"
        "CRITICAL RULES:\n"
        "- Keep each explanation concise but comprehensive (max 15 words per topic).\n"
        "- Focus on semantic meaning, not just literal keywords.\n"
        "- NEVER include phrases like 'no relevant answer', 'not applicable', 'none', or similar.\n"
        "- Every topic MUST have a meaningful explanation of what it represents.\n"
        "- If a topic seems unclear, interpret it as broadly as possible based on the survey context.\n"
        "- The classification model will output 0 for all topics if no match is found - you don't need to handle that."
    )
    
    user_prompt = (
        f"Survey/Question Context: '{answer_column_name}'\n\n"
        f"Topics to explain:\n{topics_list}\n\n"
        f"Please provide a clear, meaningful explanation for each topic (T1, T2, etc.) that helps understand "
        f"what kind of Hebrew text content would match each topic. Consider the survey context above.\n\n"
        f"Remember: Every topic must have a real explanation. Do NOT use 'no answer' or 'not applicable' phrases."
    )
    
    url = config.get_chat_completions_url()
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
    
    if config.is_azure():
        payload["max_completion_tokens"] = 1500
    else:
        payload["model"] = config.model
        payload["max_tokens"] = 1500
        payload["temperature"] = 0
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        context = data["choices"][0]["message"]["content"].strip()
        print("✅ Topic context generated successfully")
        
        # Print the full context to the progress console so user can see it
        print("=" * 50)
        print("📋 TOPIC UNDERSTANDING (Generated)")
        print("=" * 50)
        # Print each line of the context
        for line in context.split("\n"):
            if line.strip():
                print(f"  {line.strip()}")
        print("=" * 50)
        
        return context
        
    except Exception as e:
        print(f"⚠️  Failed to generate topic context: {e}")
        print("   Continuing without detailed context...")
        # Return a basic context if generation fails
        return "\n".join([f"T{i+1}: {topic}" for i, topic in enumerate(topics)])


# ================================================================
# CLASSIFICATION
# ================================================================

def classify_batch(
    texts: Sequence[str], 
    topics: Sequence[str], 
    question: str, 
    config: OpenAIConfig,
    topic_context: str = ""
) -> List[Dict[str, Union[int, str]]]:
    """
    Classify a batch of texts using PARALLEL DUAL-MODEL approach.
    
    Process:
    1. Run GPT-5.1 and DeepSeek classifications IN PARALLEL
    2. Compare results to identify agreements and conflicts
    3. For conflicts, use GPT-5.1 as a "judge" to make final decision
    
    This approach replaces logprobs-based confidence (unavailable on Azure)
    with model agreement as a confidence signal:
    - Agreement = high confidence (both models agree)
    - Disagreement = low confidence → resolve with judge
    
    Returns: List of {topic: 0/1 or "0*"/"1*"} dicts in the same order as input texts.
            Values with "*" indicate judge-resolved conflicts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Filter out short texts - return all 0s for them
    results: List[Optional[Dict[str, Union[int, str]]]] = []
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
    
    # Prepare classification inputs
    topics_text = ", ".join(topics)
    process_texts = [text for _, text in texts_to_process]
    enumerated_texts = "\n\n".join(f"{i+1}. {question}: {text}" for i, text in enumerate(process_texts))
    classify_inputs = {
        "topics": topics_text,
        "texts": enumerated_texts,
        "question": question,
        "topic_context": topic_context,  # Pre-generated detailed topic understanding
    }
    
    print("  🔄 Running parallel classification (GPT-5.1 + DeepSeek)...", file=sys.stderr)
    
    # Step 1: Run GPT-5.1 and DeepSeek in PARALLEL
    gpt_results = None
    deepseek_results = None
    gpt_error = None
    deepseek_error = None
    
    def run_gpt():
        try:
            results, _ = _classify_with_logprobs(classify_inputs, topics, config)
            return results
        except Exception as e:
            return e
    
    def run_deepseek():
        try:
            return _classify_with_deepseek(classify_inputs, topics, config)
        except Exception as e:
            return e
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        gpt_future = executor.submit(run_gpt)
        deepseek_future = executor.submit(run_deepseek)
        
        gpt_result = gpt_future.result()
        deepseek_result = deepseek_future.result()
        
        if isinstance(gpt_result, Exception):
            gpt_error = gpt_result
            print(f"  ⚠️  GPT-5.1 classification failed: {gpt_error}", file=sys.stderr)
        else:
            gpt_results = gpt_result
            print(f"  ✅ GPT-5.1 classification complete", file=sys.stderr)
        
        if isinstance(deepseek_result, Exception):
            deepseek_error = deepseek_result
            print(f"  ⚠️  DeepSeek classification failed: {deepseek_error}", file=sys.stderr)
        else:
            deepseek_results = deepseek_result
            print(f"  ✅ DeepSeek classification complete", file=sys.stderr)
    
    # Handle failures
    if gpt_results is None and deepseek_results is None:
        # Both failed - raise error
        raise RuntimeError(f"Both GPT-5.1 and DeepSeek classification failed. GPT error: {gpt_error}. DeepSeek error: {deepseek_error}")
    
    if gpt_results is None:
        # Only DeepSeek succeeded - use its results
        print(f"  ⚠️  Using DeepSeek results only (GPT-5.1 failed)", file=sys.stderr)
        final_classifications = []
        for i in range(1, len(process_texts) + 1):
            item_raw = deepseek_results.get(str(i), {}) if isinstance(deepseek_results, dict) else {}
            item = item_raw if isinstance(item_raw, dict) else {}
            final_classifications.append({t: int(item.get(t, 0) in (1, "1", True)) for t in topics})
    elif deepseek_results is None:
        # Only GPT-5.1 succeeded - use its results
        print(f"  ⚠️  Using GPT-5.1 results only (DeepSeek failed)", file=sys.stderr)
        final_classifications = []
        for i in range(1, len(process_texts) + 1):
            item_raw = gpt_results.get(str(i), {}) if isinstance(gpt_results, dict) else {}
            item = item_raw if isinstance(item_raw, dict) else {}
            final_classifications.append({t: int(item.get(t, 0) in (1, "1", True)) for t in topics})
    else:
        # Both succeeded - compare and resolve conflicts
        print(f"  ⚖️  Comparing results and resolving conflicts...", file=sys.stderr)
        
        final_classifications, conflict_report = _compare_and_resolve_classifications(
            gpt_results=gpt_results,
            deepseek_results=deepseek_results,
            texts=process_texts,
            question=question,
            topics=topics,
            config=config,
        )
        
        # Report conflict statistics
        if conflict_report:
            # Count unique texts with conflicts
            texts_with_conflicts = len(set(c['text_index'] for c in conflict_report))
            print(f"  📊 Resolved {len(conflict_report)} conflicts across {texts_with_conflicts} texts", file=sys.stderr)
            
            # Store conflict report for later analysis
            if not hasattr(classify_batch, 'conflict_reports'):
                classify_batch.conflict_reports = []
            classify_batch.conflict_reports.append({
                'batch_texts': len(process_texts),
                'total_conflicts': len(conflict_report),
                'texts_with_conflicts': texts_with_conflicts,
                'details': conflict_report
            })
        else:
            print(f"  ✅ No conflicts - both models agreed on all classifications", file=sys.stderr)
    
    # Merge results back into original order
    validated_idx = 0
    result_idx = 0
    for idx, text in enumerate(texts):
        if not text or len(text) < 2:
            # Already filled with all 0s, skip
            result_idx += 1
        else:
            # Replace placeholder with final result
            results[result_idx] = final_classifications[validated_idx]
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

    # Create length-balanced batches
    # Calculate number of batches needed (at least 1, at most len(rows_to_process))
    num_batches = max(1, (len(rows_to_process) + config.batch_size - 1) // config.batch_size)
    
    # Calculate text lengths and create (length, row_data) tuples
    rows_with_lengths = [(len(text), (idx, text)) for idx, text in rows_to_process]
    
    # Sort by length descending (largest first) for better bin-packing
    rows_with_lengths.sort(reverse=True)
    
    # Initialize batches: each batch is a list of (idx, text) tuples and tracks total length
    batches: List[Tuple[int, List[Tuple[Any, str]]]] = []
    for _ in range(num_batches):
        batches.append((0, []))  # (total_length, list of rows)
    
    # Greedy bin-packing: assign each text to the batch with smallest current total length
    for text_length, (idx, text) in rows_with_lengths:
        # Find batch with smallest total length that isn't full
        best_batch_idx = 0
        best_batch_length = batches[0][0]
        
        for i, (batch_length, batch_rows) in enumerate(batches):
            # Prefer batches that aren't full and have smaller total length
            if len(batch_rows) < config.batch_size:
                if len(batches[best_batch_idx][1]) >= config.batch_size or batch_length < best_batch_length:
                    best_batch_idx = i
                    best_batch_length = batch_length
        
        # Add to best batch
        current_length, current_rows = batches[best_batch_idx]
        current_rows.append((idx, text))
        batches[best_batch_idx] = (current_length + text_length, current_rows)
    
    # Extract just the row lists from batches
    batches = [batch_rows for _, batch_rows in batches if batch_rows]  # Remove empty batches
    
    # Calculate statistics for reporting
    batch_lengths = [sum(len(text) for _, text in batch) for batch in batches]
    avg_length = sum(batch_lengths) / len(batch_lengths) if batch_lengths else 0
    min_length = min(batch_lengths) if batch_lengths else 0
    max_length = max(batch_lengths) if batch_lengths else 0
    
    print(
        f"Processing {len(rows_to_process)} rows in {len(batches)} length-balanced batches "
        f"(max_items={config.batch_size}, workers={config.parallel_workers})..."
    )
    print(
        f"  Batch length stats: avg={avg_length:.0f} chars, min={min_length:.0f}, max={max_length:.0f} "
        f"(spread: {((max_length - min_length) / avg_length * 100) if avg_length > 0 else 0:.1f}%)",
        file=sys.stderr
    )

    # Generate detailed topic context ONCE before processing all batches
    # This gives the model full understanding of what each topic means
    topic_context = _generate_topic_context(topic_cols, main_column, config)

    failed_batches: List[Tuple[int, List[Any], Exception]] = []

    with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures: Dict[Any, Tuple[int, List[Tuple[Any, str]]]] = {}

        for batch_idx, batch in enumerate(batches, start=1):
            texts = [text for _, text in batch]
            future = executor.submit(classify_batch, texts, topic_cols, main_column, config, topic_context)
            futures[future] = (batch_idx, batch)

        for future in as_completed(futures):
            batch_idx, batch = futures[future]
            try:
                results = future.result()
                for (row_idx, text), topics_result in zip(batch, results):
                    for topic, value in topics_result.items():
                        df.at[row_idx, topic] = value
                print(f"✔ Batch {batch_idx}/{len(batches)} processed by model")
                    
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
