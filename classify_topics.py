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
# API CALL WITH BACKOFF & REPAIR
# ================================================================

def _classify_with_logprobs(inputs: dict, topics: Sequence[str], config: OpenAIConfig) -> Tuple[dict, Dict[str, float]]:
    """
    Make a direct API call to get classification with logprobs for confidence calculation.
    Returns: (results_dict, per_text_confidence_dict)
    where per_text_confidence_dict maps text number ("1", "2", ...) to confidence score.
    """
    import requests
    import math
    
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
            
            # Note: Azure doesn't support logprobs, so we return default confidence
            # Confidence is now determined by model agreement (GPT vs DeepSeek)
            text_keys = list(initial_results.keys())
            per_text_confidence = {key: 1.0 for key in text_keys}
            
            return initial_results, per_text_confidence
            
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


def _classify_with_deepseek(inputs: dict, topics: Sequence[str], config: OpenAIConfig) -> dict:
    """
    Classify texts using DeepSeek model (for parallel comparison with GPT-5.1).
    Returns: results_dict mapping text number to topic classifications.
    """
    # Build the prompt (same as GPT-5.1)
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
        "max_tokens": 3000,
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
                return results
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
                    return repaired
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
        "You are an expert linguist and logical analyst specializing in Hebrew text analysis. "
        "Two independent AI classifiers have analyzed the same text and reached different conclusions "
        "about whether a specific topic is present. Your task is to carefully analyze the text and "
        "make the final determination.\n\n"
        "You must respond with ONLY a single digit: 0 or 1.\n"
        "- 1 = The topic IS clearly mentioned or discussed in the text\n"
        "- 0 = The topic is NOT clearly mentioned or discussed in the text\n\n"
        "Be rigorous: only return 1 if the topic is explicitly present, not merely implied."
    )
    
    user_prompt = (
        f"Two expert classifiers have analyzed the following Hebrew text and disagree about whether "
        f"a specific topic is present.\n\n"
        f"**Question context:** {question}\n\n"
        f"**Text to analyze (Hebrew):**\n{text}\n\n"
        f"**Topic in question:** {topic}\n\n"
        f"**Classifier A (GPT-5.1) says:** {gpt_value} ({'topic IS present' if gpt_value == 1 else 'topic is NOT present'})\n"
        f"**Classifier B (DeepSeek) says:** {deepseek_value} ({'topic IS present' if deepseek_value == 1 else 'topic is NOT present'})\n\n"
        f"After careful analysis of the text, is the topic '{topic}' clearly mentioned or discussed?\n\n"
        f"Respond with ONLY: 0 or 1"
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
) -> Tuple[List[Dict[str, int]], List[Dict]]:
    """
    Compare GPT-5.1 and DeepSeek classifications, resolve conflicts using GPT-5.1 as judge.
    
    Returns:
        - final_classifications: List of resolved classifications
        - conflict_report: List of conflicts that were resolved (for reporting)
    """
    final_classifications: List[Dict[str, int]] = []
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
        
        final_class: Dict[str, int] = {}
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
                
                final_class[topic] = resolved
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
        
        # Ensure all topics are present - handle case where result is not a dict
        validated = {}
        result_dict = result if isinstance(result, dict) else {}
        for topic in topics:
            value = result_dict.get(topic, initial_classifications.get(topic, 0))
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
        
        # Get the value for this topic - handle case where result is not a dict
        result_dict = result if isinstance(result, dict) else {}
        value = result_dict.get(topic, initial_value)
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


# ================================================================
# CLASSIFICATION
# ================================================================

def classify_batch(texts: Sequence[str], topics: Sequence[str], question: str, config: OpenAIConfig) -> List[Dict[str, int]]:
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
    
    Returns: List of {topic: 0/1} dicts in the same order as input texts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
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
    
    # Prepare classification inputs
    topics_text = ", ".join(topics)
    process_texts = [text for _, text in texts_to_process]
    enumerated_texts = "\n\n".join(f"{i+1}. {question}: {text}" for i, text in enumerate(process_texts))
    classify_inputs = {
        "topics": topics_text,
        "texts": enumerated_texts,
        "question": question,
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
