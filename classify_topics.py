#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests
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
    batch_size: int = 5           # smaller = fewer 429s
    parallel_workers: int = 2     # lower concurrency for LiteLLM
    request_delay_seconds: float = 0.5  # delay between API requests to avoid rate limits
    # Gemini config for JSON repair
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-lite"

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

        gemini_key = os.getenv("GEMINI_AI_KEY", "")
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

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
            gemini_api_key=gemini_key,
            gemini_model=gemini_model,
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
    
    # Create LLM with LiteLLM base URL
    classification_max_tokens = max(config.max_tokens, 1000)
    validation_max_tokens = max(config.max_tokens, 800)
    
    llm_classify = ChatOpenAI(
        model=config.model,
        base_url=config.api_base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=classification_max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    
    llm_validate = ChatOpenAI(
        model=config.model,
        base_url=config.api_base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=validation_max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

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
    
    parser = JsonOutputParser()
    
    # Transform classification result for validation
    def format_for_validation(classify_result: dict) -> dict:
        """Transform batch classification result into individual validation inputs."""
        # This will be called per text, so we'll handle it in classify_batch
        return classify_result
    
    # Unified chain: classify -> validate (will be used per text in classify_batch)
    classify_chain = classify_prompt | llm_classify | parser
    validate_chain = validate_prompt | llm_validate | parser
    
    return classify_chain, validate_chain, topics_text


# ================================================================
# API CALL WITH BACKOFF & REPAIR
# ================================================================

def _extract_json(content: str) -> str:
    """
    Extract the first JSON object found in the response string.

    Handles common formatting issues such as Markdown code fences and leading text.
    """
    content = content.strip()
    if content.startswith("```"):
        # Remove Markdown fences such as ```json ... ```
        parts = content.strip("`").split("```")
        content = parts[1] if len(parts) > 1 else parts[0]
    content = content.strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("No JSON object found", content, 0)
    return content[start : end + 1]


def _repair_json_with_gemini(
    broken_json: str,
    topic_columns: Sequence[str],
    config: OpenAIConfig,
) -> Optional[dict]:
    """
    Use Gemini to repair malformed JSON.
    """
    if not config.gemini_api_key:
        print("No Gemini API key configured; skipping repair.", file=sys.stderr)
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.gemini_model}:generateContent?key={config.gemini_api_key}"
    
    prompt = (
        "The following text is intended to be JSON mapping row numbers to topic flags (in Hebrew), "
        "but it is invalid. Fix the JSON so that it is valid and matches this schema:\n"
        "{\n"
        '  "1": {"<topic name in Hebrew>": 0 or 1, ...},\n'
        '  "2": {"<topic name in Hebrew>": 0 or 1, ...},\n'
        "  ...\n"
        "}\n\n"
        "Rules:\n"
        "- The topic names are in Hebrew and must be preserved exactly.\n"
        "- Use each topic exactly once per row.\n"
        "- Respond with valid JSON only, no markdown fences or commentary.\n"
        "- Properly escape all strings.\n\n"
        f"Topics: {', '.join(topic_columns)}\n\n"
        "Broken JSON:\n"
        f"{broken_json}"
    )

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048,
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Extract JSON from Gemini response (may have markdown fences)
        content = _extract_json(content)
        return json.loads(content)
    except Exception as exc:
        print(f"Gemini JSON repair failed: {exc}", file=sys.stderr)
        return None


def invoke_chain_with_retry(chain, inputs: dict, topic_columns: Sequence[str], config: OpenAIConfig) -> dict:
    """
    Invoke the LangChain chain with retry logic and JSON repair fallback.
    """
    for attempt in range(1, config.max_retries + 1):
        try:
            result = chain.invoke(inputs)
            # Validate result structure
            if isinstance(result, dict):
                return result
            # If result is a string, try to parse it
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    repaired = _extract_json(result)
                    return json.loads(repaired)
            raise ValueError(f"Unexpected result type: {type(result)}")
        except Exception as e:
            # Try Gemini repair if we have a string response
            if isinstance(e, json.JSONDecodeError) or "JSON" in str(e):
                try:
                    # Get raw content if available
                    raw_content = str(e) if hasattr(e, 'args') else None
                    if raw_content:
                        repaired_dict = _repair_json_with_gemini(raw_content, topic_columns, config)
                        if repaired_dict is not None:
                            return repaired_dict
                except Exception:
                    pass
            
            if attempt == config.max_retries:
                raise RuntimeError(f"Failed after {attempt} retries: {e}")
            
            wait = config.retry_backoff_seconds * attempt
            if "429" in str(e) or "Rate limit" in str(e):
                print(f"Rate limited. Sleeping {wait:.1f}s...", file=sys.stderr)
            elif "500" in str(e) or "Server" in str(e):
                print(f"Server error. Retry in {wait:.1f}s...", file=sys.stderr)
            else:
                print(f"Error {e}. Retrying in {wait:.1f}s...", file=sys.stderr)
            time.sleep(wait)


# ================================================================
# CLASSIFICATION
# ================================================================

def classify_batch(texts: Sequence[str], topics: Sequence[str], question: str, config: OpenAIConfig) -> List[Dict[str, int]]:
    """
    Classify a batch of texts using the unified LangChain chain (classification + validation).
    Skips texts that are empty or have less than 2 characters (returns all 0s).
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
    
    # Create unified chains
    classify_chain, validate_chain, topics_text = create_classification_chain(config, topics, question)
    
    # Step 1: Batch classification (only for texts that passed the filter)
    process_texts = [text for _, text in texts_to_process]
    enumerated_texts = "\n\n".join(f"{i+1}. {question}: {text}" for i, text in enumerate(process_texts))
    classify_inputs = {
        "topics": topics_text,
        "texts": enumerated_texts,
    }
    
    initial_results = invoke_chain_with_retry(classify_chain, classify_inputs, topics, config)
    
    # Rate limiting: add delay after classification before starting validation
    if config.request_delay_seconds > 0:
        time.sleep(config.request_delay_seconds)
    
    # Parse initial results
    initial_classifications: List[Dict[str, int]] = []
    for i in range(1, len(process_texts) + 1):
        item = initial_results.get(str(i), {}) if isinstance(initial_results, dict) else {}
        initial_classifications.append({t: int(item.get(t, 0) in (1, "1", True)) for t in topics})
    
    # Step 2: Validate each classification using the unified chain
    validated_classifications: List[Dict[str, int]] = []
    
    for idx, (text, initial_class) in enumerate(zip(process_texts, initial_classifications)):
        # Skip validation for short texts
        if not text or len(text) < 2:
            validated_classifications.append({t: 0 for t in topics})
            continue
        
        classifications_str = ", ".join(f'"{topic}": {value}' for topic, value in initial_class.items())
        
        validate_inputs = {
            "question": question,
            "text": text,
            "classifications": classifications_str,
            "topics": topics_text,
        }
        
        try:
            validated = invoke_chain_with_retry(validate_chain, validate_inputs, topics, config)
            validated_class = {
                t: int(validated.get(t, initial_class.get(t, 0)) in (1, "1", True))
                for t in topics
            }
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
            except Exception as exc:
                print(f"✖ Batch {batch_idx} failed: {exc}", file=sys.stderr)

    return df


# ================================================================
# CLI
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Efficient LLM-based binary topic classification.")
    p.add_argument("--input", type=Path, default=Path("open_question_data.xlsx"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OpenAIConfig.from_env()

    df = pd.read_excel(args.input)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    if len(df.columns) <= 10:
        raise ValueError(f"Expected ≥11 columns, got {len(df.columns)}")

    main_column = df.columns[8]
    topics = df.columns[9:].tolist()

    print(f"Main text column (index 8): {main_column}")
    print(f"Topics ({len(topics)}): {topics}")
    print(f"Model: {cfg.model}, Base URL: {cfg.api_base_url}")

    df = update_topics(
        df,
        main_column,
        topics,
        cfg,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )

    out = args.output or args.input.with_name("open_question_data_classified.xlsx")
    df.to_excel(out, index=False)
    print(f"✅ Done. Saved to {out}")


if __name__ == "__main__":
    main()
