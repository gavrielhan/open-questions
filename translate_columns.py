#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv


# ================================================================
# CONFIGURATION
# ================================================================

@dataclass
class OpenAIConfig:
    api_key: str
    api_base_url: str
    model: str
    max_tokens: int = 2000
    temperature: float = 0.0
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    batch_size: int = 10
    parallel_workers: int = 2
    request_delay_seconds: float = 0.5

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

        return cls(
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "5")),
            retry_backoff_seconds=float(os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", "2")),
            batch_size=int(os.getenv("TRANSLATE_BATCH_SIZE", "10")),
            parallel_workers=int(os.getenv("OPENAI_PARALLEL_WORKERS", "2")),
            request_delay_seconds=float(os.getenv("OPENAI_REQUEST_DELAY_SECONDS", "0.5")),
        )


# ================================================================
# TRANSLATION
# ================================================================

def translate_text(text: str, config: OpenAIConfig) -> str:
    """Translate a single text from Hebrew to English."""
    if not text or not text.strip():
        return text
    
    # Normalize whitespace
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return ""
    
    url = config.api_base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }
    
    prompt = (
        "You are a professional translator specializing in Hebrew to English translation. "
        "Translate the following Hebrew text accurately and naturally into English. "
        "Preserve the meaning, tone, and context. Do not add explanations, only the translation.\n\n"
        f"{text}"
    )
    
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional translator. Translate Hebrew text to English accurately and naturally."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature
    }
    
    for attempt in range(1, config.max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
            translated = data["choices"][0]["message"]["content"].strip()
            return translated
        except Exception as e:
            if attempt == config.max_retries:
                print(f"Translation failed after {attempt} retries: {e}", file=sys.stderr)
                return text  # Return original on failure
            
            wait = config.retry_backoff_seconds * attempt
            if "429" in str(e) or "Rate limit" in str(e):
                print(f"Rate limited. Sleeping {wait:.1f}s...", file=sys.stderr)
            else:
                print(f"Error {e}. Retrying in {wait:.1f}s...", file=sys.stderr)
            time.sleep(wait)
    
    return text


def translate_batch(texts: Sequence[str], config: OpenAIConfig) -> List[str]:
    """Translate a batch of texts, skipping empty cells."""
    results = []
    for idx, text in enumerate(texts):
        # Skip empty cells - no API call needed
        if not text or not text.strip():
            results.append(text)
            continue
        
        # Normalize whitespace and check again
        text_normalized = text.replace("\n", " ").replace("\r", " ").strip()
        if not text_normalized:
            results.append("")
            continue
        
        # Only translate non-empty cells
        translated = translate_text(text, config)
        results.append(translated)
        
        # Rate limiting: add delay between requests (except for the last one)
        if idx < len(texts) - 1 and config.request_delay_seconds > 0:
            time.sleep(config.request_delay_seconds)
    
    return results


# ================================================================
# MAIN TRANSLATION LOGIC
# ================================================================

def translate_columns(
    df: pd.DataFrame,
    start_column_idx: int,
    config: OpenAIConfig,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Translate columns from start_column_idx onwards."""
    columns_to_translate = df.columns[start_column_idx:].tolist()
    print(f"Translating {len(columns_to_translate)} columns: {columns_to_translate}")
    print(f"Total rows: {len(df)}")
    
    # Create new dataframe with only translated columns
    translated_df = pd.DataFrame()
    translated_df.index = df.index
    
    # Translate each column
    for col_idx, col_name in enumerate(columns_to_translate):
        print(f"\nTranslating column {col_idx + 1}/{len(columns_to_translate)}: {col_name}")
        
        # Get all values for this column
        values = df[col_name].fillna("").astype(str).tolist()
        
        # Apply limit if specified
        if limit is not None:
            values = values[:limit]
        
        # Normalize whitespace
        values = [v.replace("\n", " ").replace("\r", " ").strip() for v in values]
        
        # Process in batches
        batches = [
            values[i : i + config.batch_size]
            for i in range(0, len(values), config.batch_size)
        ]
        
        translated_values = []
        
        with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
            futures = {}
            
            for batch_idx, batch in enumerate(batches, start=1):
                future = executor.submit(translate_batch, batch, config)
                futures[future] = (batch_idx, len(batches))
            
            for future in as_completed(futures):
                batch_idx, total_batches = futures[future]
                try:
                    batch_results = future.result()
                    translated_values.extend(batch_results)
                    print(f"  ✔ Batch {batch_idx}/{total_batches} translated")
                except Exception as exc:
                    print(f"  ✖ Batch {batch_idx} failed: {exc}", file=sys.stderr)
                    # Fill with original values on failure
                    batch_start = (batch_idx - 1) * config.batch_size
                    translated_values.extend(values[batch_start:batch_start + len(batch)])
        
        # Add translated column to new dataframe
        translated_df[col_name] = translated_values[:len(df)]
    
    return translated_df


# ================================================================
# CLI
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate columns from index 8 onwards from Hebrew to English.")
    p.add_argument("--input", type=Path, default=Path("open_question_data.xlsx"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--start-column", type=int, default=8, help="Column index to start translating from (default: 8)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of rows to translate")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OpenAIConfig.from_env()

    # Read Excel file
    print(f"Reading Excel file: {args.input}")
    df = pd.read_excel(args.input)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    
    if len(df.columns) <= args.start_column:
        raise ValueError(f"Expected at least {args.start_column + 1} columns, got {len(df.columns)}")

    print(f"Model: {cfg.model}, Base URL: {cfg.api_base_url}")
    print(f"Starting translation from column index {args.start_column}")

    # Translate columns
    translated_df = translate_columns(
        df,
        args.start_column,
        cfg,
        limit=args.limit,
    )

    # Save to output file
    out = args.output or args.input.with_name(f"{args.input.stem}_translated.xlsx")
    translated_df.to_excel(out, index=False)
    print(f"\n✅ Done. Saved translated columns to {out}")


if __name__ == "__main__":
    main()

