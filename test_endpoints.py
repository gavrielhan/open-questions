#!/usr/bin/env python3
"""
Test script to verify Azure AI Foundry endpoints for both GPT-5.1 and DeepSeek.

Usage:
    python test_endpoints.py
    
Make sure your .env file is in the same directory, or set OPENAI_API_KEY environment variable.
"""
import os
import sys
from openai import OpenAI

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, relying on environment variables")

# Check for API key
if not os.getenv("API_KEY"):
    print("❌ ERROR: API_KEY not found in environment or .env file")
    print("\nPlease either:")
    print("  1. Create a .env file with API_KEY=your_key")
    print("  2. Set environment variable: export API_KEY=your_key")
    sys.exit(1)

def test_gpt51_endpoint():
    """Test GPT-5.1 endpoint"""
    print("=" * 60)
    print("Testing GPT-5.1 Endpoint")
    print("=" * 60)
    
    endpoint = os.getenv("API_BASE_URL", "https://sni-ai-foundry.cognitiveservices.azure.com")
    deployment_name = os.getenv("MODEL", "gpt-5.1-gavriel")
    api_key = os.getenv("API_KEY")
    api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
    
    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment_name}")
    print(f"API Version: {api_version}")
    print()
    
    try:
        # Use direct requests to match how classify_topics.py works
        import requests
        
        url = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in one word.",
                }
            ],
            "max_completion_tokens": 50,
        }
        
        print("Sending test request...")
        print(f"URL: {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        completion = response.json()
        
        print("✅ SUCCESS!")
        content = completion["choices"][0]["message"]["content"]
        model = completion.get("model", deployment_name)
        tokens = completion["usage"]["total_tokens"]
        print(f"Response: {content}")
        print(f"Model: {model}")
        print(f"Tokens used: {tokens}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_deepseek_endpoint():
    """Test DeepSeek endpoint"""
    print("\n" + "=" * 60)
    print("Testing DeepSeek Endpoint")
    print("=" * 60)
    
    endpoint = os.getenv("REPAIR_ENDPOINT", "https://sni-ai-foundry.services.ai.azure.com/openai/v1/")
    deployment_name = os.getenv("REPAIR_MODEL", "DeepSeek-V3.1-gavriel")
    api_key = os.getenv("REPAIR_API_KEY") or os.getenv("API_KEY")
    
    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment_name}")
    print()
    
    try:
        client = OpenAI(
            base_url=endpoint,
            api_key=api_key
        )
        
        print("Sending test request...")
        completion = client.chat.completions.create(
            model=deployment_name,  # Deployment name
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in one word.",
                }
            ],
            max_tokens=50,
        )
        
        print("✅ SUCCESS!")
        print(f"Response: {completion.choices[0].message.content}")
        print(f"Model: {completion.model}")
        print(f"Tokens used: {completion.usage.total_tokens}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_deepseek_json_mode():
    """Test DeepSeek with JSON output (for re-evaluation)"""
    print("\n" + "=" * 60)
    print("Testing DeepSeek JSON Mode (Re-evaluation)")
    print("=" * 60)
    
    endpoint = os.getenv("REPAIR_ENDPOINT", "https://sni-ai-foundry.services.ai.azure.com/openai/v1/")
    deployment_name = os.getenv("REPAIR_MODEL", "DeepSeek-V3.1-gavriel")
    api_key = os.getenv("REPAIR_API_KEY") or os.getenv("API_KEY")
    
    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment_name}")
    print()
    
    try:
        client = OpenAI(
            base_url=endpoint,
            api_key=api_key
        )
        
        print("Sending test request with JSON format...")
        completion = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Always respond in valid JSON format."
                },
                {
                    "role": "user",
                    "content": 'Classify if this text is about "Technology": "The new iPhone has amazing features." Return JSON: {"Technology": 0 or 1}',
                }
            ],
            max_tokens=100,
            temperature=0.7,
        )
        
        print("✅ SUCCESS!")
        print(f"Response: {completion.choices[0].message.content}")
        print(f"Model: {completion.model}")
        print(f"Tokens used: {completion.usage.total_tokens}")
        
        # Try to parse as JSON
        import json
        try:
            result = json.loads(completion.choices[0].message.content)
            print(f"✅ Valid JSON: {result}")
        except:
            print("⚠️  Response is not valid JSON (but API call succeeded)")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


if __name__ == "__main__":
    print("\n🔍 Testing Azure AI Foundry Endpoints\n")
    
    results = {
        "GPT-5.1": test_gpt51_endpoint(),
        "DeepSeek": test_deepseek_endpoint(),
        "DeepSeek JSON": test_deepseek_json_mode(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:20} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Endpoints are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    exit(0 if all_passed else 1)

