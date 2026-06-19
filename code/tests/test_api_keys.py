#!/usr/bin/env python3
"""
code/tests/test_api_keys.py
───────────────────────────
Evaluation script to check if Google Gemini (AI Studio/Vertex AI)
and AIML API credentials are working and connected correctly.

Usage:
    python tests/test_api_keys.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to sys.path to allow imports from code/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load .env configuration
dotenv_code = Path(__file__).parent.parent / ".env"
dotenv_root = Path(__file__).parent.parent.parent / ".env"
if dotenv_code.exists():
    print(f"[*] Found .env at: {dotenv_code}")
    load_dotenv(dotenv_code)
elif dotenv_root.exists():
    print(f"[*] Found .env at: {dotenv_root}")
    load_dotenv(dotenv_root)
else:
    print("[!] Warning: No .env file found. Using existing environment variables.")

import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel as VertexGenerativeModel
from openai import OpenAI


def test_gemini_studio_api():
    """Test standard Google AI Studio Gemini API access."""
    print("\n" + "=" * 50)
    print("1. Testing Google AI Studio (google-generativeai)...")
    print("=" * 50)
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        print("[~] Skip: No GEMINI_API_KEY, GOOGLE_API_KEY, or API_KEY found in environment.")
        return False

    print(f"[*] Configuring google-generativeai with API key: {api_key[:6]}...{api_key[-4:] if len(api_key) > 10 else ''}")
    try:
        genai.configure(api_key=api_key)
        
        # We use a simple model for testing connection
        model_name = "gemini-2.5-flash"
        print(f"[*] Sending test request to: {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Respond with exactly the word: SUCCESS")
        
        output = response.text.strip()
        print(f"[+] Response received: '{output}'")
        if "SUCCESS" in output.upper():
            print("[✓] Google AI Studio Gemini API is WORKING!")
            return True
        else:
            print("[!] Warning: Response received but does not match expected output.")
            return True
    except Exception as e:
        print(f"[✗] Google AI Studio Gemini API test FAILED.")
        print(f"    Error detail: {e}")
        return False


def test_vertex_ai():
    """Test Google Cloud Vertex AI access."""
    print("\n" + "=" * 50)
    print("2. Testing GCP Vertex AI...")
    print("=" * 50)
    
    project_id = os.environ.get("GCP_PROJECT_ID")
    region = os.environ.get("GCP_REGION", "us-central1")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Clean up model name in case it uses AI Studio format
    if "/" not in model_name and not model_name.startswith("publishers/"):
        vertex_model_name = model_name
    else:
        vertex_model_name = "gemini-2.5-flash" # fallback to safe Vertex format
        
    print(f"[*] GCP Project ID : {project_id}")
    print(f"[*] GCP Region     : {region}")
    print(f"[*] Target Model   : {vertex_model_name}")

    if not project_id or project_id == "your-gcp-project-id":
        print("[~] Skip: GCP_PROJECT_ID is not configured in .env.")
        print("    If you want to use Vertex AI, please set GCP_PROJECT_ID and run:")
        print("    'gcloud auth application-default login'")
        return False

    try:
        print("[*] Initializing Vertex AI SDK...")
        vertexai.init(project=project_id, location=region)
        
        print(f"[*] Sending test request to Vertex AI Gemini model '{vertex_model_name}'...")
        model = VertexGenerativeModel(vertex_model_name)
        response = model.generate_content("Respond with exactly the word: SUCCESS")
        
        output = response.text.strip()
        print(f"[+] Response received: '{output}'")
        if "SUCCESS" in output.upper():
            print("[✓] GCP Vertex AI is WORKING!")
            return True
        else:
            print("[!] Warning: Response received but does not match expected output.")
            return True
    except Exception as e:
        print(f"[✗] GCP Vertex AI test FAILED.")
        print(f"    Error detail: {e}")
        print("\n    Troubleshooting Vertex AI authentication:")
        print("    1. Ensure you have run: gcloud auth application-default login")
        print("    2. Make sure your GCP project has the Vertex AI API enabled.")
        print("    3. Check if your account has 'Vertex AI User' role on the project.")
        return False


def test_aiml_api():
    """Test AIML API (Qwen Escalation) access."""
    print("\n" + "=" * 50)
    print("3. Testing AIML API (Qwen Escalation)...")
    print("=" * 50)
    
    api_key = os.environ.get("AIML_API_KEY")
    base_url = os.environ.get("AIML_API_BASE_URL", "https://api.aimlapi.com/v1")
    model_name = os.environ.get("ESCALATION_MODEL", "alibaba/qwen3-vl-32b-instruct")
    
    print(f"[*] AIML Base URL  : {base_url}")
    print(f"[*] Target Model   : {model_name}")
    
    if not api_key or api_key == "your-aiml-api-key":
        print("[~] Skip: AIML_API_KEY is not configured in .env.")
        return False

    print(f"[*] Configuring OpenAI client pointing to AIML API...")
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"[*] Sending test chat completion to {model_name}...")
        
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Respond with exactly the word: SUCCESS"}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        output = chat_completion.choices[0].message.content.strip()
        print(f"[+] Response received: '{output}'")
        if "SUCCESS" in output.upper():
            print("[✓] AIML API is WORKING!")
            return True
        else:
            print("[!] Warning: Response received but does not match expected output.")
            return True
    except Exception as e:
        print(f"[✗] AIML API test FAILED.")
        print(f"    Error detail: {e}")
        print("\n    Troubleshooting AIML API:")
        print("    1. Ensure AIML_API_KEY is correct and active in your dashboard.")
        print("    2. Check if the model name is correct and supported by your subscription.")
        return False


def main():
    print("=" * 70)
    print("HackerRank Orchestrate — API Key Verification Tool")
    print("=" * 70)
    
    # Run tests
    studio_ok = test_gemini_studio_api()
    vertex_ok = test_vertex_ai()
    aiml_ok = test_aiml_api()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"- Google AI Studio (API Key) : {'[PASS]' if studio_ok else '[SKIP/FAIL]'}")
    print(f"- GCP Vertex AI (ADC Auth)   : {'[PASS]' if vertex_ok else '[SKIP/FAIL]'}")
    print(f"- AIML API (Qwen Model)      : {'[PASS]' if aiml_ok else '[SKIP/FAIL]'}")
    print("=" * 70)
    
    if studio_ok or vertex_ok or aiml_ok:
        print("[✓] At least one API channel is working. You are ready to run tests!")
    else:
        print("[!] Warning: No API credentials could be verified successfully.")
        print("    Please set up your .env file with active credentials before running the pipeline.")


if __name__ == "__main__":
    main()
