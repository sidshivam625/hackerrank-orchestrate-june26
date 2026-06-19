import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json

load_dotenv(os.path.join("code", ".env"))

async def main():
    api_key = os.getenv("AIML_API_KEY")
    if not api_key:
        print("AIML_API_KEY not found in environment.")
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.aimlapi.com/v1"
    )

    with open("code/prompts/escalation_prompt.txt") as f:
        prompt_tmpl = f.read()

    with open("code/prompts/system_prompt.txt") as f:
        sys_prompt = f.read()

    prompt = prompt_tmpl.format(
        user_claim="Customer: The bumper is cracked.",
        claim_object="car",
        claim_context="Object: car, User says: The bumper is cracked.",
        primary_result="not_enough_information",
        primary_flags="none",
        evidence_requirement="- front_bumper: The claimed car panel or bumper should be visible."
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    try:
        response = await client.chat.completions.create(
            model="alibaba/qwen3-vl-32b-instruct",
            messages=messages,
            max_tokens=8192,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        print("=== RAW RESPONSE ===")
        raw_text = response.choices[0].message.content
        print(repr(raw_text))
        print("====================")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error repr: {repr(e)}")

if __name__ == "__main__":
    asyncio.run(main())
