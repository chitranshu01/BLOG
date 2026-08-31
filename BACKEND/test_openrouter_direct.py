import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY not found")

url = "https://openrouter.ai/api/v1/chat/completions"

payload = {
    "model": "qwen/qwen3-30b-a3b-instruct-2507",
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly: OpenRouter works"
        }
    ],
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

print("Testing direct OpenRouter request...")

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )

    print("Status:", response.status_code)
    print("Response:")
    print(response.text)

except Exception as exc:
    print("FAILED")
    print(type(exc).__name__)
    print(exc)