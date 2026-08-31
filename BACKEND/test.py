import os
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv()


class RouterTest(BaseModel):
    needs_research: bool = Field(description="Whether web research is needed")
    mode: str = Field(description="closed_book, hybrid, or open_book")
    reason: str = Field(description="Short explanation")


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY was not found. "
            "Check backend/.env and make sure load_dotenv() can find it."
        )

    print("1. OPENROUTER_API_KEY: FOUND")

    llm = ChatOpenRouter(
        model="qwen/qwen3-30b-a3b-instruct-2507",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        timeout=60,
        max_retries=1,
    )

    # ------------------------------------------------------------
    # Test 1: normal ChatOpenRouter request
    # ------------------------------------------------------------
    print("\n2. Testing normal ChatOpenRouter request...")

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a test assistant. Reply very briefly."
                ),
                HumanMessage(
                    content="Reply with exactly: OpenRouter ChatOpenRouter works"
                ),
            ]
        )

        print("SUCCESS")
        print("Response:", response.content)

    except Exception as exc:
        print("FAILED")
        print("Error type:", type(exc).__name__)
        print("Error:", exc)
        return

    # ------------------------------------------------------------
    # Test 2: structured output
    # This mirrors what your BlogForge router does.
    # ------------------------------------------------------------
    print("\n3. Testing structured output...")

    try:
        structured_llm = llm.with_structured_output(RouterTest)

        result = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a routing test. "
                        "For the topic below, decide whether web research "
                        "is needed. Return only the requested structured fields."
                    )
                ),
                HumanMessage(
                    content=(
                        "Topic: React.js components and props\n"
                        "As-of date: 2026-08-18"
                    )
                ),
            ]
        )

        print("SUCCESS")
        print("Structured result:")
        print(result.model_dump())

    except Exception as exc:
        print("FAILED")
        print("Error type:", type(exc).__name__)
        print("Error:", exc)
        return

    print("\nALL TESTS PASSED")
    print("ChatOpenRouter is responding correctly.")


if __name__ == "__main__":
    main()