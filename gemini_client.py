import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


class GeminiConfigurationError(RuntimeError):
    """
    Raised when Gemini has not been configured correctly.
    """
    pass


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """
    Create and cache the Gemini API client.
    """

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is not configured. "
            "Copy .env.example to .env and add "
            "your Google AI Studio API key."
        )

    return genai.Client(
        api_key=api_key
    )


def generate_text(
    prompt: str,
    *,
    system_instruction: str | None = None,
    temperature: float = 0.4,
    max_output_tokens: int = 2048,
    response_schema=None,
):
    """
    Send a prompt to Gemini and return generated text.
    """

    client = get_client()

    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    if system_instruction:
        config_kwargs[
            "system_instruction"
        ] = system_instruction

    if response_schema is not None:
        config_kwargs[
            "response_mime_type"
        ] = "application/json"

        config_kwargs[
            "response_schema"
        ] = response_schema

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            **config_kwargs
        ),
    )

    text = response.text

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return text.strip()