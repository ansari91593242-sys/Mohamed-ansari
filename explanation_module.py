from gemini_client import generate_text


def explain_topic(topic: str) -> str:

    prompt = f"""
Explain the following topic to a beginner:

{topic}

Use this structure:

1. Simple definition
2. How it works
3. A small example or analogy
4. Key points to remember

Keep it concise, accurate, and easy to understand.
"""

    return generate_text(
        prompt,
        system_instruction=(
            "You are EduGenie. Your job is to simplify "
            "complex academic concepts without losing "
            "important meaning."
        ),
        temperature=0.35,
        max_output_tokens=1400,
    )