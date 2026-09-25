from gemini_client import generate_text


def summarize_text(text: str) -> str:

    prompt = f"""
Summarize the following educational passage.

Passage:
{text}

Requirements:

- Preserve the central facts.
- Preserve important relationships.
- Remove repetition and minor details.
- Use short paragraphs or bullet points where helpful.
- Do not introduce facts that are absent from the passage.
- Make the result useful for revision.
"""

    return generate_text(
        prompt,
        system_instruction=(
            "You are EduGenie, an educational "
            "summarization assistant."
        ),
        temperature=0.3,
        max_output_tokens=1600,
    )