from gemini_client import generate_text


def answer_question(question: str) -> str:

    prompt = f"""
Answer the student's question accurately and clearly.

Question:
{question}

Requirements:

- Give the direct answer first.
- Explain the reasoning or important context briefly.
- Use simple language suitable for a learner.
- If the question is ambiguous, state the assumption you used.
- Do not invent citations or sources.
"""

    return generate_text(
        prompt,
        system_instruction=(
            "You are EduGenie, a careful educational "
            "question-answering assistant."
        ),
        temperature=0.3,
        max_output_tokens=1200,
    )