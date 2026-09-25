from pydantic import BaseModel, Field

from gemini_client import generate_text


class QuizQuestion(BaseModel):
    question: str

    options: list[str] = Field(
        min_length=4,
        max_length=4
    )

    correct_answer: str

    explanation: str


class QuizResponse(BaseModel):
    questions: list[QuizQuestion] = Field(
        min_length=3,
        max_length=3
    )


def generate_quiz(
    source_text: str
) -> dict:

    prompt = f"""
Create exactly three multiple-choice questions
from the learning material below.

Learning material:
{source_text}

Rules:

- Exactly 3 questions.
- Exactly 4 options per question.
- Only one option is correct.
- correct_answer must exactly match one option.
- Questions must test understanding.
- Avoid obscure wording.
- Include a short explanation for the correct answer.
"""

    raw = generate_text(
        prompt,
        system_instruction=(
            "You generate educational quizzes and must "
            "follow the supplied JSON schema."
        ),
        temperature=0.5,
        max_output_tokens=2200,
        response_schema=QuizResponse,
    )

    quiz = QuizResponse.model_validate_json(
        raw
    )

    return {
        "questions": [
            question.model_dump()
            for question in quiz.questions
        ]
    }