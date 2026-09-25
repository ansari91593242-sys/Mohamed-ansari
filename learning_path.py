from pydantic import BaseModel, Field

from gemini_client import generate_text


class LearningStep(BaseModel):

    stage: str

    topics: list[str]

    suggested_time: str

    practice: list[str]

    resources: list[str]


class LearningPath(BaseModel):

    topic: str

    learner_level: str

    goal: str

    steps: list[LearningStep] = Field(
        min_length=3,
        max_length=5
    )


def get_learning_recommendations(
    topic: str
) -> dict:

    prompt = f"""
Build a structured learning path for this topic:

{topic}

Create a progression from beginner to advanced.

Include:

- The learner level you infer.
- A practical learning goal.
- 3 to 5 stages.
- Topics in each stage.
- Approximate study time.
- Practice activities.
- Resource suggestions.

Resource suggestions should be resource types
or well-known learning sources, not fabricated URLs.
"""

    raw = generate_text(
        prompt,
        system_instruction=(
            "You are EduGenie, a curriculum-planning "
            "assistant. Adapt the learning sequence "
            "to the apparent starting level."
        ),
        temperature=0.45,
        max_output_tokens=2600,
        response_schema=LearningPath,
    )

    path = LearningPath.model_validate_json(
        raw
    )

    return path.model_dump()