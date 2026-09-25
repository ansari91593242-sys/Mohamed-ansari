from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from qna import answer_question
from explanation_module import explain_topic
from quiz_module import generate_quiz
from summary_module import summarize_text
from learning_path import get_learning_recommendations


app = FastAPI(
    title="EduGenie",
    description="Google Gemini Powered Learning Assistant",
    version="1.0.0",
)


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

templates = Jinja2Templates(directory="templates")


# -----------------------------
# Request Models
# -----------------------------

class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=20000
    )


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=10000
    )


# -----------------------------
# Frontend
# -----------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "EduGenie",
    }


# -----------------------------
# Question Answering
# -----------------------------

@app.post("/qa")
def qa(payload: QuestionRequest):
    return {
        "result": answer_question(payload.question)
    }


# -----------------------------
# Concept Explanation
# -----------------------------

@app.post("/explain")
def explain(payload: TextRequest):
    return {
        "result": explain_topic(payload.text)
    }


# -----------------------------
# Quiz Generation
# -----------------------------

@app.post("/quiz")
def quiz(payload: TextRequest):
    return {
        "result": generate_quiz(payload.text)
    }


# -----------------------------
# Summarization
# -----------------------------

@app.post("/summarize")
def summarize(payload: TextRequest):
    return {
        "result": summarize_text(payload.text)
    }


# -----------------------------
# Learning Path
# -----------------------------

@app.post("/learn/recommendations")
def learn(payload: TextRequest):
    return {
        "result": get_learning_recommendations(payload.text)
    }