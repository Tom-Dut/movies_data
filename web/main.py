import os
import json
import logging
from typing import Literal

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from agent import movie_agent_executor

logger = logging.getLogger(__name__)

app = FastAPI(title="Movie RAG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryInput(BaseModel):
    text: str
    history: list[ConversationMessage] = Field(default_factory=list)


@app.post("/movie-rag-agent")
def query_movie_agent(query: QueryInput):
    chat_history = [
        HumanMessage(content=message.content)
        if message.role == "user"
        else AIMessage(content=message.content)
        for message in query.history
    ]

    def event_stream():
        yield _sse_event({
            "type": "progress",
            "message": "Analyse de votre question…",
        })
        try:
            for chunk in movie_agent_executor.stream({
                "input": query.text,
                "chat_history": chat_history,
            }):
                for action in chunk.get("actions", []):
                    yield _sse_event({
                        "type": "tool_start",
                        "message": _tool_message(action.tool, started=True),
                    })

                for step in chunk.get("steps", []):
                    yield _sse_event({
                        "type": "tool_complete",
                        "message": _tool_message(step.action.tool, started=False),
                    })

                if "output" in chunk:
                    yield _sse_event({
                        "type": "final",
                        "output": chunk["output"],
                    })
        except Exception:
            logger.exception("Error while streaming the movie agent response")
            yield _sse_event({
                "type": "error",
                "message": "Une erreur est survenue pendant le traitement de la demande.",
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _tool_message(tool_name: str, started: bool) -> str:
    tool_labels = {
        "MovieReviewSearch": "les avis de films (Chroma)",
        "MovieDatabaseQA": "la base de films (Neo4j)",
        "UserReviewSimilarity": "la similarité entre utilisateurs",
        "MovieGenreSimilarity": "la similarité entre films",
        "MovieRecommendation": "les recommandations",
    }
    label = tool_labels.get(tool_name, "un outil")
    if started:
        return f"Recherche dans {label}…"
    return f"Recherche terminée dans {label}."