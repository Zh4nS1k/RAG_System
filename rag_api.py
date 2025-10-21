from __future__ import annotations

from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_pipeline import collect_chunk_metadata, query_rag

app = FastAPI(title="KZ Legal RAG", version="0.1.0")


class QuestionPayload(BaseModel):
    question_id: int
    question: str
    top_k: Optional[int] = None


class QuestionsBatch(BaseModel):
    questions: List[QuestionPayload]
    default_top_k: Optional[int] = None


class RelevantChunk(BaseModel):
    document_name: Optional[str]
    page_number: Optional[int]


AnswerType = Union[str, int, float]


class QueryResponse(BaseModel):
    question_id: int
    relevant_chunks: List[RelevantChunk]
    answer: AnswerType


def _coerce_answer(raw_answer: str) -> AnswerType:
    stripped = raw_answer.strip()
    if not stripped:
        return ""

    try:
        integer = int(stripped)
        return integer
    except ValueError:
        pass

    try:
        floating = float(stripped.replace(",", "."))
        return floating
    except ValueError:
        pass

    return stripped


@app.post("/query", response_model=List[QueryResponse])
def query_endpoint(payload: QuestionsBatch) -> List[QueryResponse]:
    if not payload.questions:
        raise HTTPException(status_code=400, detail="Payload must contain at least one question")

    responses: List[QueryResponse] = []
    fallback_k = payload.default_top_k or 5

    for item in payload.questions:
        top_k = item.top_k or fallback_k
        try:
            answer, chunks = query_rag(item.question, top_k=top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        relevant_chunks = [RelevantChunk(**chunk) for chunk in collect_chunk_metadata(chunks)]
        responses.append(
            QueryResponse(
                question_id=item.question_id,
                relevant_chunks=relevant_chunks,
                answer=_coerce_answer(answer),
            )
        )

    return responses


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
