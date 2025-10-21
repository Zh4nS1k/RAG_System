from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from api import openai_key

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db" / "kz_legal_codes"
COLLECTION_NAME = "kz_legal_codes"
DEFAULT_TOP_K = 5

_PROMPT = ChatPromptTemplate.from_template(
    """Ты — юридический ассистент. Отвечай только на основе переданного контекста из нормативных актов РК.

Правила:
1. Если информации недостаточно, ответь "Я не знаю".
2. Не выдумывай факты и не используй внешние знания.
3. Отвечай кратко и на английском языке.

Контекст:
{context}

Вопрос: {question}

Ответ:"""
)

_embeddings: OpenAIEmbeddings | None = None
_vector_store: Chroma | None = None
_llm: ChatOpenAI | None = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_key,
            chunk_size=80,
        )
    return _embeddings


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        if not PERSIST_DIR.exists():
            raise FileNotFoundError(
                f"Vector store directory {PERSIST_DIR} is missing. Run rag_data.py to build it."
            )
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(PERSIST_DIR),
        )
    return _vector_store


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0,
        )
    return _llm


def _format_context(chunks: List[Document]) -> str:
    formatted_parts: List[str] = []
    for chunk in chunks:
        name = chunk.metadata.get("document_name", "Неизвестный документ")
        page = chunk.metadata.get("page_number")
        page_info = f", страница {page}" if page else ""
        formatted_parts.append(f"[{name}{page_info}]\n{chunk.page_content}")
    return "\n\n".join(formatted_parts)


def query_rag(question: str, top_k: int = DEFAULT_TOP_K) -> Tuple[str, List[Document]]:
    vector_store = get_vector_store()
    retrieved = vector_store.similarity_search(question, k=top_k)

    if not retrieved:
        return "Я не знаю", []

    context = _format_context(retrieved)
    message = _PROMPT.invoke({"context": context, "question": question})
    answer = get_llm().invoke(message)
    return answer.content.strip(), retrieved


def collect_chunk_metadata(chunks: List[Document]) -> List[Dict[str, object]]:
    seen = set()
    structured: List[Dict[str, object]] = []

    for chunk in chunks:
        name = chunk.metadata.get("document_name")
        page = chunk.metadata.get("page_number")
        key = (name, page)
        if key in seen:
            continue
        seen.add(key)
        structured.append(
            {
                "document_name": name,
                "page_number": page,
            }
        )
    return structured


def summarise_hits(chunks: List[Document]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for chunk in chunks:
        name = chunk.metadata.get("document_name", "Unknown")
        counts[name] += 1
    return dict(counts)
