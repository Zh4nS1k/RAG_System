from __future__ import annotations

from typing import Iterable

from rag_pipeline import collect_chunk_metadata, query_rag, summarise_hits


def ask_question(question: str, k: int = 5) -> None:
    print("🔍 Вопрос:", question)
    try:
        answer, chunks = query_rag(question, top_k=k)
    except FileNotFoundError as exc:
        print(f'❌ {exc}')
        return

    if not chunks:
        print("⚠️ Подходящие документы не найдены. Попробуйте переформулировать запрос.")
        return

    hits = summarise_hits(chunks)
    print("📚 Найдены совпадения в документах:")
    for name, count in hits.items():
        print(f" • {name}: {count} фрагмент(ов)")

    print("\n🤖 Ответ:")
    print(answer)

    print("\n🔎 Источники:")
    for chunk in collect_chunk_metadata(chunks):
        name = chunk.get("document_name") or "Неизвестный документ"
        page = chunk.get("page_number")
        page_info = f", страница {page}" if page else ""
        print(f" • {name}{page_info}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    questions = [
        "Какие существуют виды договоров в гражданском праве Республики Казахстан?",
        "Какие налоги существуют в Казахстане?",
        "Какие права имеют работники по Трудовому кодексу РК?",
        "Что считается административным правонарушением в Казахстане?",
    ]

    for q in questions:
        ask_question(q, k=5)
