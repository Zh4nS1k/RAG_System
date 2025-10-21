import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from api import openai_key

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - handled at runtime
    PdfReader = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "laws"
PERSIST_DIR = BASE_DIR / "chroma_db" / "kz_legal_codes"
COLLECTION_NAME = "kz_legal_codes"
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}
MIN_CHARS_PER_PAGE = 50


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def load_text_document(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if len(text) < MIN_CHARS_PER_PAGE:
        return []

    metadata = {
        "document_name": path.name,
        "source_path": _relative_path(path),
        "page_number": 1,
        "source_type": path.suffix.lower().lstrip(".") or "text",
    }
    return [Document(page_content=text, metadata=metadata)]


def load_pdf_document(path: Path) -> List[Document]:
    if PdfReader is None:
        raise RuntimeError(
            "pypdf is required to read PDF files. Install it with `pip install pypdf`."
        )

    reader = PdfReader(str(path))
    documents: List[Document] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if len(text) < MIN_CHARS_PER_PAGE:
            continue
        metadata = {
            "document_name": path.name,
            "source_path": _relative_path(path),
            "page_number": page_index,
            "source_type": "pdf",
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def load_legal_documents() -> List[Document]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents: List[Document] = []

    if not any(DATA_DIR.iterdir()):
        print(f"⚠️ {DATA_DIR} is empty. Add legal documents (PDF/TXT) and rerun the script.")
        return documents

    for file_path in sorted(DATA_DIR.rglob("*")):
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                loaded = load_pdf_document(file_path)
            elif suffix in SUPPORTED_TEXT_EXTENSIONS:
                loaded = load_text_document(file_path)
            else:
                print(f"⚠️ Skipping unsupported file: {file_path.name}")
                continue
        except Exception as exc:  # pragma: no cover - runtime feedback
            print(f"❌ Failed to load {file_path.name}: {exc}")
            continue

        if not loaded:
            print(f"⚠️ No textual content extracted from {file_path.name}")
            continue

        documents.extend(loaded)
        print(f"📄 {file_path.name}: added {len(loaded)} page(s)")

    return documents


def build_vector_store() -> None:
    documents = load_legal_documents()
    if not documents:
        print("❌ No documents were loaded. Nothing to index.")
        return

    print(f"\n📊 Summary: {len(documents)} document page(s) ready for chunking")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_key,
        chunk_size=80,
    )

    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
        print("🗑️ Cleared existing database")
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )

    print(f"✅ Successfully stored {len(chunks)} vectors")
    print(f"💾 DB path: {PERSIST_DIR}")

    try:
        sanity = vector_store.similarity_search("административное правонарушение", k=3)
        print("🔎 Sanity search (top documents):", [doc.metadata.get("document_name") for doc in sanity])
    except Exception as exc:  # pragma: no cover - runtime feedback
        print("⚠️ Sanity search failed:", exc)


if __name__ == "__main__":
    build_vector_store()
