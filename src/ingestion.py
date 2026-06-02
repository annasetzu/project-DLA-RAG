from pathlib import Path
from typing import List

from llama_index.core import Document
from pypdf import PdfReader


def load_txt_file(file_path: Path) -> Document:
    """Load a TXT file and return a LlamaIndex Document."""
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return Document(
        text=text,
        metadata={"source": str(file_path), "file_name": file_path.name},
    )


def load_pdf_file(file_path: Path) -> Document:
    """Load a PDF file and return a LlamaIndex Document."""
    reader = PdfReader(str(file_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(f"\n--- Page {page_number} ---\n{page_text}")

    text = "\n".join(pages)

    return Document(
        text=text,
        metadata={"source": str(file_path), "file_name": file_path.name},
    )


def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Load all PDF and TXT files from a folder."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"La cartella {folder_path} non esiste.")

    documents: List[Document] = []

    for file_path in folder.iterdir():
        if file_path.suffix.lower() == ".txt":
            documents.append(load_txt_file(file_path))
        elif file_path.suffix.lower() == ".pdf":
            documents.append(load_pdf_file(file_path))

    if not documents:
        raise ValueError(
            f"Nessun documento PDF/TXT trovato nella cartella {folder_path}."
        )

    return documents
