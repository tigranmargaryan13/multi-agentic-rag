"""
Multimodal ingestion: pymupdf text + parallel GPT-4 Vision for image pages → FAISS.
Run once: python ingest.py   |   Re-ingest: delete faiss_index/ and re-run.
"""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import BaseLLMSettings

settings = BaseLLMSettings()

SOURCE_FOLDERS = {
    "esg": "ESG_documents",
    "ifrs": "IFRS_documents",
    "competitor": "annual_reports_of_comperitors",
}

_VISION_PROMPT = (
    "Describe all visual content on this document page in detail: "
    "tables (with all data), charts, diagrams, flowcharts, figures. "
    "Skip decorative elements. Be thorough — used for semantic search."
)


def _table_to_markdown(table_data: list) -> str | None:
    """Convert extracted table data to markdown. Returns None if the table is mostly empty."""
    if not table_data:
        return None
    total = sum(len(row) for row in table_data)
    if total == 0:
        return None
    empty = sum(1 for row in table_data for cell in row if not cell or not str(cell).strip())
    if empty / total > 0.7:   # >70% empty cells → likely a false-positive
        return None
    lines = []
    for i, row in enumerate(table_data):
        cells = [str(c or "").strip().replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(row)) + " |")
    return "\n".join(lines)


def _describe_page(pdf_path: Path, page_index: int, llm: ChatOpenAI) -> str:
    with fitz.open(str(pdf_path)) as pdf:
        pix = pdf[page_index].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": _VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])
    return llm.invoke([msg]).content


def load_pdf(pdf_path: Path, source_type: str, llm: ChatOpenAI, use_vision: bool) -> list[Document]:
    pdf = fitz.open(str(pdf_path))
    pages: list[Document] = []

    # Parallel vision for all image-bearing pages
    vision: dict[int, str] = {}
    if use_vision:
        image_pages = [i for i in range(len(pdf)) if pdf[i].get_images()]
        print(f"    {len(image_pages)} image pages — running vision in parallel …")
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_describe_page, pdf_path, i, llm): i for i in image_pages}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    vision[i] = fut.result()
                except Exception as e:
                    vision[i] = f"[vision error: {e}]"

    for i, page in enumerate(pdf):
        text = page.get_text("text").strip()

        # Structured tables (markdown format, quality-filtered)
        tables_md = []
        try:
            for tbl in page.find_tables():
                md = _table_to_markdown(tbl.extract())
                if md:
                    tables_md.append(md)
        except Exception:
            pass
        if tables_md:
            text = f"{text}\n\n[TABLE]\n" + "\n\n".join(tables_md) + "\n[/TABLE]"

        # Vision description for image-bearing pages
        if i in vision:
            text = f"{text}\n\n[Visual Content]\n{vision[i]}".strip()

        if text:
            pages.append(Document(
                page_content=text,
                metadata={"source_type": source_type, "filename": pdf_path.name, "page": i + 1},
            ))
    pdf.close()
    return pages


def ingest(use_vision: bool = True) -> None:
    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0, api_key=settings.OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    base = Path(settings.DOCUMENTS_BASE_DIR)

    for source, folder_name in SOURCE_FOLDERS.items():
        folder = base / folder_name
        if not folder.exists():
            continue
        chunks = []
        for pdf_file in sorted(folder.glob("*.pdf")):
            print(f"  {pdf_file.name}")
            pages = load_pdf(pdf_file, source, llm, use_vision)
            chunks.extend(splitter.split_documents(pages))
            print(f"    → {len(pages)} pages, {len(chunks)} chunks so far")

        if not chunks:
            continue
        index_path = Path(settings.FAISS_INDEX_DIR) / source
        index_path.mkdir(parents=True, exist_ok=True)
        FAISS.from_documents(chunks, embeddings).save_local(str(index_path))
        print(f"  ✓ {source} saved ({len(chunks)} chunks)\n")


if __name__ == "__main__":
    ingest()
