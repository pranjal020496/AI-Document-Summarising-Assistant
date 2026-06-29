
import hashlib
import io
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

load_dotenv(".env", override=True)

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        groq_api_key = ""

if not groq_api_key:
    st.error(
        "GROQ_API_KEY was not found. "
        "Configure it locally in .env or in Streamlit secrets."
    )
    st.stop()

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)


# --------------------------------------------------
# PDF processing functions
# --------------------------------------------------

def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract text while preserving PDF page numbers."""

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.replace("\x00", " ").strip()

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    return pages


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    """Split text into overlapping sections."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = proposed_end

        if proposed_end < len(text):
            section = text[start:proposed_end]

            paragraph_boundary = section.rfind("\n")
            sentence_boundary = section.rfind(". ")

            best_boundary = max(
                paragraph_boundary,
                sentence_boundary,
            )

            if best_boundary > chunk_size * 0.6:
                end = start + best_boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(pages: list[dict]) -> list[dict]:
    """Create searchable chunks from extracted pages."""

    chunks = []

    for page in pages:
        page_chunks = split_text_into_chunks(page["text"])

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):
            chunks.append(
                {
                    "page": page["page"],
                    "chunk": chunk_number,
                    "text": chunk_text,
                }
            )

    return chunks


def create_search_index(chunks: list[dict]):
    """Create a TF-IDF search index."""

    chunk_texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50_000,
    )

    chunk_matrix = vectorizer.fit_transform(chunk_texts)

    return vectorizer, chunk_matrix


# --------------------------------------------------
# Retrieval functions
# --------------------------------------------------

def retrieve_relevant_chunks(
    question: str,
    chunks: list[dict],
    vectorizer,
    chunk_matrix,
    top_k: int = 5,
) -> list[dict]:
    """Find document chunks most relevant to a question."""

    question = question.strip()

    if not question:
        raise ValueError("The question cannot be empty.")

    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        chunk_matrix,
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        result = chunks[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


def retrieve_with_neighboring_pages(
    question: str,
    pages: list[dict],
    chunks: list[dict],
    vectorizer,
    chunk_matrix,
    top_k: int = 5,
    page_radius: int = 1,
) -> list[dict]:
    """Retrieve relevant chunks and surrounding pages."""

    primary_results = retrieve_relevant_chunks(
        question=question,
        chunks=chunks,
        vectorizer=vectorizer,
        chunk_matrix=chunk_matrix,
        top_k=top_k,
    )

    if not primary_results:
        return []

    anchor_page = primary_results[0]["page"]

    neighboring_pages = set(
        range(
            max(1, anchor_page - page_radius),
            min(len(pages), anchor_page + page_radius) + 1,
        )
    )

    combined_results = []
    seen = set()

    for result in primary_results:
        key = (result["page"], result["chunk"])

        if key not in seen:
            combined_results.append(result.copy())
            seen.add(key)

    for chunk in chunks:
        key = (chunk["page"], chunk["chunk"])

        if chunk["page"] in neighboring_pages and key not in seen:
            neighbor = chunk.copy()
            neighbor["score"] = 0.0

            combined_results.append(neighbor)
            seen.add(key)

    combined_results.sort(
        key=lambda item: (
            item["page"],
            item["chunk"],
        )
    )

    return combined_results


def build_context(results: list[dict]) -> str:
    """Combine retrieved text with page references."""

    context_sections = []

    for result in results:
        context_sections.append(
            f"[Source: PDF page {result['page']}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(context_sections)


# --------------------------------------------------
# AI question-answering function
# --------------------------------------------------

def answer_question(
    question: str,
    pages: list[dict],
    chunks: list[dict],
    vectorizer,
    chunk_matrix,
) -> tuple[str, list[dict]]:
    """Retrieve relevant content and generate a grounded answer."""

    retrieved_results = retrieve_with_neighboring_pages(
        question=question,
        pages=pages,
        chunks=chunks,
        vectorizer=vectorizer,
        chunk_matrix=chunk_matrix,
        top_k=5,
        page_radius=1,
    )

    context = build_context(retrieved_results)

    prompt = f"""
You are a document-analysis assistant.

Answer the user's question using only the supplied document context.

Rules:
1. Do not use outside knowledge.
2. Include all relevant information present in the context.
3. If a list continues across pages, preserve the complete list.
4. Cite claims using PDF page numbers, such as [Page 19].
5. Cite multiple pages when the answer spans multiple pages.
6. If the information is unavailable, say so clearly.
7. Do not invent facts or page references.

DOCUMENT CONTEXT
----------------
{context}

USER QUESTION
-------------
{question}
"""

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt,
    )

    return response.output_text, retrieved_results


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.title("📄 AI Document Assistant")

st.write(
    "Upload a PDF and ask questions about its content. "
    "Answers are generated using relevant sections retrieved "
    "from the uploaded document."
)

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        with st.spinner("Processing the PDF..."):
            try:
                pages = extract_pages(pdf_bytes)
                chunks = create_chunks(pages)

                if not chunks:
                    st.error(
                        "No readable text was found. "
                        "The PDF may contain scanned images."
                    )
                    st.stop()

                vectorizer, chunk_matrix = create_search_index(chunks)

                st.session_state.file_hash = file_hash
                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.vectorizer = vectorizer
                st.session_state.chunk_matrix = chunk_matrix

            except Exception as error:
                st.error(f"Could not process the PDF: {error}")
                st.stop()

    pages = st.session_state.pages
    chunks = st.session_state.chunks
    vectorizer = st.session_state.vectorizer
    chunk_matrix = st.session_state.chunk_matrix

    col1, col2 = st.columns(2)

    with col1:
        st.metric("PDF pages", len(pages))

    with col2:
        st.metric("Searchable chunks", len(chunks))

    question = st.text_area(
        "Ask a question about the document",
        placeholder="What are the main objectives of this thesis?",
    )

    if st.button("Generate answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner("Searching the document and generating an answer..."):
                try:
                    answer, sources = answer_question(
                        question=question,
                        pages=pages,
                        chunks=chunks,
                        vectorizer=vectorizer,
                        chunk_matrix=chunk_matrix,
                    )

                    st.subheader("Answer")
                    st.markdown(answer)

                    with st.expander("Retrieved sources"):
                        displayed_sources = set()

                        for source in sources:
                            source_key = (
                                source["page"],
                                source["chunk"],
                            )

                            if source_key in displayed_sources:
                                continue

                            displayed_sources.add(source_key)

                            st.markdown(
                                f"**Page {source['page']} — "
                                f"Chunk {source['chunk']}**"
                            )

                            st.write(source["text"][:600])

                            if len(source["text"]) > 600:
                                st.write("...")

                            st.divider()

                except Exception as error:
                    st.error(f"Answer generation failed: {error}")

else:
    st.info("Upload a PDF to begin.")
