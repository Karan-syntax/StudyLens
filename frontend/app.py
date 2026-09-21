"""Streamlit UI for the multi-source RAG learning assistant.

This app is independent from the original command-line backend in main.py.
Run with: streamlit run app.py
"""

import hashlib
import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag_sources import (
    SUPPORTED_FILE_TYPES,
    load_uploaded_file,
    load_webpage,
    load_youtube_transcript,
    source_label,
)


load_dotenv()
st.set_page_config(page_title="StudyLens — PDF & Video Learning Assistant", page_icon="🔍", layout="wide")


def apply_style() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

          html, body, [class*="css"] {
              font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
              letter-spacing: -0.011em;
          }

          .block-container {
              max-width: 1140px;
              padding-top: 1.8rem;
              padding-bottom: 7rem;
          }

          /* Ambient Background */
          .stApp {
              background-image: 
                  radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
                  radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.06) 0px, transparent 50%),
                  radial-gradient(at 50% 100%, rgba(59, 130, 246, 0.05) 0px, transparent 50%);
              background-attachment: fixed;
          }

          /* Modern Hero Banner */
          .hero-container {
              position: relative;
              border-radius: 24px;
              padding: 2.8rem 3rem;
              margin: 0 0 2rem 0;
              background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 45%, #2e1065 100%);
              border: 1px solid rgba(255, 255, 255, 0.12);
              box-shadow: 
                  0 20px 40px -15px rgba(15, 23, 42, 0.6),
                  inset 0 1px 0 rgba(255, 255, 255, 0.15);
              overflow: hidden;
          }
          .hero-container::before {
              content: "";
              position: absolute;
              top: -80px;
              right: -50px;
              width: 320px;
              height: 320px;
              background: radial-gradient(circle, rgba(139, 92, 246, 0.45) 0%, rgba(139, 92, 246, 0) 70%);
              border-radius: 50%;
              filter: blur(40px);
              pointer-events: none;
          }
          .hero-container::after {
              content: "";
              position: absolute;
              bottom: -60px;
              right: 180px;
              width: 240px;
              height: 240px;
              background: radial-gradient(circle, rgba(236, 72, 153, 0.3) 0%, rgba(236, 72, 153, 0) 70%);
              border-radius: 50%;
              filter: blur(40px);
              pointer-events: none;
          }
          .hero-content {
              position: relative;
              z-index: 2;
              max-width: 740px;
          }
          .hero-badge {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              background: rgba(255, 255, 255, 0.08);
              backdrop-filter: blur(12px);
              -webkit-backdrop-filter: blur(12px);
              border: 1px solid rgba(255, 255, 255, 0.15);
              padding: 0.35rem 0.9rem;
              border-radius: 9999px;
              font-size: 0.74rem;
              font-weight: 700;
              text-transform: uppercase;
              letter-spacing: 0.1em;
              color: #a5b4fc;
              margin-bottom: 1rem;
              box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          }
          .hero-badge .badge-dot {
              width: 7px;
              height: 7px;
              border-radius: 50%;
              background: #34d399;
              box-shadow: 0 0 10px #34d399;
              animation: pulse-dot 2s infinite ease-in-out;
          }
          @keyframes pulse-dot {
              0%, 100% { opacity: 1; transform: scale(1); }
              50% { opacity: 0.4; transform: scale(0.85); }
          }
          .hero-title {
              color: #ffffff !important;
              font-size: clamp(2.2rem, 3.8vw, 3.1rem) !important;
              font-weight: 800 !important;
              line-height: 1.12 !important;
              letter-spacing: -0.04em !important;
              margin: 0 0 0.85rem 0 !important;
              background: linear-gradient(180deg, #ffffff 30%, #cbd5e1 100%);
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
          }
          .hero-desc {
              color: #cbd5e1;
              font-size: 1.05rem;
              line-height: 1.65;
              margin: 0 0 1.25rem 0;
              font-weight: 400;
          }
          .hero-features {
              display: flex;
              flex-wrap: wrap;
              gap: 10px;
              margin-top: 0.8rem;
          }
          .feature-pill {
              display: inline-flex;
              align-items: center;
              gap: 6px;
              background: rgba(255, 255, 255, 0.08);
              border: 1px solid rgba(255, 255, 255, 0.12);
              padding: 0.3rem 0.8rem;
              border-radius: 8px;
              font-size: 0.82rem;
              color: #e2e8f0;
              font-weight: 500;
          }

          /* Sidebar Styling */
          [data-testid="stSidebar"] {
              border-right: 1px solid rgba(148, 163, 184, 0.15) !important;
              background: linear-gradient(180deg, rgba(15, 23, 42, 0.02) 0%, rgba(15, 23, 42, 0.06) 100%);
          }
          .sidebar-brand-card {
              display: flex;
              align-items: center;
              gap: 12px;
              padding: 0.5rem 0.2rem 0.2rem;
              margin-bottom: 0.25rem;
          }
          .sidebar-logo-icon {
              width: 42px;
              height: 42px;
              border-radius: 12px;
              background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
              display: flex;
              align-items: center;
              justify-content: center;
              font-size: 1.35rem;
              box-shadow: 0 8px 18px -4px rgba(99, 102, 241, 0.4);
          }
          .sidebar-title {
              font-size: 1.32rem;
              font-weight: 800;
              letter-spacing: -0.035em;
              line-height: 1.2;
          }
          .section-tag {
              display: flex;
              align-items: center;
              gap: 8px;
              font-size: 0.74rem;
              font-weight: 700;
              text-transform: uppercase;
              letter-spacing: 0.09em;
              color: #6366f1;
              margin: 0.9rem 0 0.4rem;
          }

          /* Active Source Status Card */
          .status-card {
              background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.06) 100%);
              border: 1px solid rgba(99, 102, 241, 0.25);
              border-radius: 16px;
              padding: 1.1rem;
              margin: 1rem 0;
              box-shadow: 0 8px 24px -6px rgba(99, 102, 241, 0.12);
              position: relative;
              overflow: hidden;
          }
          .status-card::before {
              content: "";
              position: absolute;
              left: 0;
              top: 0;
              height: 100%;
              width: 4px;
              background: linear-gradient(180deg, #6366f1, #a855f7);
          }
          .status-badge {
              display: inline-flex;
              align-items: center;
              gap: 6px;
              font-size: 0.72rem;
              font-weight: 700;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              color: #10b981;
              margin-bottom: 0.4rem;
          }
          .status-dot {
              width: 6px;
              height: 6px;
              background: #10b981;
              border-radius: 50%;
              box-shadow: 0 0 8px #10b981;
          }
          .status-name {
              display: block;
              font-size: 0.95rem;
              font-weight: 700;
              overflow: hidden;
              text-overflow: ellipsis;
              white-space: nowrap;
              margin-bottom: 0.35rem;
          }
          .status-meta {
              display: flex;
              align-items: center;
              gap: 8px;
              font-size: 0.82rem;
              opacity: 0.78;
              font-weight: 500;
          }
          .status-pill {
              background: rgba(99, 102, 241, 0.12);
              color: #4338ca;
              padding: 0.15rem 0.5rem;
              border-radius: 6px;
              font-family: 'JetBrains Mono', monospace;
              font-size: 0.74rem;
              font-weight: 600;
          }

          /* Empty State Showcase */
          .empty-state-card {
              border: 1.5px dashed rgba(99, 102, 241, 0.35);
              border-radius: 20px;
              padding: 2.5rem 2rem;
              text-align: center;
              background: linear-gradient(180deg, rgba(99, 102, 241, 0.02) 0%, rgba(168, 85, 247, 0.03) 100%);
              margin-top: 1rem;
              transition: all 0.3s ease;
          }
          .empty-state-card:hover {
              border-color: rgba(99, 102, 241, 0.6);
              transform: translateY(-2px);
              box-shadow: 0 12px 30px -10px rgba(99, 102, 241, 0.1);
          }
          .empty-state-icon {
              font-size: 2.8rem;
              margin-bottom: 0.8rem;
              display: inline-block;
              filter: drop-shadow(0 4px 10px rgba(99, 102, 241, 0.3));
          }
          .empty-state-title {
              font-size: 1.25rem;
              font-weight: 700;
              margin-bottom: 0.4rem;
          }
          .empty-state-desc {
              font-size: 0.95rem;
              opacity: 0.75;
              max-width: 500px;
              margin: 0 auto 1.4rem;
              line-height: 1.6;
          }
          .guide-grid {
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
              gap: 14px;
              max-width: 700px;
              margin: 0 auto;
              text-align: left;
          }
          .guide-step {
              background: rgba(255, 255, 255, 0.65);
              border: 1px solid rgba(148, 163, 184, 0.22);
              border-radius: 14px;
              padding: 0.95rem 1.1rem;
              box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
          }
          .guide-num {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              width: 24px;
              height: 24px;
              border-radius: 7px;
              background: #6366f1;
              color: #fff;
              font-size: 0.74rem;
              font-weight: 700;
              margin-bottom: 0.4rem;
          }
          .guide-text {
              font-size: 0.86rem;
              font-weight: 650;
              margin-bottom: 0.15rem;
          }
          .guide-sub {
              font-size: 0.76rem;
              opacity: 0.75;
              line-height: 1.4;
          }

          /* Interactive Control Overrides */
          .stButton > button {
              border-radius: 12px;
              min-height: 2.65rem;
              font-weight: 600;
              letter-spacing: -0.01em;
              transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
              border: 1px solid rgba(0, 0, 0, 0.08);
          }
          .stButton > button:hover {
              transform: translateY(-2px);
              box-shadow: 0 8px 20px -4px rgba(99, 102, 241, 0.35);
          }
          .stButton > button:active {
              transform: translateY(0px);
          }
          .stButton > button[kind="primary"] {
              background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
              border: none !important;
              box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
          }

          /* Input Fields */
          [data-testid="stTextInput"] input, [data-testid="stSelectbox"] > div > div {
              border-radius: 12px !important;
              border: 1px solid rgba(148, 163, 184, 0.28) !important;
              transition: all 0.2s ease;
              box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
          }
          [data-testid="stTextInput"] input:focus, [data-testid="stSelectbox"] > div > div:focus-within {
              border-color: #6366f1 !important;
              box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18) !important;
          }

          /* Chat Messages */
          [data-testid="stChatMessage"] {
              border: 1px solid rgba(148, 163, 184, 0.18);
              border-radius: 20px;
              padding: 0.95rem 1.3rem;
              margin-bottom: 1.1rem;
              box-shadow: 0 4px 18px -4px rgba(15, 23, 42, 0.04);
              backdrop-filter: blur(10px);
              -webkit-backdrop-filter: blur(10px);
              transition: border-color 0.2s ease;
          }
          [data-testid="stChatMessage"]:hover {
              border-color: rgba(99, 102, 241, 0.3);
          }

          /* Chat Input */
          [data-testid="stChatInput"] {
              border-radius: 18px !important;
              border: 1.5px solid rgba(99, 102, 241, 0.35) !important;
              box-shadow: 0 10px 30px -8px rgba(0, 0, 0, 0.08) !important;
              transition: all 0.2s ease;
          }
          [data-testid="stChatInput"]:focus-within {
              border-color: #6366f1 !important;
              box-shadow: 0 10px 35px -6px rgba(99, 102, 241, 0.25) !important;
          }

          /* Custom Source Card Accordion */
          [data-testid="stExpander"] {
              border-radius: 14px !important;
              border: 1px solid rgba(148, 163, 184, 0.22) !important;
              background: rgba(248, 250, 252, 0.6) !important;
              overflow: hidden;
              margin-top: 0.6rem;
          }
          .source-citation-card {
              background: rgba(255, 255, 255, 0.7);
              border: 1px solid rgba(148, 163, 184, 0.2);
              border-radius: 10px;
              padding: 0.8rem 1rem;
              margin-bottom: 0.6rem;
          }
          .source-badge {
              display: inline-flex;
              align-items: center;
              gap: 5px;
              background: rgba(99, 102, 241, 0.12);
              color: #4f46e5;
              padding: 0.2rem 0.6rem;
              border-radius: 6px;
              font-size: 0.76rem;
              font-weight: 700;
              margin-bottom: 0.35rem;
          }

          /* Hide Streamlit Community Cloud Badge, GitHub Fork Button & Footer */
          footer {
              visibility: hidden !important;
              display: none !important;
          }
          header[data-testid="stHeader"] {
              background: transparent !important;
          }
          .viewerBadge_container__r5tak,
          .viewerBadge_link__1SuGQ,
          [data-testid="stStatusWidget"],
          [data-testid="manage-app-button"],
          #MainMenu,
          .stDeployButton,
          [class*="viewerBadge"] {
              display: none !important;
              visibility: hidden !important;
          }

          /* Creator Profile Card in Sidebar */
          .creator-card {
              display: flex;
              align-items: center;
              gap: 12px;
              padding: 0.75rem 0.9rem;
              margin-top: 1.5rem;
              border-radius: 14px;
              background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.05) 100%);
              border: 1px solid rgba(148, 163, 184, 0.2);
              box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
          }
          .creator-avatar {
              width: 42px;
              height: 42px;
              border-radius: 50%;
              object-fit: cover;
              border: 2px solid #6366f1;
              box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
          }
          .creator-info {
              display: flex;
              flex-direction: column;
          }
          .creator-name {
              font-size: 0.88rem;
              font-weight: 700;
              color: inherit;
              line-height: 1.25;
          }
          .creator-role {
              font-size: 0.74rem;
              opacity: 0.7;
              font-weight: 500;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_models():
    key = os.getenv("GOOGLE_API_KEY")
    if not key and hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        key = st.secrets["GOOGLE_API_KEY"]
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to your Streamlit Cloud Secrets or local .env file.")
    return (
        GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=key),
        ChatGoogleGenerativeAI(model="gemini-3.7-flash", google_api_key=key),
    )


def clean_response(content) -> str:
    """Keep Gemini text only; never render response metadata or signatures."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    if isinstance(content, list):
        return "\n".join(filter(None, (clean_response(item) for item in content))).strip()
    return str(content).strip()


def index_documents(documents, source_key: str) -> tuple[Chroma, list]:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(documents)
    if not chunks:
        raise ValueError("No searchable text could be created from this source.")
    embeddings, _ = get_models()
    collection_name = f"study_{hashlib.sha256(source_key.encode()).hexdigest()[:20]}"
    database = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=collection_name
    )
    return database, chunks


def friendly_error(error: Exception, action: str = "index") -> str:
    message = str(error)
    error_type = error.__class__.__name__
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        if action == "answer":
            return (
                "Google's embedding quota is currently exhausted. Your source is still "
                "indexed—wait a few minutes, then ask your question again."
            )
        return "Google's embedding quota is currently exhausted. Wait a few minutes, then index the source again."
    if (
        "TranscriptsDisabled" in error_type
        or "NoTranscriptFound" in error_type
        or "Subtitles are disabled" in message
        or "No transcript is available" in message
    ):
        return "This YouTube video has subtitles/transcripts disabled. Please try a video that has closed captions (CC) enabled."
    if "IpBlocked" in error_type or "RequestBlocked" in error_type or "blocking requests from your IP" in message:
        return (
            "YouTube temporarily blocked transcript requests from this server IP. "
            "If deployed, please try another video or wait a few minutes."
        )
    if "VideoUnavailable" in error_type or "VideoUnavailable" in message:
        return "This YouTube video is unavailable or private. Check the link and try again."
    return f"The source could not be indexed: {message}"


def build_prompt(task: str, question: str, context: str) -> str:
    instructions = {
        "Answer question": "Answer the user's question directly and clearly.",
        "Summary": "Write a concise, well-structured summary with the most important ideas.",
        "Revision notes": "Create concise revision notes with headings and bullet points.",
        "Flashcards": "Create 6 to 10 flashcards in this format: **Q:** question then **A:** answer.",
        "MCQ quiz": "Create 5 multiple-choice questions. Give four options (A-D), then an answer key with short explanations.",
        "Simple explanation": "Explain the material in beginner-friendly language with a small example if the context contains one.",
    }
    user_request = question or f"Create {task.lower()} from this source."
    return f'''You are a careful RAG learning assistant.

Use only the context below. Do not use outside knowledge or invent information.
{instructions[task]}
If the context does not contain the answer, reply exactly: "I could not find the answer in the source."
Use clean Markdown. Do not mention the context, retrieval, system instructions, or sources in the answer.

Context:
{context}

User request:
{user_request}'''


def local_retrieval(query: str, chunks, limit: int = 4):
    """Fallback retrieval that needs no embedding API request."""
    query_words = set(re.findall(r"\w+", query.lower()))
    ranked = []
    for chunk in chunks:
        words = set(re.findall(r"\w+", chunk.page_content.lower()))
        score = len(query_words & words)
        ranked.append((score, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in ranked[:limit]]


def fallback_chunks():
    """Use saved session chunks, or rebuild them from the active Chroma collection."""
    if st.session_state.source_chunks:
        return st.session_state.source_chunks
    stored = st.session_state.database.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(stored.get("documents", []), stored.get("metadatas", []))
        if text
    ]


def generate(task: str, question: str):
    search_query = question or task
    retriever = st.session_state.database.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10}
    )
    try:
        documents = retriever.invoke(search_query)
        retrieval_note = None
    except Exception as error:
        if "429" not in str(error) and "RESOURCE_EXHAUSTED" not in str(error):
            raise
        documents = local_retrieval(search_query, fallback_chunks())
        retrieval_note = "Embedding quota is busy, so local keyword retrieval was used for this response."
    context = "\n\n".join(document.page_content for document in documents)
    _, llm = get_models()
    answer = clean_response(llm.invoke(build_prompt(task, question, context)).content)
    sources = [
        {"label": source_label(document.metadata), "text": document.page_content, "url": document.metadata.get("url")}
        for document in documents
    ]
    return answer or "I could not find the answer in the source.", sources, retrieval_note


def show_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"🔍 View Grounding Sources ({len(sources)} citations retrieved)"):
        for idx, source in enumerate(sources, 1):
            url_link = f" • [Open source link ↗]({source['url']})" if source.get("url") else ""
            st.markdown(
                f"""
                <div class='source-citation-card'>
                    <div class='source-badge'>Citation {idx} • {source['label']}</div>
                    <div style='font-size: 0.85rem; line-height: 1.5; color: var(--text-color); margin-top: 0.35rem;'>
                        {source['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if url_link:
                st.markdown(url_link)


def load_selected_source(kind: str, uploaded_file, url: str):
    if kind == "File upload":
        if not uploaded_file:
            raise ValueError("Choose a file before indexing.")
        return load_uploaded_file(uploaded_file), uploaded_file.name
    if not url.strip():
        raise ValueError("Paste a link before indexing.")
    if kind == "Web page":
        return load_webpage(url.strip()), url.strip()
    return load_youtube_transcript(url.strip()), url.strip()


apply_style()
for key, value in {
    "database": None,
    "messages": [],
    "source_name": None,
    "source_stats": None,
    "source_key": None,
    "source_chunks": [],
}.items():
    st.session_state.setdefault(key, value)

with st.sidebar:
    st.markdown(
        """
        <div class='sidebar-brand-card'>
            <div class='sidebar-logo-icon'>🔍</div>
            <div>
                <div class='sidebar-title'>StudyLens</div>
                <div style='font-size: 0.74rem; opacity: 0.75; font-weight: 500;'>Smart Knowledge & Study Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Turn your documents, articles, and video transcripts into grounded answers & revision tools.")
    st.divider()
    st.markdown("<div class='section-tag'><span>01</span> Choose Knowledge Source</div>", unsafe_allow_html=True)
    source_kind = st.radio("Knowledge source", ["File upload", "Web page", "YouTube video"], label_visibility="collapsed")
    uploaded_file, source_url = None, ""
    if source_kind == "File upload":
        uploaded_file = st.file_uploader(
            "Upload a file", type=SUPPORTED_FILE_TYPES,
            help="PDF, DOCX, PPTX, TXT, CSV, or Markdown",
        )
    elif source_kind == "Web page":
        source_url = st.text_input("Webpage URL", placeholder="https://example.com/article")
    else:
        source_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        st.caption("Public videos with available captions are supported.")

    if st.button("⚡ Index & Process Source", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading, chunking, and indexing your source..."):
                documents, name = load_selected_source(source_kind, uploaded_file, source_url)
                source_key = f"{source_kind}:{name}:{len(documents)}"
                database, chunks = index_documents(documents, source_key)
            st.session_state.database = database
            st.session_state.source_name = name
            st.session_state.source_key = source_key
            st.session_state.source_stats = {"documents": len(documents), "chunks": len(chunks)}
            st.session_state.source_chunks = chunks
            st.session_state.messages = []
            st.success("Source indexed successfully.")
        except Exception as error:
            st.error(friendly_error(error))

    if st.session_state.database is not None:
        stats = st.session_state.source_stats
        st.markdown(
            f"""
            <div class='status-card'>
                <div class='status-badge'>
                    <span class='status-dot'></span> Active Knowledge Source
                </div>
                <span class='status-name' title='{st.session_state.source_name}'>{st.session_state.source_name}</span>
                <div class='status-meta'>
                    <span class='status-pill'>📑 {stats['documents']} sections</span>
                    <span class='status-pill'>🧩 {stats['chunks']} chunks</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("Remove active source", use_container_width=True):
        st.session_state.database = None
        st.session_state.source_name = None
        st.session_state.source_stats = None
        st.session_state.source_chunks = []
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        <div class="creator-card">
            <img class="creator-avatar" src="https://avatars.githubusercontent.com/u/152914104?v=4" alt="Karan Kr Verma" />
            <div class="creator-info">
                <span class="creator-name">Karan Kr Verma</span>
                <span class="creator-role">Creator & Developer</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="hero-container">
        <div class="hero-content">
            <h1 class="hero-title">Turn any lesson into deep understanding.</h1>
            <p class="hero-desc">
                Connect documents, web articles, or YouTube lectures. Chat with grounded accuracy, extract structured revision notes, or generate quizzes instantly.
            </p>
            <div class="hero-features">
                <span class="feature-pill">⚡ Low Latency RAG</span>
                <span class="feature-pill">🎯 Strict Context Grounding</span>
                <span class="feature-pill">📝 Instant Flashcards & Quizzes</span>
                <span class="feature-pill">🔗 Verifiable Citations</span>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if st.session_state.database is None:
    st.markdown(
        """
        <div class='empty-state-card'>
            <div class='empty-state-icon'>💡</div>
            <div class='empty-state-title'>Your workspace is ready</div>
            <div class='empty-state-desc'>
                To begin exploring, choose a knowledge source from the sidebar on your left and click <b>Index & Process Source</b>.
            </div>
            <div class='guide-grid'>
                <div class='guide-step'>
                    <div class='guide-num'>1</div>
                    <div class='guide-text'>Select Source Type</div>
                    <div class='guide-sub'>Upload PDF, DOCX, Markdown, or paste any Web URL or YouTube link.</div>
                </div>
                <div class='guide-step'>
                    <div class='guide-num'>2</div>
                    <div class='guide-text'>Build Vector Store</div>
                    <div class='guide-sub'>Automated chunking and embedding storage powered by ChromaDB.</div>
                </div>
                <div class='guide-step'>
                    <div class='guide-num'>3</div>
                    <div class='guide-text'>Ask & Learn</div>
                    <div class='guide-sub'>Query concepts, create study flashcards, MCQ quizzes, or summaries.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("<div class='section-tag'><span>02</span> Select Study Mode & Action</div>", unsafe_allow_html=True)
    task = st.selectbox(
        "Choose an output mode",
        ["Answer question", "Summary", "Revision notes", "Flashcards", "MCQ quiz", "Simple explanation"],
        help="Select how you want the RAG assistant to process your queries or synthesize content.",
    )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_sources(message.get("sources", []))

    placeholder = "Ask a question about the active source..." if task == "Answer question" else "Optional: add a topic or instruction for this study output..."
    request = st.chat_input(placeholder)
    if request or (task != "Answer question" and st.button(f"Create {task.lower()}", type="primary")):
        effective_request = request or ""
        user_label = effective_request or f"Create {task.lower()}"
        st.session_state.messages.append({"role": "user", "content": user_label})
        with st.chat_message("user"):
            st.markdown(user_label)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant material and preparing your result..."):
                try:
                    answer, sources, retrieval_note = generate(task, effective_request)
                except Exception as error:
                    st.error(friendly_error(error, action="answer"))
                    st.stop()
            st.markdown(answer)
            if retrieval_note:
                st.caption(f"Note: {retrieval_note}")
            show_sources(sources)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
