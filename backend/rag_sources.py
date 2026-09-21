"""Source loading helpers for the Streamlit multi-source RAG interface."""

import csv
import io
import re
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pptx import Presentation
from youtube_transcript_api import YouTubeTranscriptApi


SUPPORTED_FILE_TYPES = ["pdf", "docx", "pptx", "txt", "csv", "md"]


def _text_document(text: str, name: str, source_type: str) -> list[Document]:
    text = text.strip()
    if not text:
        raise ValueError("The selected source does not contain readable text.")
    return [Document(page_content=text, metadata={"source": name, "source_type": source_type})]


def load_uploaded_file(uploaded_file) -> list[Document]:
    """Load one supported Streamlit upload into LangChain documents."""
    name = uploaded_file.name
    extension = Path(name).suffix.lower()
    data = uploaded_file.getvalue()

    if extension == ".pdf":
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(data)
                temporary_path = temp_file.name
            documents = PyPDFLoader(temporary_path).load()
            for document in documents:
                document.metadata.update({"source": name, "source_type": "PDF"})
            return documents
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

    if extension == ".docx":
        document = DocxDocument(io.BytesIO(data))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return _text_document(text, name, "DOCX")

    if extension == ".pptx":
        presentation = Presentation(io.BytesIO(data))
        slides = []
        for number, slide in enumerate(presentation.slides, start=1):
            text = "\n".join(
                shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()
            )
            if text:
                slides.append(Document(
                    page_content=text,
                    metadata={"source": name, "source_type": "PPTX", "slide": number},
                ))
        if not slides:
            raise ValueError("No readable text was found in this presentation.")
        return slides

    decoded = data.decode("utf-8", errors="replace")
    if extension == ".csv":
        rows = list(csv.reader(io.StringIO(decoded)))
        text = "\n".join(" | ".join(row) for row in rows)
        return _text_document(text, name, "CSV")
    return _text_document(decoded, name, "Markdown" if extension == ".md" else "Text")


def load_webpage(url: str) -> list[Document]:
    """Extract readable text from a public HTTP(S) webpage."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a complete webpage URL beginning with http:// or https://.")
    response = requests.get(
        url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (RAG Learning Assistant)"}
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    text = soup.get_text("\n", strip=True)
    documents = _text_document(text, title, "Web page")
    documents[0].metadata["url"] = url
    return documents


def extract_youtube_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/")
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [""])[0] or parsed.path.split("/")[-1]
    raise ValueError("Enter a valid YouTube video URL.")


def _fetch_with_ytdlp(url: str) -> list[dict]:
    """Fallback transcript extraction via yt-dlp (resilient against IP blocks)."""
    try:
        import json
        import urllib.request
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subtitles = info.get("subtitles", {}) or {}
            auto_subtitles = info.get("automatic_captions", {}) or {}

            # Preferred order: manual en/hi, then auto en/hi, then any available
            target_entry = None
            for lang in ["en", "hi"]:
                if lang in subtitles:
                    target_entry = subtitles[lang]
                    break
            if not target_entry:
                for lang in ["en", "hi"]:
                    if lang in auto_subtitles:
                        target_entry = auto_subtitles[lang]
                        break
            if not target_entry and subtitles:
                target_entry = next(iter(subtitles.values()))
            if not target_entry and auto_subtitles:
                target_entry = next(iter(auto_subtitles.values()))

            if not target_entry:
                return []

            # Find json3 format for timestamps or fallback to vtt/srv
            json_url = next((f["url"] for f in target_entry if f.get("ext") == "json3"), None)
            if json_url:
                req = urllib.request.Request(json_url, headers={"User-Agent": "Mozilla/5.0"})
                data = json.loads(urllib.request.urlopen(req, timeout=15).read())
                items = []
                for event in data.get("events", []):
                    start = event.get("tStartMs", 0) / 1000.0
                    segs = event.get("segs", [])
                    line = "".join(s.get("utf8", "") for s in segs).strip()
                    if line:
                        items.append({"start": start, "text": line})
                return items
    except Exception:
        pass
    return []


def load_youtube_transcript(url: str) -> list[Document]:
    """Turn an available YouTube transcript into timestamped documents.

    English is preferred, followed by Hindi. If neither is available, the first
    public transcript returned by YouTube is used.
    """
    video_id = extract_youtube_id(url)
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("The YouTube video ID could not be identified.")

    transcript = None
    # 1. Try youtube-transcript-api first
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            matched_transcript = transcript_list.find_transcript(["en", "hi"])
        except Exception:
            available = list(transcript_list)
            if not available:
                raise ValueError("No transcript is available for this video.")
            matched_transcript = available[0]
        transcript = matched_transcript.fetch().to_raw_data()
    except Exception as primary_error:
        # 2. Fallback to yt-dlp if blocked or failed
        transcript = _fetch_with_ytdlp(url)
        if not transcript:
            raise primary_error

    groups, text, start = [], [], None
    for item in transcript:
        item_start = float(item["start"])
        if start is None:
            start = item_start
        if item_start - start >= 60 and text:
            groups.append(Document(
                page_content=" ".join(text),
                metadata={"source": url, "source_type": "YouTube", "start_seconds": int(start)},
            ))
            text, start = [], item_start
        text.append(item["text"])
    if text:
        groups.append(Document(
            page_content=" ".join(text),
            metadata={"source": url, "source_type": "YouTube", "start_seconds": int(start or 0)},
        ))
    if not groups:
        raise ValueError("No transcript is available for this video.")
    return groups


def source_label(metadata: dict) -> str:
    """Return a concise, human-readable citation label."""
    source_type = metadata.get("source_type", "Source")
    if source_type == "PDF":
        return f"PDF · page {metadata.get('page', 0) + 1}"
    if source_type == "PPTX":
        return f"Presentation · slide {metadata.get('slide', 1)}"
    if source_type == "YouTube":
        seconds = int(metadata.get("start_seconds", 0))
        return f"YouTube · {seconds // 60}:{seconds % 60:02d}"
    return f"{source_type} · {metadata.get('source', 'uploaded source')}"
