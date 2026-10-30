from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


class QuillError(Exception):
    """An actionable error safe to display in the CLI."""


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str

    def __post_init__(self):
        if not all(math.isfinite(t) for t in (self.start, self.end)):
            raise ValueError("Subtitle timestamps must be finite numbers")
        if self.start < 0 or self.end < self.start or not self.text.strip():
            raise ValueError("Invalid subtitle segment")


def timestamp(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, data) -> None:
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def fingerprint(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def video_link(url: str, seconds: float) -> str:
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "t"]
    query.append(("t", str(int(seconds))))
    return urlunsplit(parts._replace(query=urlencode(query), fragment=""))


def segment_line(segment: Segment) -> str:
    return f"[{timestamp(segment.start)}–{timestamp(segment.end)}] {segment.text.strip()}"


def chunks(segments: list[Segment], max_chars: int) -> list[list[Segment]]:
    """Bound the formatted input, including oversized individual subtitle entries."""
    result, current, size = [], [], 0
    # Reserve space for timestamp labels and newlines; never silently truncate text.
    text_limit = max_chars - 32
    for segment in segments:
        text = segment.text.strip()
        for offset in range(0, len(text), text_limit):
            part = Segment(segment.start, segment.end, text[offset : offset + text_limit])
            length = len(segment_line(part)) + 1
            if current and size + length > max_chars:
                result.append(current)
                current, size = [], 0
            current.append(part)
            size += length
    if current:
        result.append(current)
    return result


def render_transcript(metadata: dict, segments: list[Segment]) -> str:
    title = metadata["title"].replace("\n", " ")
    lines = [
        f"# {title}",
        "",
        f"Source: {metadata['url']}",
        f"Uploader: {metadata.get('uploader') or 'Unknown'}",
        f"Text source: {metadata['text_source']}",
        "",
        "Captions and automatic transcription may contain recognition errors; not manually proofread.",
        "",
    ]
    for segment in segments:
        link = video_link(metadata["url"], segment.start)
        lines += [f"[{timestamp(segment.start)}]({link}) {segment.text.strip()}", ""]
    return "\n".join(lines)


def serialize_segments(segments: list[Segment]) -> list[dict]:
    return [asdict(segment) for segment in segments]
