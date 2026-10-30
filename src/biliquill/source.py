from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from yt_dlp import YoutubeDL
from yt_dlp.networking import Request
from yt_dlp.networking.exceptions import RequestError
from yt_dlp.utils import DownloadError

from .core import QuillError, Segment, serialize_segments, write_json


def validate_url(url: str) -> None:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme not in {"http", "https"} or not (
        host == "b23.tv" or host == "bilibili.com" or host.endswith(".bilibili.com")
    ):
        raise QuillError("Enter a Bilibili video URL (b23.tv short links are supported)")


def output_name(url: str) -> str:
    from .core import fingerprint

    match = re.search(r"/(BV[0-9A-Za-z]+|av\d+)(?:/|$)", urlsplit(url).path)
    name = match.group(1) if match else fingerprint(url)[:12]
    page = parse_qs(urlsplit(url).query).get("p", ["1"])[0]
    if not page.isdigit() or int(page) < 1:
        raise QuillError("The video page parameter p must be a positive integer")
    return name + (f"-p{int(page)}" if int(page) != 1 else "")


def clock_value(value: str) -> float:
    values = value.replace(",", ".").split(":")
    return sum(float(part) * 60**index for index, part in enumerate(reversed(values)))


def parse_subtitle(text: str, ext: str) -> list[Segment]:
    if ext == "json":
        body = json.loads(text).get("body", [])
        rows = [
            Segment(float(x["from"]), float(x["to"]), str(x["content"]).strip())
            for x in body
            if str(x.get("content", "")).strip()
        ]
    elif ext in {"srt", "vtt"}:
        rows = []
        for block in re.split(r"\n\s*\n", text.replace("\r", "").strip()):
            lines = block.splitlines()
            index = next((i for i, line in enumerate(lines) if "-->" in line), None)
            if index is None:
                continue
            start, end = lines[index].split("-->", 1)
            content = html.unescape(re.sub(r"<[^>]+>", "", " ".join(lines[index + 1 :]))).strip()
            if content:
                rows.append(
                    Segment(
                        clock_value(start.strip()), clock_value(end.strip().split()[0]), content
                    )
                )
    else:
        raise ValueError(f"Unsupported subtitle format: {ext}")
    return sorted(rows, key=lambda s: (s.start, s.end))


def subtitle_candidates(info: dict, language: str):
    candidates = []
    for automatic, key in enumerate(("subtitles", "automatic_captions")):
        for lang, formats in (info.get(key) or {}).items():
            for entry in formats:
                ext = entry.get("ext")
                if ext not in {"json", "srt", "vtt"} or not entry.get("url"):
                    continue  # Bilibili danmaku XML is comments, not subtitles.
                rank = (
                    0
                    if lang.lower() == language.lower()
                    else (1 if lang.lower().split("-")[0] == language.lower().split("-")[0] else 2)
                )
                candidates.append(
                    (
                        (rank, automatic, {"json": 0, "srt": 1, "vtt": 2}[ext]),
                        lang,
                        automatic,
                        entry,
                    )
                )
    return sorted(candidates, key=lambda item: item[0])


def acquire(
    url: str,
    cache: Path,
    *,
    language: str,
    asr_model: str,
    device: str,
    compute_type: str,
    batch_size: int,
    cookies: str | None,
    browser: str | None,
    refresh: bool = False,
) -> tuple[dict, list[Segment]]:
    cache.mkdir(parents=True, exist_ok=True)
    saved = cache / "source.json"
    identity = {"url": url, "language": language, "asr_model": asr_model}
    if saved.exists() and not refresh:
        data = json.loads(saved.read_text(encoding="utf-8"))
        if data.get("identity") == identity:
            print("Reusing the saved transcript", file=sys.stderr)
            return data["metadata"], [Segment(**s) for s in data["segments"]]
    options = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "format": "bestaudio/best",
        "outtmpl": str(cache / "audio.%(ext)s"),
    }
    if cookies:
        options["cookiefile"] = cookies
    if browser:
        options["cookiesfrombrowser"] = (browser,)
    try:
        with YoutubeDL(options) as ydl:
            print("Checking video metadata and existing captions...", file=sys.stderr)
            info = ydl.extract_info(url, download=False)
            if not info or info.get("_type") in {"playlist", "multi_video"}:
                raise QuillError(
                    "Provide a single video URL or select a specific video page with ?p="
                )
            metadata = {
                "id": str(info["id"]),
                "title": info.get("title") or str(info["id"]),
                "url": url,
                "uploader": info.get("uploader"),
                "duration": info.get("duration"),
            }
            segments = []
            for _, lang, automatic, entry in subtitle_candidates(info, language):
                try:
                    headers = {**info.get("http_headers", {}), **entry.get("http_headers", {})}
                    with ydl.urlopen(Request(entry["url"], headers=headers)) as response:
                        content = response.read().decode("utf-8-sig")
                    segments = parse_subtitle(content, entry["ext"])
                    if segments:
                        metadata["text_source"] = (
                            f"{'Automatic' if automatic else 'Existing'} captions / {lang}"
                        )
                        break
                except (
                    RequestError,
                    DownloadError,
                    ValueError,
                    KeyError,
                    TypeError,
                    OSError,
                ) as exc:
                    print(
                        f"Caption {lang} failed ({type(exc).__name__}); trying the next candidate",
                        file=sys.stderr,
                    )
            if not segments:
                print("No usable captions; downloading audio...", file=sys.stderr)
                downloaded = ydl.process_ie_result(info, download=True)
                paths = [d.get("filepath") for d in downloaded.get("requested_downloads", [])]
                audio = next(
                    (Path(p) for p in paths if p and Path(p).is_file()),
                    None,
                )
                if audio is None:
                    audio = Path(ydl.prepare_filename(downloaded))
                if not audio.is_file():
                    raise QuillError("Downloaded audio file was not found")
                from .asr import transcribe

                segments = transcribe(
                    audio,
                    cache,
                    model_name=asr_model,
                    device=device,
                    compute_type=compute_type,
                    batch_size=batch_size,
                    language=language.split("-")[0],
                    refresh=refresh,
                )
                metadata["text_source"] = f"Local {device} transcription / {asr_model}"
    except DownloadError as exc:
        raise QuillError("yt-dlp failed; try --cookies or --cookies-from-browser") from exc
    if not segments:
        raise QuillError("No usable speech or captions were found")
    write_json(
        saved,
        {"identity": identity, "metadata": metadata, "segments": serialize_segments(segments)},
    )
    return metadata, segments
