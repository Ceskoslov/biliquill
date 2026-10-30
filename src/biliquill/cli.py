from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .core import QuillError, atomic_write, render_transcript
from .llm import LLM, summarize
from .source import acquire, output_name, validate_url


def env(name: str, fallback: str = "") -> str:
    return os.environ.get(f"BILIQUILL_{name}") or os.environ.get(f"OPENAI_{name}") or fallback


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="biliquill - turn Bilibili videos into transcripts, summaries, and chapters"
    )
    p.add_argument("url", help="Bilibili video URL; supports ?p= pages and b23.tv")
    p.add_argument("-o", "--output", type=Path, help="Output directory (default: output/video-id)")
    p.add_argument("--model", default=env("MODEL"), help="LLM model name, or BILIQUILL_MODEL")
    p.add_argument(
        "--provider",
        choices=["openai", "ollama", "local"],
        default=os.environ.get("BILIQUILL_PROVIDER", "openai"),
        help="LLM backend: openai (default), native Ollama, or local OpenAI-compatible server",
    )
    p.add_argument(
        "--base-url",
        help="Override the provider's base URL; local and Ollama default to loopback addresses",
    )
    p.add_argument(
        "--llm-timeout",
        type=float,
        default=600,
        help="LLM request timeout in seconds (default: 600, including model loading)",
    )
    p.add_argument(
        "--context-size",
        type=int,
        help="Ollama context window in tokens (default: 16384; Ollama only)",
    )
    p.add_argument(
        "--language", default="zh", help="Preferred caption and ASR language (default: zh)"
    )
    p.add_argument(
        "--notes-language",
        choices=["en", "zh"],
        default="en",
        help="Summary and chapter language (default: en); transcripts stay in the source language",
    )
    p.add_argument("--asr-model", default="large-v3-turbo", help="Local Whisper model name or path")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--compute-type", help="Default: int8_float16 for CUDA, int8 for CPU")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--chunk-chars", type=int, default=6000, help="Maximum transcript characters per chunk"
    )
    p.add_argument(
        "--reduce-chars",
        type=int,
        default=12000,
        help="Maximum note characters per synthesis request",
    )
    auth = p.add_mutually_exclusive_group()
    auth.add_argument("--cookies", help="Path to a Netscape-format cookies file for yt-dlp")
    auth.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="Browser name, e.g. firefox or chrome, for authenticated captions",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-extract and regenerate, ignoring biliquill checkpoints",
    )
    p.add_argument(
        "--transcript-only",
        action="store_true",
        help="Write only the transcript; no LLM configuration needed",
    )
    p.add_argument("--version", action="version", version=f"biliquill {__version__}")
    return p


def run(args: argparse.Namespace) -> Path:
    validate_url(args.url)
    if args.batch_size < 1 or args.chunk_chars < 1000 or args.reduce_chars < 1000:
        raise QuillError(
            "batch-size must be positive; chunk-chars/reduce-chars must be at least 1000"
        )
    if args.provider not in {"openai", "ollama", "local"}:
        raise QuillError("BILIQUILL_PROVIDER must be openai, ollama, or local")
    if not math.isfinite(args.llm_timeout) or args.llm_timeout <= 0:
        raise QuillError("llm-timeout must be a positive finite number")
    if args.context_size is not None and (args.context_size <= 0 or args.provider != "ollama"):
        raise QuillError("context-size must be positive and requires --provider ollama")
    if args.provider == "openai":
        base_url = args.base_url or env("BASE_URL", "https://api.openai.com/v1")
        key = env("API_KEY")
    else:
        prefix = f"BILIQUILL_{args.provider.upper()}_"
        default_url = (
            "http://localhost:11434" if args.provider == "ollama" else "http://localhost:1234/v1"
        )
        base_url = args.base_url or os.environ.get(prefix + "BASE_URL") or default_url
        key = os.environ.get(prefix + "API_KEY", "")
    host = urlsplit(base_url).hostname
    if urlsplit(base_url).scheme not in {"https", "http"} or not host:
        raise QuillError("Base URL must be a valid HTTP(S) address")
    if not args.transcript_only:
        if not args.model:
            raise QuillError("Set BILIQUILL_MODEL or pass --model")
        if host == "api.openai.com" and not key:
            raise QuillError("Set BILIQUILL_API_KEY or OPENAI_API_KEY")
    destination = args.output or Path("output") / output_name(args.url)
    cache = destination / ".biliquill"
    metadata, segments = acquire(
        args.url,
        cache,
        language=args.language,
        asr_model=args.asr_model,
        device=args.device,
        compute_type=args.compute_type or ("int8_float16" if args.device == "cuda" else "int8"),
        batch_size=args.batch_size,
        cookies=args.cookies,
        browser=args.cookies_from_browser,
        refresh=args.refresh,
    )
    atomic_write(destination / "transcript.md", render_transcript(metadata, segments))
    if not args.transcript_only:
        llm = LLM(
            base_url,
            key or None,
            args.model,
            cache,
            refresh=args.refresh,
            language=args.notes_language,
            provider=args.provider,
            timeout=args.llm_timeout,
            context_size=args.context_size,
        )
        try:
            summary, chapters = summarize(
                metadata,
                segments,
                llm,
                chunk_chars=args.chunk_chars,
                reduce_chars=args.reduce_chars,
            )
        finally:
            llm.close()
        atomic_write(destination / "summary.md", summary)
        atomic_write(destination / "chapters.md", chapters)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        destination = run(args)
    except (QuillError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"biliquill: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\nStopped. Completed steps are cached; rerun the same command to resume.",
            file=sys.stderr,
        )
        return 130
    print(f"Done: {destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
