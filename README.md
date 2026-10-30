# biliquill 

**English** · [简体中文](README.zh-CN.md)

A small quill for long Bilibili videos.

Give it a URL. It tries existing captions first, downloads audio with yt-dlp when needed, transcribes locally on your GPU, and uses a local or hosted LLM to write timestamped notes.

```text
Bilibili URL → existing captions ──────────┐
             no captions → audio → ASR ───┤→ transcript.md
                                          └→ chunk notes → synthesis → summary.md
                                                        └→ chapters.md
```

No server, database, or task queue. Audio stays on your machine; transcript text and notes are sent to your configured LLM endpoint. CLI text and documentation default to English. Summaries and chapters also default to English; use `--notes-language zh` for Chinese. Transcripts preserve the source language.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). For Linux x86_64 or WSL2 with an NVIDIA GPU:

```bash
git clone https://github.com/Ceskoslov/biliquill.git
cd biliquill
uv sync --extra gpu
```

The GPU extra installs faster-whisper, CUDA 12 cuBLAS, and cuDNN 9 runtime libraries. A working NVIDIA driver is still required. The first transcription downloads its Whisper model; defaults are `large-v3-turbo`, `int8_float16`, and batch size 8. Lower `--batch-size` if VRAM is tight.

For captions only: `uv sync`. For CPU transcription: `uv sync --extra asr` and run with `--device cpu`. Native Windows needs separately configured CUDA/cuDNN DLLs; WSL2 is recommended.

Alternatively, install a standalone command:

```bash
uv tool install 'biliquill[gpu] @ git+https://github.com/Ceskoslov/biliquill'
```

## Run

Configure your OpenAI-compatible provider once:

```bash
export BILIQUILL_API_KEY='your-api-key'
export BILIQUILL_BASE_URL='https://api.openai.com/v1'
export BILIQUILL_MODEL='your-model-name'

uv run biliquill 'https://www.bilibili.com/video/BV13Wf2B8ET6'
```

If installed with `uv tool install`, run `biliquill URL` directly. `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` are accepted as fallbacks; `BILIQUILL_*` takes precedence. No particular model is assumed. `.env.example` documents the settings; `.env` files are not automatically loaded.

The client uses `POST <base-url>/chat/completions`. Your model must accept system/user messages and follow JSON instructions. No vendor-specific structured-output parameters are required. Keys are optional for local endpoints that do not require authentication.

### Local models

Select a model already installed in your local server. No cloud API key is needed.

```bash
# Native Ollama API; start Ollama and download a model beforehand
ollama list
uv run biliquill URL --provider ollama --model YOUR_INSTALLED_MODEL

# LM Studio: load a model and start its local API server
uv run biliquill URL --provider local --model YOUR_LOADED_MODEL

# llama.cpp or another OpenAI-compatible local server
uv run biliquill URL --provider local --model YOUR_SERVED_MODEL \
  --base-url http://localhost:8080/v1

# Chinese notes with a smaller Ollama context and smaller input chunks
uv run biliquill URL --provider ollama --model YOUR_INSTALLED_MODEL \
  --notes-language zh --context-size 8192 --chunk-chars 2000 --reduce-chars 4000
```

`--provider ollama` defaults to `http://localhost:11434/api/chat`; use the server root or `/api` as `--base-url`. It disables streaming, requests JSON for chunk notes, and leaves the final Markdown synthesis unconstrained. `--context-size` sets Ollama's `num_ctx` (default: 16384 tokens). Smaller windows may require smaller chunk/reduction limits and shorter outputs.

`--provider local` uses the same Chat Completions protocol as the hosted client, with a default base URL of `http://localhost:1234/v1`. It works with local servers that implement that protocol; it does not load GGUF or Transformers model files directly. Start the server and load/download your model first. The tool does not start servers or download LLM weights.

Set `BILIQUILL_PROVIDER=ollama` or `local` to make a backend your default. Local modes use their own optional `BILIQUILL_OLLAMA_BASE_URL` / `BILIQUILL_LOCAL_BASE_URL` settings and do not inherit cloud base URLs or keys. An explicit `--base-url` takes precedence. For authenticated local proxies, use `BILIQUILL_OLLAMA_API_KEY` or `BILIQUILL_LOCAL_API_KEY`.

LLM requests allow 600 seconds by default, including initial model loading; override with `--llm-timeout 900`. Summarization stays on the configured local server; acquiring the video still requires Bilibili access. The local model must follow JSON instructions and provide usable final content.

The default output directory is `output/<video-id>/`; `?p=2` gets a separate `<video-id>-p2` directory:

```text
transcript.md     Source transcript with timestamp links
summary.md        Integrated summary followed by detailed chunk notes
chapters.md       LLM-generated chapter titles, links, and short descriptions
.biliquill/       Audio, transcription checkpoints, and LLM response cache
```

## Options

```bash
# Chinese notes; keep the transcript in the video's original language
uv run biliquill URL --notes-language zh

# Explicit output directory and model
uv run biliquill URL -o ./notes --model your-model

# Captions requiring a logged-in account
uv run biliquill URL --cookies-from-browser firefox
# Or: --cookies /path/to/cookies.txt

# Select a video page
uv run biliquill 'https://www.bilibili.com/video/BVxxxxxxxxxx?p=2'

# Transcript only; no LLM configuration needed
uv run biliquill URL --transcript-only

# CPU transcription
uv run biliquill URL --device cpu

# Smaller VRAM / context window
uv run biliquill URL --batch-size 2 --chunk-chars 3000 --reduce-chars 6000

# Use a downloaded faster-whisper model
uv run biliquill URL --asr-model /path/to/model

# Regenerate instead of reusing biliquill checkpoints
uv run biliquill URL --refresh
```

Replace `URL` with the actual address. `--language zh` controls preferred caption language and the ASR language hint. Captions in that language are preferred, followed by other available languages; manual captions win when language matches are equal. Bilibili JSON, SRT, and WebVTT are supported. Danmaku XML is explicitly excluded: comments are not captions.

## Long videos and resuming

- Audio is decoded and transcribed in ten-minute windows, with a checkpoint after each window. The full recording is never loaded as one waveform. Window boundaries may have minor sentence-boundary errors.
- Whisper decoding explicitly uses `without_timestamps=False`; the prototype showed missing sentences with timestamp-free decoding.
- Transcript chunks default to 6,000 **characters**, not exact tokens. Adjust `--chunk-chars` for your model's context size.
- Each chunk note is cached. Oversized synthesis inputs are recursively compressed instead of silently truncated; `--reduce-chars` bounds each group of notes.
- Transient network failures, rate limits, and invalid LLM output are retried up to three times. Authentication/parameter errors fail immediately.
- `transcript.md` is written before LLM calls. Rerun the same command after an interruption to reuse completed steps. Changes to the model, endpoint, prompts, or input invalidate the relevant LLM cache. API keys are not stored in the cache.
- Run one process per output directory. Use different directories for different videos, especially when passing `-o`.

Some captions require Bilibili login; supply your own cookies if necessary. Captions, ASR, and LLM notes can contain errors. Chapter timestamps are validated against the input transcript, but topic boundaries remain approximate. The tool preserves attribution and uncertainty; it does not independently verify every claim in a video.

## Development

```bash
uv sync --extra asr
uv run pytest
uv run ruff check .
uv build
```

Tests cover caption-first acquisition and audio fallback, comment exclusion, chunk limits, timestamp validation, retries/cache invalidation, recursive synthesis, checkpoint reuse, and CLI outputs. LLM tests use simulated responses and require no API key. A real provider requires your configuration; GPU execution requires suitable hardware.

## References

- [yt-dlp Python embedding](https://github.com/yt-dlp/yt-dlp#embedding-yt-dlp)
- [faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu)
- [OpenAI Docs: Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [Ollama native Chat API](https://docs.ollama.com/api/chat)

## License

[The Unlicense](LICENSE). Applies to this repository's code. Dependencies, models, and video content retain their own licenses.
