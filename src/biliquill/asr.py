from __future__ import annotations

import ctypes
import importlib.util
import json
import sys
from pathlib import Path

from .core import QuillError, Segment, fingerprint


def load_cuda_libraries() -> None:
    """Preload pip's optional CUDA libraries before CTranslate2 opens them."""
    if not sys.platform.startswith("linux"):
        return
    for module, patterns in [
        ("nvidia.cublas", ["libcublasLt.so.12", "libcublas.so.12"]),
        ("nvidia.cudnn", ["libcudnn.so.9"]),
    ]:
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            continue
        if spec and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                for pattern in patterns:
                    path = Path(location) / "lib" / pattern
                    if path.exists():
                        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def transcribe(
    audio: Path,
    cache: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    batch_size: int,
    language: str,
    refresh: bool = False,
) -> list[Segment]:
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        from opencc import OpenCC
    except ImportError as exc:
        raise QuillError(
            "Missing ASR dependencies; install biliquill[gpu], or biliquill[asr] for CPU"
        ) from exc
    try:
        if device == "cuda":
            load_cuda_libraries()
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        pipeline = BatchedInferencePipeline(model=model)
        cc = OpenCC("t2s")
        # Decode in bounded 10-minute windows and checkpoint each completed window.
        import av

        with av.open(str(audio)) as container:
            duration = float(container.duration / av.time_base)
        window = 600
        settings = fingerprint(
            {"model": model_name, "language": language, "size": audio.stat().st_size, "decode": 1}
        )[:16]
        segments = []
        for start in range(0, int(duration) + 1, window):
            if start >= duration:
                break
            path = cache / f"asr-{settings}-{start:06}.json"
            if path.exists() and not refresh:
                rows = json.loads(path.read_text(encoding="utf-8"))
            else:
                from .core import serialize_segments, write_json

                audio_data = decode_window(audio, start, min(window, duration - start))
                result, _ = pipeline.transcribe(
                    audio_data,
                    language=language,
                    batch_size=batch_size,
                    beam_size=5,
                    vad_filter=True,
                    without_timestamps=False,
                )
                rows = serialize_segments(
                    [
                        Segment(
                            start + s.start,
                            min(duration, start + s.end),
                            cc.convert(s.text.strip()),
                        )
                        for s in result
                        if s.text.strip()
                    ]
                )
                write_json(path, rows)
            segments.extend(Segment(**s) for s in rows)
            print(f"Transcription {min(start + window, duration) / duration:.0%}", file=sys.stderr)
        return segments
    except (RuntimeError, OSError, ValueError) as exc:
        raise QuillError(
            "Local transcription failed; check CUDA/cuDNN and VRAM, or lower --batch-size. "
            "For CPU use --device cpu --compute-type int8. "
            f"({type(exc).__name__})"
        ) from exc


def decode_window(path: Path, start: int, duration: float):
    """Seek a compressed file without decoding the complete multi-hour recording."""
    import av
    import numpy as np

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        arrays = []
        for frame in container.decode(stream):
            frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else start
            if frame_time >= start + duration:
                break
            for converted in resampler.resample(frame):
                time = (
                    float(converted.pts * converted.time_base)
                    if converted.pts is not None
                    else frame_time
                )
                samples = converted.to_ndarray().reshape(-1)
                lo = max(0, round((start - time) * 16000))
                hi = min(len(samples), round((start + duration - time) * 16000))
                if hi > lo:
                    arrays.append(samples[lo:hi])
        for converted in resampler.resample(None):
            arrays.append(converted.to_ndarray().reshape(-1))
    if not arrays:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(arrays).astype(np.float32)[: round(duration * 16000)] / 32768
