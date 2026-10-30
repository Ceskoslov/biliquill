import io
import json
import wave
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import httpx
import pytest

from biliquill import asr, cli, source
from biliquill.core import (
    QuillError,
    Segment,
    chunks,
    segment_line,
    timestamp,
    video_link,
    write_json,
)
from biliquill.llm import LLM, parse_note, reduce_notes

URL = "https://www.bilibili.com/video/BV13Wf2B8ET6?p=2"
METADATA = {
    "id": "BV13Wf2B8ET6",
    "title": "测试视频",
    "url": URL,
    "uploader": "作者",
    "duration": 40,
    "text_source": "Existing captions / zh",
}
SEGMENTS = [
    Segment(1.25, 5, "第一部分：用例子解释观点。"),
    Segment(10, 15, "第二部分：不同意见与结论。"),
]


def note(batch=SEGMENTS):
    return {
        "title": "主题",
        "summary": "作者通过例子论证观点，并保留不同意见。",
        "chapters": [
            {"start": timestamp(batch[0].start), "title": "论证", "summary": "解释观点及其限定。"}
        ],
    }


def client_with(tmp_path, handler):
    llm = LLM("https://example.test/v1", "test-secret", "test-model", tmp_path)
    llm.client.close()
    llm.client = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


def response(content, status=200, finish="stop"):
    if status != 200:
        return httpx.Response(status, json={"error": "test"})
    return httpx.Response(
        200, json={"choices": [{"finish_reason": finish, "message": {"content": content}}]}
    )


def test_bilibili_json_and_vtt_parsing():
    body = {
        "body": [
            {"from": 2, "to": 4, "content": "后一句"},
            {"from": 0, "to": 1, "content": "前一句"},
            {"from": 4, "to": 5, "content": "  "},
        ]
    }
    assert [s.text for s in source.parse_subtitle(json.dumps(body), "json")] == ["前一句", "后一句"]
    vtt = "WEBVTT\n\n00:01.250 --> 00:03.500 align:start\n<b>你好</b> &amp; 再见\n"
    assert source.parse_subtitle(vtt, "vtt") == [Segment(1.25, 3.5, "你好 & 再见")]
    srt = "1\r\n00:00:01,000 --> 00:00:02,000\r\n第一行\r\n第二行\r\n"
    assert source.parse_subtitle(srt, "srt") == [Segment(1, 2, "第一行 第二行")]


def test_subtitle_priority_excludes_danmaku():
    info = {
        "subtitles": {
            "danmaku": [{"ext": "xml", "url": "comments"}],
            "en": [{"ext": "vtt", "url": "english"}],
            "zh-CN": [{"ext": "json", "url": "manual"}],
        },
        "automatic_captions": {"zh": [{"ext": "json", "url": "auto"}]},
    }
    candidates = source.subtitle_candidates(info, "zh")
    # Exact-language automatic captions precede another Chinese locale.
    assert [c[3]["url"] for c in candidates] == ["auto", "manual", "english"]
    assert all(c[3]["ext"] != "xml" for c in candidates)


@pytest.mark.parametrize(
    "url", ["https://evil.test/video/BV1", "https://bilibili.com.evil.test/", "file:///tmp/video"]
)
def test_url_validation(url):
    with pytest.raises(QuillError):
        source.validate_url(url)


def test_page_identity_and_timestamp_links():
    assert source.output_name(URL) == "BV13Wf2B8ET6-p2"
    assert source.output_name(URL.replace("p=2", "p=1")) == "BV13Wf2B8ET6"
    assert "?p=2&t=42" in video_link(URL + "&t=3", 42)
    assert timestamp(3601.9) == "01:00:01"
    with pytest.raises(QuillError):
        source.output_name(URL.replace("p=2", "p=../../bad"))


@pytest.mark.parametrize(
    "start,end,text", [(3, 2, "x"), (-1, 2, "x"), (0, float("nan"), "x"), (0, 1, " ")]
)
def test_bad_segment_rejected(start, end, text):
    with pytest.raises(ValueError):
        Segment(start, end, text)


def test_chunks_bound_oversized_subtitles_without_losing_text():
    text = "这是很长的字幕。" * 1000
    result = chunks([Segment(0, 100, text)], 1000)
    assert len(result) > 1
    assert "".join(s.text for batch in result for s in batch) == text
    assert all(sum(len(segment_line(s)) + 1 for s in batch) <= 1000 for batch in result)


def test_note_rejects_invented_or_skipped_start_time():
    valid = note()
    assert parse_note("```json\n" + json.dumps(valid) + "\n```", SEGMENTS) == valid
    valid["chapters"][0]["start"] = "00:00:09"
    with pytest.raises(ValueError, match="beginning|timestamps"):
        parse_note(json.dumps(valid), SEGMENTS)
    valid["chapters"][0]["start"] = "00:00:10"
    with pytest.raises(ValueError, match="beginning"):
        parse_note(json.dumps(valid), SEGMENTS)


def test_llm_retries_then_caches_and_invalidates_model(tmp_path, monkeypatch):
    monkeypatch.setattr("biliquill.llm.time.sleep", lambda _: None)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response("答案", 429 if len(calls) == 1 else 200)

    llm = client_with(tmp_path, handler)
    assert llm.complete("问题") == "答案"
    assert llm.complete("问题") == "答案"
    assert len(calls) == 2
    llm.model = "another-model"
    assert llm.complete("问题") == "答案"
    assert len(calls) == 3
    assert "test-secret" not in "".join(p.read_text() for p in tmp_path.iterdir())
    llm.close()


def test_note_language_defaults_to_english_and_changes_cache(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response("Notes")

    llm = client_with(tmp_path, handler)
    llm.complete("Topic")
    assert "English" in calls[0]["messages"][0]["content"]
    llm.language = "zh"
    llm.complete("Topic")
    assert len(calls) == 2
    assert "Simplified Chinese" in calls[1]["messages"][0]["content"]
    llm.close()


@pytest.mark.parametrize("base_url", ["http://localhost:11434", "http://localhost:11434/api/"])
def test_ollama_native_requests_and_response_formats(tmp_path, base_url):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        assert request.url.path == "/api/chat"
        assert "authorization" not in request.headers
        content = json.dumps(note()) if len(calls) == 1 else "# Summary"
        return httpx.Response(
            200, json={"done": True, "done_reason": "stop", "message": {"content": content}}
        )

    llm = LLM(base_url, None, "installed-model", tmp_path, provider="ollama", context_size=8192)
    llm.client.close()
    llm.client = httpx.Client(transport=httpx.MockTransport(handler))
    result = llm.complete("Chunk", validate=lambda value: parse_note(value, SEGMENTS))
    assert parse_note(result, SEGMENTS) == note()
    assert calls[0]["format"] == "json"
    assert calls[0]["options"] == {"num_ctx": 8192}
    assert calls[0]["stream"] is False
    assert llm.complete("Synthesis") == "# Summary"
    assert "format" not in calls[1]
    llm.close()


@pytest.mark.parametrize("done,reason", [(False, "stop"), (True, "length")])
def test_incomplete_ollama_responses_are_not_cached(tmp_path, monkeypatch, done, reason):
    monkeypatch.setattr("biliquill.llm.time.sleep", lambda _: None)
    llm = LLM("http://localhost:11434", None, "local-model", tmp_path, provider="ollama")
    llm.client.close()
    llm.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"done": done, "done_reason": reason, "message": {"content": "Partial"}}
            )
        )
    )
    with pytest.raises(QuillError, match="incomplete|truncated"):
        llm.complete("Question")
    assert not list(tmp_path.glob("llm-*"))
    llm.close()


@pytest.mark.parametrize(
    "provider,expected",
    [("local", "http://localhost:1234/v1"), ("ollama", "http://localhost:11434")],
)
def test_local_providers_ignore_cloud_configuration(
    tmp_path, fake_source, monkeypatch, provider, expected
):
    monkeypatch.setenv("BILIQUILL_BASE_URL", "https://cloud.test/v1")
    monkeypatch.setenv("BILIQUILL_API_KEY", "private-cloud-key")
    monkeypatch.setenv("OPENAI_API_KEY", "another-cloud-key")
    monkeypatch.delenv(f"BILIQUILL_{provider.upper()}_BASE_URL", raising=False)
    monkeypatch.delenv(f"BILIQUILL_{provider.upper()}_API_KEY", raising=False)
    seen = []
    original_init = LLM.__init__

    def init(self, base_url, api_key, *args, **kwargs):
        seen.append((base_url, api_key, kwargs))
        original_init(self, base_url, api_key, *args, **kwargs)

    monkeypatch.setattr(LLM, "__init__", init)
    monkeypatch.setattr(cli, "summarize", lambda *a, **k: ("Summary", "Chapters"))
    assert (
        cli.main([URL, "-o", str(tmp_path), "--provider", provider, "--model", "installed-model"])
        == 0
    )
    assert seen[0][0] == expected
    assert seen[0][1] is None
    assert seen[0][2]["provider"] == provider


def test_local_custom_endpoint_key_and_timeout(tmp_path, fake_source, monkeypatch):
    monkeypatch.setenv("BILIQUILL_PROVIDER", "local")
    monkeypatch.setenv("BILIQUILL_LOCAL_API_KEY", "local-only-key")
    seen = []
    original_init = LLM.__init__

    def init(self, base_url, api_key, *args, **kwargs):
        seen.append((base_url, api_key, kwargs))
        original_init(self, base_url, api_key, *args, **kwargs)

    monkeypatch.setattr(LLM, "__init__", init)
    monkeypatch.setattr(cli, "summarize", lambda *a, **k: ("Summary", "Chapters"))
    assert (
        cli.main(
            [
                URL,
                "-o",
                str(tmp_path),
                "--model",
                "model.gguf",
                "--base-url",
                "http://localhost:8080/v1",
                "--llm-timeout",
                "900",
            ]
        )
        == 0
    )
    assert seen[0][0:2] == ("http://localhost:8080/v1", "local-only-key")
    assert seen[0][2]["timeout"] == 900


@pytest.mark.parametrize(
    "options",
    [
        ["--llm-timeout", "nan"],
        ["--llm-timeout", "0"],
        ["--provider", "local", "--context-size", "8192"],
        ["--provider", "ollama", "--context-size", "-1"],
    ],
)
def test_invalid_local_options_fail_before_download(monkeypatch, options):
    monkeypatch.setattr(cli, "acquire", lambda *a, **k: pytest.fail("Must validate first"))
    assert cli.main([URL, "--model", "test", *options]) == 1


def test_llm_invalid_json_not_cached_and_auth_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr("biliquill.llm.time.sleep", lambda _: None)
    replies = iter(["bad json", json.dumps(note())])
    llm = client_with(tmp_path, lambda _: response(next(replies)))
    text = llm.complete("问题", validate=lambda value: parse_note(value, SEGMENTS))
    assert parse_note(text, SEGMENTS) == note()
    assert len(list(tmp_path.glob("llm-*"))) == 1
    llm.close()
    attempts = []
    llm = client_with(tmp_path, lambda r: attempts.append(r) or response("", 401))
    with pytest.raises(QuillError, match="401"):
        llm.complete("不同的问题")
    assert len(attempts) == 1
    llm.close()


def test_truncated_llm_output_never_becomes_a_report(tmp_path, monkeypatch):
    monkeypatch.setattr("biliquill.llm.time.sleep", lambda _: None)
    llm = client_with(tmp_path, lambda _: response("半个答案", finish="length"))
    with pytest.raises(QuillError, match="truncated"):
        llm.complete("问题")
    assert not list(tmp_path.glob("llm-*"))
    llm.close()


def test_hierarchical_reduction_bounds_all_note_inputs():
    calls = []

    def complete(prompt):
        calls.append(prompt)
        assert len(prompt.split("<notes>\n")[1].split("\n</notes>")[0]) <= 1000
        return "压缩后的笔记" if "Compress these notes" in prompt else "最终总结"

    llm = SimpleNamespace(complete=complete)
    assert reduce_notes(llm, ["材料" * 3000], 1000) == "最终总结"
    assert len(calls) == 7


class FakeYDL:
    info: ClassVar[dict] = {}
    downloads = 0
    requests: ClassVar[list] = []
    subtitle_payload = json.dumps({"body": [dict(**asdict(SEGMENTS[0]))]})

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def extract_info(self, url, download):
        assert not download
        return self.info.copy()

    def urlopen(self, request):
        self.requests.append(request.url)
        return io.BytesIO(self.subtitle_payload.encode())

    def process_ie_result(self, info, download):
        assert download
        type(self).downloads += 1
        path = Path(self.options["outtmpl"].replace("%(ext)s", "m4a"))
        path.write_bytes(b"fake audio")
        return {**info, "requested_downloads": [{"filepath": str(path)}]}


@pytest.fixture
def fake_source(monkeypatch):
    FakeYDL.info = {
        "id": "BV13Wf2B8ET6",
        "title": "测试视频",
        "duration": 40,
        "subtitles": {"zh": [{"ext": "json", "url": "https://sub.test/a.json"}]},
    }
    FakeYDL.downloads, FakeYDL.requests = 0, []
    FakeYDL.subtitle_payload = json.dumps(
        {"body": [{"from": s.start, "to": s.end, "content": s.text} for s in SEGMENTS]}
    )
    monkeypatch.setattr(source, "YoutubeDL", FakeYDL)
    return FakeYDL


def acquire(tmp_path):
    return source.acquire(
        URL,
        tmp_path,
        language="zh",
        asr_model="test",
        device="cuda",
        compute_type="int8_float16",
        batch_size=8,
        cookies=None,
        browser=None,
    )


def test_subtitles_skip_audio_and_repeat_skips_network(tmp_path, fake_source, monkeypatch):
    monkeypatch.setattr(asr, "transcribe", lambda *a, **k: pytest.fail("ASR must not run"))
    metadata, segments = acquire(tmp_path)
    assert segments == SEGMENTS
    assert metadata["text_source"] == "Existing captions / zh"
    assert fake_source.downloads == 0
    assert len(fake_source.requests) == 1
    monkeypatch.setattr(source, "YoutubeDL", lambda _: pytest.fail("Should use checkpoint"))
    assert acquire(tmp_path) == (metadata, segments)


@pytest.mark.parametrize("broken", [False, True])
def test_no_subtitles_or_broken_subtitles_fall_back_to_audio(
    tmp_path, fake_source, monkeypatch, broken
):
    if broken:
        fake_source.subtitle_payload = "not json"
    else:
        fake_source.info["subtitles"] = {"danmaku": [{"ext": "xml", "url": "comments"}]}
    monkeypatch.setattr(asr, "transcribe", lambda *a, **k: SEGMENTS)
    metadata, segments = acquire(tmp_path)
    assert segments == SEGMENTS
    assert fake_source.downloads == 1
    assert metadata["text_source"].startswith("Local cuda")


def test_cli_full_subtitle_pipeline(tmp_path, fake_source, monkeypatch):
    monkeypatch.setenv("BILIQUILL_MODEL", "test-model")
    monkeypatch.setenv("BILIQUILL_API_KEY", "test-secret")
    monkeypatch.setenv("BILIQUILL_BASE_URL", "https://llm.test/v1")
    calls = []

    def handler(request):
        calls.append(request)
        prompt = json.loads(request.content)["messages"][-1]["content"]
        return response(
            json.dumps(note()) if "<transcript>" in prompt else "## 主旨\n作者提出了观点。"
        )

    original_init = LLM.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.client.close()
        self.client = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(LLM, "__init__", init)
    assert cli.main([URL, "-o", str(tmp_path)]) == 0
    assert all(
        (tmp_path / name).is_file() for name in ["transcript.md", "summary.md", "chapters.md"]
    )
    assert "?p=2&t=1" in (tmp_path / "chapters.md").read_text()
    assert "Chunk notes" in (tmp_path / "summary.md").read_text()
    assert len(calls) == 2
    assert cli.main([URL, "-o", str(tmp_path)]) == 0
    assert len(calls) == 2


def test_cli_missing_llm_config_fails_before_downloading(tmp_path, monkeypatch):
    for name in ["BILIQUILL_MODEL", "OPENAI_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "acquire", lambda *a, **k: pytest.fail("Must validate first"))
    assert cli.main([URL, "-o", str(tmp_path)]) == 1


def test_cli_keeps_transcript_when_llm_fails(tmp_path, fake_source, monkeypatch):
    monkeypatch.setattr(LLM, "complete", lambda *a, **k: (_ for _ in ()).throw(QuillError("离线")))
    assert (
        cli.main(
            [URL, "-o", str(tmp_path), "--model", "test", "--base-url", "http://localhost:1234/v1"]
        )
        == 1
    )
    assert (tmp_path / "transcript.md").is_file()
    assert not (tmp_path / "summary.md").exists()


def test_window_decoder_seeks_to_correct_audio(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    # Three distinct tones make seeking to the wrong window detectable.
    rate = 16000
    waveform = np.concatenate(
        [np.sin(2 * np.pi * hz * np.arange(rate) / rate) for hz in [220, 440, 880]]
    )
    path = tmp_path / "tones.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((waveform * 16000).astype(np.int16).tobytes())
    decoded = asr.decode_window(path, 1, 1)
    assert len(decoded) == rate
    assert np.max(np.abs(decoded - waveform[rate : 2 * rate] * 16000 / 32768)) < 0.001


def test_asr_checkpoint_skips_completed_windows(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper")
    import av
    import faster_whisper

    from biliquill.core import fingerprint

    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")
    key = fingerprint({"model": "test", "language": "zh", "size": 4, "decode": 1})[:16]
    write_json(tmp_path / f"asr-{key}-000000.json", [asdict(SEGMENTS[0])])

    class Container:
        duration = 120 * av.time_base

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(av, "open", lambda _: Container())
    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *a, **k: object())
    monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", lambda **k: object())
    monkeypatch.setattr(asr, "decode_window", lambda *a: pytest.fail("Must use checkpoint"))
    assert asr.transcribe(
        audio,
        tmp_path,
        model_name="test",
        device="cpu",
        compute_type="int8",
        batch_size=8,
        language="zh",
    ) == [SEGMENTS[0]]
