# biliquill 

[English](README.md) · **简体中文**

把长视频，写成可回看的笔记。

一个小型 Python CLI：输入 Bilibili URL，优先提取已有字幕；没有可用字幕时，用 yt-dlp 下载音频并在本地 GPU 上转写；最后调用 OpenAI 兼容 LLM，完成分段摘要和综合整理。

也支持 Ollama 原生接口，以及 LM Studio、llama.cpp 等本地 OpenAI 兼容服务，使用方式见下方“本地模型”。

```text
Bilibili URL → 已有字幕 ───────────────────┐
             无可用字幕 → 下载音频 → 本地 ASR ├→ transcript.md
                                        └→ 分段摘要 → 综合 → summary.md
                                                       └→ chapters.md
```

没有 Web 服务、数据库或任务队列。音频保留在本机；选定 LLM 服务接收的是字幕/转写文本和分段笔记。CLI 和文档说明默认使用英文。摘要和章节默认输出英文，使用 `--notes-language zh` 可切换为中文；逐字稿保留视频原语言。

## 安装

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。Linux x86_64 + NVIDIA GPU（包括 WSL2）：

```bash
git clone https://github.com/Ceskoslov/biliquill.git
cd biliquill
uv sync --extra gpu
```

GPU extra 安装 faster-whisper、CUDA 12 cuBLAS 和 cuDNN 9 运行库，仍需要宿主机可用的 NVIDIA 驱动。首次转写会下载 Whisper 模型；默认 `large-v3-turbo`、`int8_float16`、batch size 8。GPU 内存不足可用 `--batch-size 2`。

只处理已有字幕：`uv sync`。需要 CPU 转写：`uv sync --extra asr`，运行时加 `--device cpu`。Windows 原生环境需要自行配置 CUDA/cuDNN 动态库；推荐 WSL2。CPU 不要求 NVIDIA GPU。

也可以安装成独立命令：

```bash
uv tool install 'biliquill[gpu] @ git+https://github.com/Ceskoslov/biliquill'
```

## 一键运行

先配置你使用的 OpenAI 兼容服务。变量只需设置一次；也支持 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`，`BILIQUILL_*` 优先。

```bash
export BILIQUILL_API_KEY='你的 API key'
export BILIQUILL_BASE_URL='https://api.openai.com/v1'
export BILIQUILL_MODEL='你的模型名称'

uv run biliquill 'https://www.bilibili.com/video/BV13Wf2B8ET6'
```

已经通过 `uv tool install` 安装的用户直接运行 `biliquill URL`。兼容接口使用 `POST <base-url>/chat/completions`；不绑定特定模型，不强制使用供应商专有的结构化输出参数。模型应支持 system/user messages，并能按提示返回 JSON。需要 key 的第三方服务也使用上述 API key 变量；无需鉴权的本机服务可以不设置。

输出默认在 `output/BV13Wf2B8ET6/`，分 P URL 使用独立目录，如 `output/BV…-p2/`：

```text
transcript.md    带原视频时间跳转链接的逐字稿
summary.md       综合总结 + 每段详细摘要
chapters.md      LLM 生成的章节、时间跳转链接和概要
.biliquill/      音频、转写检查点和 LLM 缓存
```

`summary.md` 要求模型区分作者观点与已验证事实，保留论证、案例、限定和重复内容提示。`chapters.md` 的时间戳只能取自输入逐字稿，主题边界仍可能有偏差。

## 本地模型

先启动本地模型服务，并下载或加载你要使用的模型。摘要和综合整理无需云端 API key。

```bash
# 查看 Ollama 已安装模型，再选择其中一个
ollama list
uv run biliquill URL --provider ollama --model YOUR_INSTALLED_MODEL

# LM Studio：先加载模型并启动本地 API 服务
uv run biliquill URL --provider local --model YOUR_LOADED_MODEL

# llama.cpp 等其他本地 OpenAI 兼容服务
uv run biliquill URL --provider local --model YOUR_SERVED_MODEL \
  --base-url http://localhost:8080/v1

# 中文摘要，较小的 Ollama 上下文与输入分段
uv run biliquill URL --provider ollama --model YOUR_INSTALLED_MODEL \
  --notes-language zh --context-size 8192 --chunk-chars 2000 --reduce-chars 4000
```

`--provider ollama` 默认连接 `http://localhost:11434/api/chat`。`--base-url` 可填写服务根地址或以 `/api` 结尾的地址。使用非流式响应，分段摘要启用 JSON 模式，最终综合仍输出 Markdown。`--context-size` 设置 Ollama 的 `num_ctx`，默认为 16384 tokens；较小的上下文需要相应减小分段和综合输入，并为输出保留空间。

`--provider local` 默认连接 `http://localhost:1234/v1`，适用于提供 Chat Completions 接口的本地服务。工具通过 HTTP 调用已运行的模型服务，不会直接加载 GGUF 或 Transformers 权重，也不会自行启动服务器或下载 LLM。

可设置 `BILIQUILL_PROVIDER=ollama` 或 `local` 作为默认后端。本地模式使用独立的 `BILIQUILL_OLLAMA_BASE_URL` / `BILIQUILL_LOCAL_BASE_URL` 配置，不继承云端服务地址与 API key；显式的 `--base-url` 优先。需要鉴权的本地代理可使用 `BILIQUILL_OLLAMA_API_KEY` / `BILIQUILL_LOCAL_API_KEY`。

LLM 请求超时默认为 600 秒，包含首次加载模型的时间，可通过 `--llm-timeout 900` 调整。摘要在配置的本地服务中完成，获取原视频仍需要访问 Bilibili。本地模型需要能够遵循 JSON 指令并返回有效正文。

## 常用选项

```bash
# 输出中文摘要和章节
uv run biliquill URL --notes-language zh

# 指定输出目录与模型
uv run biliquill URL -o ./notes --model your-model

# 已登录浏览器中的字幕；或使用 --cookies /path/to/cookies.txt
uv run biliquill URL --cookies-from-browser firefox

# 分 P
uv run biliquill 'https://www.bilibili.com/video/BVxxxxxxxxxx?p=2'

# 只生成逐字稿，不调用 LLM
uv run biliquill URL --transcript-only

# CPU 转写
uv run biliquill URL --device cpu

# 小显存 / 小上下文模型
uv run biliquill URL --batch-size 2 --chunk-chars 3000 --reduce-chars 6000

# 指定本地 Whisper 模型目录，避免首次联网下载
uv run biliquill URL --asr-model /path/to/faster-whisper-model

# 忽略本工具已有检查点，重新生成
uv run biliquill URL --refresh
```

参数中的 `URL` 替换为实际地址。`--language zh` 默认优先中文字幕，同语言内优先人工字幕，再尝试自动字幕；没有中文时尝试其他可用语言。支持 Bilibili JSON 字幕、SRT 和 WebVTT。弹幕 XML 明确排除，不会当成逐字稿。`--language` 同时作为 ASR 的语言提示。

## 长视频与中断恢复

- 音频按十分钟窗口解码、转写并保存检查点，避免将数小时波形一次性装进内存。窗口交界处可能存在轻微断句偏差。
- 保留带时间标记的 Whisper 解码（`without_timestamps=False`），避免此项目原型中观察到的部分片段漏句问题。
- 默认按约 6000 **字符**切分逐字稿，不是精确 token 计数；较小上下文模型可减小 `--chunk-chars`。
- 每个分段摘要单独缓存。综合输入超过 `--reduce-chars` 时进行多层压缩，不直接截断整片笔记。
- 网络、限流与无效 LLM 输出最多重试三次；HTTP 鉴权/参数错误直接报告。先写出逐字稿，再调用 LLM。
- 再次执行相同命令会复用已完成步骤。模型、接口或输入变化会使对应 LLM 缓存失效；API key 不写入缓存。
- 同一个输出目录一次只运行一个任务。把不同视频放进不同目录；使用 `-o` 时不要让多个任务共享路径。

字幕可能需要 Bilibili 登录。遇到风控或登录限制时，使用自己的 cookies。自动字幕和 ASR 均可能误识别人名、书名、外语与谐音；LLM 总结也需要审阅。工具不负责验证视频中全部学术或历史主张。

## 开发与验证

```bash
uv sync --extra asr
uv run pytest
uv run ruff check .
uv build
```

测试覆盖字幕优先/音频回退、弹幕排除、长文本分段、时间戳约束、LLM 重试/缓存、多层综合和 CLI 产物；测试中的 LLM 使用本地模拟响应，不要求真实 key。真实服务需要用户配置，GPU 路径需在有 GPU 的环境验证。

## 实现参考

- [yt-dlp Python embedding](https://github.com/yt-dlp/yt-dlp#embedding-yt-dlp)
- [faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu)
- [OpenAI Docs: Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [Ollama 原生 Chat API](https://docs.ollama.com/api/chat)

## License

[The Unlicense](LICENSE)。适用于本仓库代码；第三方依赖、模型和视频内容保留各自许可。
