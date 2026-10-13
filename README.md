# Multimodal Video Search with LLM Validation

Search long videos in natural language, validate likely scenes with a local vision-language model, and cut matching segments into MP4 clips. Example: search a five-hour market video for **“a girl selling apples.”**

## Architecture

```text
Video → sampled frames → CLIP embeddings → FAISS timestamp index
                                           ↓
Query → LangChain query expansion → retrieval candidates
                                           ↓
                       LangGraph → Qwen2.5-VL validation
                                           ↓
                             temporal padding → FFmpeg clips
```

## Stack

- Python, LangChain, LangGraph
- `clip-ViT-B-32` through Sentence Transformers for text/image retrieval
- FAISS for local vector similarity search
- Qwen2.5-VL through Ollama for local vision validation
- OpenCV and FFmpeg for frame extraction and video cutting

No paid inference API key is needed. First-time model downloads need internet access. CPU execution works but can be slow; a supported GPU helps.

## Setup (Ubuntu)

```bash
sudo apt update
sudo apt install ffmpeg python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Ollama, then download the local vision model:

```bash
ollama pull qwen2.5vl:3b
ollama list
```

## Run

Index the video once:

```bash
python index_video.py "/path/to/market_video.mp4" --interval 2
```

Search and cut matching segments:

```bash
python search_graph.py "/path/to/market_video.mp4" "a girl selling apples"
```

Increase retrieval candidates or use a denser index for short events:

```bash
python search_graph.py market_video.mp4 "a person handing apples to a customer" --top-k 80 --vlm qwen2.5vl:3b
python index_video.py market_video.mp4 --interval 1
```

Generated clips are saved in `clips/`; the reusable frame index is saved in `data/<video-name>/`.

## LangGraph flow

```text
START → expand_query → retrieve → validate
                              ├─ match → refine → cut → END
                              ├─ no match, attempts left → retry → retrieve
                              └─ no match after 3 attempts → END
```

## Limitations

- This is a starter prototype; verify results on your own footage.
- Sampling every 2 seconds can miss brief actions. Use `--interval 1` for better recall, at higher indexing cost.
- Visual embeddings can find apples and people, but cannot reliably prove a transaction or action. The VLM verifier helps but can still make mistakes.
- Current temporal refinement is approximate: it uses the validator's proposed offsets and adds 1.5 seconds of padding. It does not yet do dense frame-by-frame boundary detection.
- Audio/transcription search is not included yet. Whisper can be added as a second retrieval channel.
- If you replace a video while keeping its filename, delete its old `data/<video-name>/` index and re-index it.

## Troubleshooting

- `No index found`: run `index_video.py` first.
- `ffmpeg not found`: install FFmpeg and check `ffmpeg -version`.
- Ollama/model errors: ensure Ollama is running and `ollama list` shows `qwen2.5vl:3b`.
- No results: use a broader query, increase `--top-k`, or re-index with a smaller interval.
