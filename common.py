from __future__ import annotations
import json, re, subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
CLIPS_DIR = Path("clips")
MODEL_NAME = "clip-ViT-B-32"

def video_key(video_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(video_path).stem)

def index_dir(video_path: str) -> Path:
    return DATA_DIR / video_key(video_path)

def load_clip_model():
    # Downloads the open CLIP model weights on first run.
    return SentenceTransformer(MODEL_NAME)

def get_video_info(video_path: str) -> dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        raise RuntimeError("Could not read video FPS.")
    return {"fps": float(fps), "frame_count": float(count), "duration": float(count / fps)}

def extract_frames(video_path: str, interval: float = 2.0):
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    timestamp = 0.0
    while timestamp < info["duration"]:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield timestamp, Image.fromarray(frame)
        timestamp += interval
    cap.release()

def embed_image(model, image: Image.Image) -> np.ndarray:
    return np.asarray(model.encode(image, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)

def embed_text(model, text: str) -> np.ndarray:
    return np.asarray(model.encode(text, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)

def run_ffmpeg_cut(video_path: str, start: float, end: float, output_path: str):
    start = max(0.0, float(start)); end = max(start + 0.1, float(end))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video_path, "-t", f"{end-start:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2500:]}")
    return output_path

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def timestamp_label(seconds: float) -> str:
    seconds = int(max(0, seconds)); h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
