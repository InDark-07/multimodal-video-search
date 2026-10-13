
import argparse
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import re
import subprocess
import sys
from video_source import resolve_video

from common import (
    extract_frames,
    embed_image,
    index_dir,
    load_clip_model,
    save_json,
    video_key,
    get_video_info,
)

def is_youtube_url(source):
    host = urlparse(source).netloc.lower()
    return host in {
        "youtube.com", "www.youtube.com", "m.youtube.com",
        "youtu.be", "www.youtu.be"
    }

def resolve_video(source):
    """Download a YouTube video locally, or validate a local file."""
    if not is_youtube_url(source):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        return str(path.resolve())

    download_dir = Path("data/youtube_downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "yt_dlp",
            "--no-playlist",
            "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--output", str(download_dir / "%(id)s.%(ext)s"),
            "--print", "after_move:filepath",
            source,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "YouTube download failed. Check the URL, network, "
            "yt-dlp version and video access.\n" + result.stderr[-2500:]
        )

    output = result.stdout.strip().splitlines()
    if not output:
        raise RuntimeError("yt-dlp did not return the downloaded video path.")

    video_path = Path(output[-1])
    if not video_path.is_file():
        raise FileNotFoundError(f"Downloaded video not found: {video_path}")

    print(f"Downloaded video: {video_path}")
    return str(video_path.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Index a local or YouTube video."
    )
    parser.add_argument("video", help="Local video path or YouTube URL")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    video_path = resolve_video(args.video)
    out_dir = index_dir(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP model...")
    model = load_clip_model()
    info = get_video_info(video_path)

    rows = []
    vectors = []

    total = max(1, int(info["duration"] / args.interval))
    from tqdm import tqdm

    for timestamp, image in tqdm(
        extract_frames(video_path, args.interval),
        total=total,
        desc="Sampling/embedding",
    ):
        vectors.append(embed_image(model, image))
        rows.append({
            "timestamp": round(timestamp, 3),
            "frame_id": len(rows),
        })

    if not vectors:
        raise RuntimeError("No frames extracted. Check the video or codec.")

    import faiss
    import numpy as np

    matrix = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(out_dir / "frames.faiss"))

    save_json(out_dir / "metadata.json", {
        "video_path": video_path,
        "video_key": video_key(video_path),
        "duration": info["duration"],
        "fps": info["fps"],
        "interval": args.interval,
        "frames": rows,
    })

    print(f"Indexed {len(rows)} frames.")
    print(f"Index saved at: {out_dir.resolve()}")


if __name__ == "__main__":
    main()