
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def is_youtube_url(source: str) -> bool:
    host = (urlparse(source).hostname or "").lower()
    return host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }


def resolve_video(source: str) -> str:
    """Return the local path for a local video or accessible YouTube URL."""
    if not is_youtube_url(source):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        return str(path)

    download_dir = Path("data/youtube_downloads").resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    # Derive the expected cached filename from the YouTube video ID.
    parsed = urlparse(source)
    video_id = (
        parsed.path.rstrip("/").split("/")[-1]
        if parsed.hostname and parsed.hostname.lower() in {
            "youtu.be", "www.youtu.be"
        }
        else __import__("urllib.parse", fromlist=["parse_qs"])
        .parse_qs(parsed.query).get("v", [""])[0]
    )

    if not video_id:
        raise ValueError("Could not extract the YouTube video ID.")

    # Reuse a previously downloaded video.
    cached = list(download_dir.glob(f"{video_id}.*"))
    cached = [
        p for p in cached
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    ]
    if cached:
        return str(cached[0])

    result = subprocess.run(
        [
            sys.executable, "-m", "yt_dlp",
            "--no-playlist",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", str(download_dir / "%(id)s.%(ext)s"),
            source,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "YouTube download failed:\n" + result.stderr[-3000:]
        )

    cached = [
        p for p in download_dir.glob(f"{video_id}.*")
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    ]
    if not cached:
        raise RuntimeError(
            "Download finished but no video file was found. "
            "Check the yt-dlp output."
        )

    return str(cached[0])