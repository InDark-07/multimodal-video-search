from __future__ import annotations
import argparse, base64, io, json, re
from pathlib import Path
from typing import TypedDict, Any
import cv2, faiss
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from common import CLIPS_DIR, embed_text, index_dir, load_clip_model, load_json, run_ffmpeg_cut, timestamp_label
from video_source import resolve_video


class SearchState(TypedDict, total=False):
    video_path: str; query: str; variants: list[str]; top_k: int; attempt: int
    candidates: list[dict[str, Any]]; validated: list[dict[str, Any]]; clips: list[str]; error: str

def parse_json(text: str, fallback):
    try: return json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]|\{.*\}", text, re.S)
        if match:
            try: return json.loads(match.group(0))
            except Exception: pass
    return fallback

def frame_at(video_path: str, seconds: float):
    cap = cv2.VideoCapture(video_path); cap.set(cv2.CAP_PROP_POS_MSEC, max(0, seconds)*1000)
    ok, frame = cap.read(); cap.release()
    if not ok: return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO(); image.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def llm_for(model_name: str): return ChatOllama(model=model_name, temperature=0)

def expand_query(state: SearchState, llm):
    prompt = f'''Create 3 concise visual search variants for this video query. Keep the meaning and include visible context/action synonyms. Return only a JSON array of strings. Query: {state["query"]}'''
    try:
        variants = parse_json(llm.invoke([HumanMessage(content=prompt)]).content, [])
        if not isinstance(variants, list): variants = []
        variants = [str(x).strip() for x in variants if str(x).strip()][:3]
    except Exception: variants = []
    return {"variants": [state["query"], *variants]}

def retrieve(state: SearchState, clip_model):
    folder = index_dir(state["video_path"]); ip, mp = folder/"frames.faiss", folder/"metadata.json"
    if not ip.exists() or not mp.exists(): return {"error": f"No index found at {folder}. Run index_video.py first."}
    index = faiss.read_index(str(ip)); metadata = load_json(mp); rows = metadata["frames"]
    top_k = min(int(state.get("top_k", 50)), index.ntotal); scored = {}
    for query in state.get("variants") or [state["query"]]:
        q = embed_text(clip_model, query).reshape(1, -1).astype("float32")
        scores, ids = index.search(q, top_k)
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0: scored[int(idx)] = max(float(score), scored.get(int(idx), -1.0))
    ranked = sorted(scored.items(), key=lambda p: p[1], reverse=True); interval = float(metadata["interval"]); windows = []
    for idx, score in ranked:
        t = float(rows[idx]["timestamp"]); start = max(0, t-interval); end = min(float(metadata["duration"]), t+interval)
        found = next((w for w in windows if start <= w["end"]+4 and end >= w["start"]-4), None)
        if found:
            found["start"] = min(found["start"], start); found["end"] = max(found["end"], end)
            found["score"] = max(found["score"], score); found["hits"].append(t)
        else: windows.append({"start": start, "end": end, "score": score, "hits": [t]})
    windows.sort(key=lambda w: w["score"], reverse=True)
    candidates = [{"start": round(w["start"],2), "end": round(w["end"],2), "score": round(w["score"],4), "hits": w["hits"]} for w in windows[:min(15, max(5, int(state.get("top_k",50)/5)))]]
    return {"candidates": candidates}

def validate(state: SearchState, llm):
    validated = []
    for candidate in state.get("candidates", []):
        start, end = candidate["start"], candidate["end"]
        times = sorted(set([start, (start+end)/2, end])); content = [{"type":"text", "text": (
            f'User query: {state["query"]}\nCandidate timestamps: {timestamp_label(start)} to {timestamp_label(end)}. '
            'Judge only visible evidence. Does the sequence support the requested action, not merely contain similar objects? '
            'Return JSON only: {"match": true/false, "confidence": 0.0, "reason": "brief evidence", '
            '"event_start_offset": 0.0, "event_end_offset": 1.0}. Offsets are seconds relative to candidate start. '
            'Do not infer selling merely because a person is near fruit.'
        )}]
        for t in times:
            img = frame_at(state["video_path"], t)
            if img is not None: content.append({"type":"image_url", "image_url":{"url":to_data_url(img)}})
        if len(content) == 1: continue
        try:
            response = llm.invoke([SystemMessage(content="You are a cautious video-search verifier. Never invent visual evidence."), HumanMessage(content=content)])
            verdict = parse_json(response.content, {})
        except Exception as exc: verdict = {"match":False,"confidence":0,"reason":str(exc)}
        try: confidence = float(verdict.get("confidence",0))
        except (ValueError, TypeError): confidence = 0
        if verdict.get("match") is True and confidence >= 0.55:
            a = max(0.0, float(verdict.get("event_start_offset",0)))
            b = max(a+1.0, float(verdict.get("event_end_offset",end-start)))
            validated.append({**candidate, "start":max(0,start+a), "end":min(end,start+b), "confidence":confidence, "reason":str(verdict.get("reason",""))})
    return {"validated": validated}

def refine(state: SearchState):
    result = []
    for item in state.get("validated", []):
        item = dict(item); item["start"] = max(0, item["start"]-1.5); item["end"] += 1.5; result.append(item)
    return {"validated": result}

def cut(state: SearchState):
    outputs = []; CLIPS_DIR.mkdir(exist_ok=True)
    for i, item in enumerate(state.get("validated", []), 1):
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", state["query"]).strip("_")[:40] or "search"
        output = CLIPS_DIR / f"{safe}_{i}_{int(item['start'])}-{int(item['end'])}.mp4"
        try: run_ffmpeg_cut(state["video_path"], item["start"], item["end"], str(output)); item["output"] = str(output); outputs.append(str(output))
        except Exception as exc: item["cut_error"] = str(exc)
    return {"clips": outputs, "validated": state.get("validated", [])}

def route_after_validation(state: SearchState):
    if state.get("validated"): return "refine"
    if state.get("attempt",1) < 3: return "retry"
    return "end"

def build_graph(video_path: str, query: str, top_k: int, vlm_name: str):
    llm = llm_for(vlm_name); clip_model = load_clip_model()
    graph = StateGraph(SearchState)
    graph.add_node("expand_query", lambda s: expand_query(s,llm))
    graph.add_node("retrieve", lambda s: retrieve(s,clip_model))
    graph.add_node("validate", lambda s: validate(s,llm))
    graph.add_node("refine", refine); graph.add_node("cut", cut)
    graph.add_node("retry", lambda s: {"attempt":s.get("attempt",1)+1, "top_k":min(200,s.get("top_k",top_k)*2)})
    graph.add_edge(START,"expand_query"); graph.add_edge("expand_query","retrieve"); graph.add_edge("retrieve","validate")
    graph.add_conditional_edges("validate",route_after_validation,{"refine":"refine","retry":"retry","end":END})
    graph.add_edge("retry","retrieve"); graph.add_edge("refine","cut"); graph.add_edge("cut",END)
    return graph.compile(), {"video_path":video_path,"query":query,"top_k":top_k,"attempt":1}


def main():
    parser = argparse.ArgumentParser(
        description="Search a local or YouTube video and cut matching clips."
    )
    parser.add_argument("video", help="Local video path or YouTube URL")
    parser.add_argument("query", help="Natural-language scene description")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--vlm",
        default="qwen2.5vl:3b",
        help="Ollama vision model name",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive")

    # Resolve URL to the same local video used by the indexing script.
    video_path = resolve_video(args.video)

    graph, initial = build_graph(
        video_path,
        args.query,
        args.top_k,
        args.vlm,
    )

    result = graph.invoke(initial)

    if result.get("error"):
        raise RuntimeError(result["error"])

    matches = result.get("validated", [])

    if not matches:
        print(
            "No confident matches. Try a broader query "
            "or re-index at --interval 1."
        )
        return

    print(f"\nFound {len(matches)} validated segment(s):")

    for i, match in enumerate(matches, 1):
        print(
            f"{i}. {timestamp_label(match['start'])} - "
            f"{timestamp_label(match['end'])} | "
            f"confidence={match.get('confidence', 0):.2f} | "
            f"{match.get('reason', '')}\n"
            f"   Clip: {match.get('output', 'not cut')}"
        )


if __name__ == "__main__":
    main()

