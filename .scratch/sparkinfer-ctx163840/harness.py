#!/usr/bin/env python3
"""Measure SparkInfer decode speed: text (DSpark-eligible) vs image (AR only).

Client-side streaming timing, stdlib only. Arms:
  text-dspark  greedy text requests      -> speculation eligible
  text-ar      temperature 0.7 text      -> sampling, so AR (control for the image arm)
  image        greedy image requests     -> vision always stays on AR

Per request it records TTFT, post-first-token decode tok/s (client clock) and the
server's own ttft_ms / generation_ms / decode_tps, plus /metrics speculation
counters before and after. Every arm runs sequentially, one active request at a
time, because DSpark is declined whenever another request overlaps.

Usage:
  python3 harness.py --base-url http://127.0.0.1:8192 \
      --image /mnt/d/Code/MJ-Project/ai-model-nvfp4/.scratch/sparkinfer-ctx163840/image-text.png \
      --max-tokens 192 --out results.json
"""
import argparse
import base64
import json
import statistics
import sys
import time
import urllib.request

SPEC_KEYS = (
    "sparkinfer_speculative_runs_total",
    "sparkinfer_speculative_tokens_total",
    "sparkinfer_speculative_handoffs_total",
    "sparkinfer_prefix_cache_hits_total",
)


def http_json(url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def metrics(base):
    req = urllib.request.Request(base + "/metrics")
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8", "replace")
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] in SPEC_KEYS:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def stream_chat(base, body, timeout=600):
    """POST a streaming chat completion; return timing and text."""
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    pieces = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    pieces.append(piece)
    t_end = time.perf_counter()
    text = "".join(pieces)
    ct = (usage or {}).get("completion_tokens") or len(pieces)
    ttft = (t_first - t0) if t_first else float("nan")
    gen = t_end - t_first if t_first else float("nan")
    return {
        "ttft_ms": round(ttft * 1000, 1),
        "wall_s": round(t_end - t0, 3),
        "completion_tokens": ct,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "decode_tps_client": round((ct - 1) / gen, 1) if gen and gen > 0 and ct > 1 else None,
        "server_ttft_ms": (usage or {}).get("ttft_ms"),
        "server_generation_ms": (usage or {}).get("generation_ms"),
        "server_decode_tps": (usage or {}).get("decode_tps"),
        "text": text,
    }


TEXT_PROMPTS = [
    ("chat", "Explain in one short paragraph why the daytime sky is blue."),
    ("json", "Return a JSON object with keys id, name, tags for a fictional product. Output only JSON, no prose."),
    ("code", "Write a Python function that reverses a singly linked list, with a short docstring."),
    ("math", "A train leaves at 14:20 and travels 210 km at 84 km/h. Show the arrival time step by step."),
    ("list", "List five differences between TCP and UDP, one per line."),
    ("prose", "Summarize the water cycle in about 120 words."),
    ("json2", "Return a JSON array of 5 objects, each with keys city and population, for major Japanese cities. Only JSON."),
    ("code2", "Write a bash one-liner that lists the 10 largest files under /var, then explain it briefly."),
]

IMAGE_PROMPTS = [
    ("img_code", "Read the clearance code shown in this image. Reply with only the code."),
    ("img_shapes", "Name the three shapes in this image and their colors, one per line as 'shape: color'."),
    ("img_describe", "Describe this image in one short sentence."),
]


def image_part(path):
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}


def run_arm(base, name, tasks, max_tokens, model, image_path=None, temperature=None):
    rows = []
    before = metrics(base)
    for label, prompt in tasks:
        content = [{"type": "text", "text": prompt}]
        if image_path:
            content.append(image_part(image_path))
        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if temperature is None:
            body["temperature"] = 0
        else:
            body["temperature"] = temperature
        t = stream_chat(base, body)
        t["label"] = label
        t["prompt"] = prompt
        rows.append(t)
        print("  %-12s ttft %8.1f ms  decode %7s tok/s  ct %s  %.1fs"
              % (label, t["ttft_ms"], t["decode_tps_client"], t["completion_tokens"], t["wall_s"]),
              flush=True)
    after = metrics(base)
    delta = {k: round(after.get(k, 0) - before.get(k, 0), 1) for k in SPEC_KEYS}
    decodes = [r["decode_tps_client"] for r in rows if r["decode_tps_client"]]
    ttfts = [r["ttft_ms"] for r in rows]
    summary = {
        "arm": name,
        "requests": len(rows),
        "decode_median": round(statistics.median(decodes), 1) if decodes else None,
        "decode_mean": round(statistics.mean(decodes), 1) if decodes else None,
        "decode_min": min(decodes) if decodes else None,
        "decode_max": max(decodes) if decodes else None,
        "ttft_median_ms": round(statistics.median(ttfts), 1),
        "metrics_delta": delta,
        "rows": rows,
    }
    print("  => %s: decode median %s tok/s, ttft median %.1f ms, metrics delta %s"
          % (name, summary["decode_median"], summary["ttft_median_ms"], delta), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8192")
    ap.add_argument("--image")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--model", default="Qwen3.8-27B-NVFP4-DSpark")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--arms", default="text-dspark,text-ar,image")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    info = http_json(base + "/v1/info")
    print("server /v1/info: %s" % json.dumps(info), flush=True)

    out = {"info": info, "max_tokens": args.max_tokens, "arms": []}
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]

    for arm in wanted:
        print("[arm %s]" % arm, flush=True)
        if arm == "text-dspark":
            out["arms"].append(run_arm(base, arm, TEXT_PROMPTS, args.max_tokens, args.model))
        elif arm == "text-ar":
            out["arms"].append(run_arm(base, arm, TEXT_PROMPTS[:4], args.max_tokens, args.model, temperature=0.7))
        elif arm == "image":
            if not args.image:
                sys.exit("--image is required for the image arm")
            out["arms"].append(run_arm(base, arm, IMAGE_PROMPTS, args.max_tokens, args.model, image_path=args.image))
        else:
            sys.exit("unknown arm: %s" % arm)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()