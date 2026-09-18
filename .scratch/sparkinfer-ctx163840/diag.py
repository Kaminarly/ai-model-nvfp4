#!/usr/bin/env python3
"""Fine-grained A/B for SparkInfer: greedy (DSpark) vs temperature 0.7 (AR).

Same two prompts, same max_tokens, both modes, sequential. Records per-chunk
arrival times so the decode rate can be split into segments (to see whether a
speculative run warms up or is uniformly slow), plus /metrics speculation deltas
per request. Printing is compact so the two ctx runs can be diffed.

Usage:
  python3 diag.py --base-url http://127.0.0.1:8192 --label ctx163840 \
      --max-tokens 512 --out diag-ctx163840.json
"""
import argparse
import json
import time
import urllib.request

SPEC = ("sparkinfer_speculative_runs_total", "sparkinfer_speculative_tokens_total")

PROMPTS = [
    ("prose", "Summarize the water cycle in about 400 words."),
    ("code", "Write a Python implementation of a binary search tree with insert, search and in-order traversal, with docstrings."),
]


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=60) as r:
        text = r.read().decode("utf-8", "replace")
    out = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 2 and p[0] in SPEC:
            out[p[0]] = float(p[1])
    return out


def run(base, prompt, max_tokens, temperature, model):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": temperature,
    }
    req = urllib.request.Request(base + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    before = metrics(base)
    t0 = time.perf_counter()
    stamps = []
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
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
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if (d.get("content") or "") + (d.get("reasoning_content") or ""):
                    stamps.append(time.perf_counter())
    t_end = time.perf_counter()
    after = metrics(base)
    ct = (usage or {}).get("completion_tokens") or len(stamps)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in SPEC}

    # segmented decode rate, using chunk count as the token proxy
    segs = []
    bounds = [0, 128, 256, 384, 512]
    for lo, hi in zip(bounds, bounds[1:]):
        if len(stamps) > hi:
            segs.append(round((hi - lo) / (stamps[hi - 1] - stamps[lo]), 1) if hi - lo > 1 else None)
        elif len(stamps) > lo + 1:
            segs.append(round((len(stamps) - lo - 1) / (stamps[-1] - stamps[lo]), 1))
        else:
            segs.append(None)
    ttft = (stamps[0] - t0) * 1000 if stamps else None
    gen = stamps[-1] - stamps[0] if len(stamps) > 1 else None
    return {
        "chunks": len(stamps),
        "completion_tokens": ct,
        "ttft_ms": round(ttft, 1) if ttft else None,
        "wall_s": round(t_end - t0, 3),
        "decode_tps_client": round((ct - 1) / gen, 1) if gen else None,
        "segments_tps": segs,
        "spec_delta": delta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8192")
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--model", default="Qwen3.8-27B-NVFP4-DSpark")
    ap.add_argument("--out", default="diag.json")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    info = json.loads(urllib.request.urlopen(base + "/v1/info", timeout=30).read().decode())
    result = {"label": args.label, "info": info, "max_tokens": args.max_tokens, "requests": []}
    print("label=%s max_context=%s" % (args.label, info.get("max_context")), flush=True)
    for name, prompt in PROMPTS:
        for mode, temp in (("dspark-greedy", 0), ("ar-temp0.7", 0.7)):
            r = run(base, prompt, args.max_tokens, temp, args.model)
            r["prompt"] = name
            r["mode"] = mode
            result["requests"].append(r)
            print("  %-6s %-12s ct=%-4s chunks=%-4s ttft=%-7s decode=%-6s segs=%s spec=%s"
                  % (name, mode, r["completion_tokens"], r["chunks"], r["ttft_ms"],
                     r["decode_tps_client"], r["segments_tps"], r["spec_delta"]), flush=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()