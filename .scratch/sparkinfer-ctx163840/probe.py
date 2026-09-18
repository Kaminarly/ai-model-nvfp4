#!/usr/bin/env python3
"""Repeatability probe: N identical greedy repeats of one prompt.

Used to tell "cleanly fast" from "intermittently degraded" near the ctx cliff:
a healthy DSpark run gives the same decode rate every time (speculation engaged,
counter +1 per request); a run that is short on scratch VRAM degrades some
requests and not others.

  python3 probe.py --model Qwen3.8-27B-NVFP4-DSpark --prompt code --repeats 5
"""
import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python implementation of a binary search tree with insert, search and in-order traversal, with docstrings.",
    "prose": "Summarize the water cycle in about 400 words.",
}


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=60) as r:
        text = r.read().decode("utf-8", "replace")
    out = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 2 and p[0] in ("sparkinfer_speculative_runs_total",
                                    "sparkinfer_speculative_tokens_total"):
            out[p[0]] = float(p[1])
    return out


def once(base, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(base + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    before = metrics(base)
    t0 = time.perf_counter()
    first = None
    n = 0
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
                    if first is None:
                        first = time.perf_counter()
                    n += 1
    end = time.perf_counter()
    after = metrics(base)
    ct = (usage or {}).get("completion_tokens") or n
    gen = end - first
    return {
        "decode": round((ct - 1) / gen, 1) if gen > 0 and ct > 1 else None,
        "ttft_ms": round((first - t0) * 1000, 1) if first else None,
        "ct": ct,
        "spec_runs": after.get("sparkinfer_speculative_runs_total", 0) - before.get("sparkinfer_speculative_runs_total", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8192")
    ap.add_argument("--model", default="Qwen3.8-27B-NVFP4-DSpark")
    ap.add_argument("--prompt", default="code", choices=sorted(PROMPTS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    info = json.loads(urllib.request.urlopen(base + "/v1/info", timeout=30).read().decode())
    print("probe prompt=%s repeats=%d max_tokens=%d max_context=%s"
          % (args.prompt, args.repeats, args.max_tokens, info.get("max_context")), flush=True)
    rows = []
    for i in range(args.repeats):
        r = once(base, args.model, PROMPTS[args.prompt], args.max_tokens)
        rows.append(r)
        print("  #%d decode=%-7s ttft=%-8s ct=%-4s spec_runs=%s"
              % (i + 1, r["decode"], r["ttft_ms"], r["ct"], r["spec_runs"]), flush=True)
    vals = [r["decode"] for r in rows if r["decode"]]
    print("  => min %s  median %s  max %s" % (min(vals), round(statistics.median(vals), 1), max(vals)), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"info": info, "prompt": args.prompt, "max_tokens": args.max_tokens, "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()