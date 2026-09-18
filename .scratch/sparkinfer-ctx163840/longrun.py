#!/usr/bin/env python3
"""Long-generation speed probe.

Sends one greedy streaming request with a long output budget and reports the
decode rate per segment (default 4096 tokens), so the decay over a long
generation is visible: DSpark stops at attention tier boundaries, so the tail
runs at AR speed. While streaming, a background poller samples /v1/capacity so
the per-request KV reservation can be read off free_kv_blocks.

  python3 longrun.py --max-tokens 32768 --segment 4096 --out longrun-maxtok32k.json
"""
import argparse
import json
import threading
import time
import urllib.request

PROMPTS = {
    "long": (
        "Write a complete, runnable Python implementation of a personal finance tracker: "
        "an argparse CLI, SQLite storage, CSV import/export, monthly reports, and unit tests. "
        "Include the full source of every module and explain each design decision as you go. "
        "Be thorough: do not abbreviate, do not use placeholders like '...', and do not stop "
        "until the implementation is complete."
    ),
    "code": "Write a Python implementation of a binary search tree with insert, search and in-order traversal, with docstrings.",
    "prose": "Summarize the water cycle in about 400 words.",
}
PROMPT = PROMPTS["long"]


def jget(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
        text = r.read().decode("utf-8", "replace")
    out = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 2 and p[0] in ("sparkinfer_speculative_runs_total",
                                    "sparkinfer_speculative_tokens_total",
                                    "sparkinfer_speculative_handoffs_total",
                                    "sparkinfer_speculative_tier_stops_total"):
            out[p[0]] = float(p[1])
    return out


class Poller(threading.Thread):
    """Sample /v1/capacity while a request is in flight."""

    def __init__(self, base, interval=1.5):
        super().__init__(daemon=True)
        self.base = base
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            row = {"t": time.time()}
            try:
                cap = jget(self.base + "/v1/capacity", timeout=10)
                for k in ("active_requests", "free_kv_blocks", "accepting_requests"):
                    row[k] = cap.get(k)
            except Exception as exc:                                  # noqa: BLE001
                row["error"] = str(exc)[:60]
            self.samples.append(row)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8192")
    ap.add_argument("--model", default="Qwen3.8-27B-NVFP4-DSpark")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--segment", type=int, default=4096)
    ap.add_argument("--preset", default="long", choices=sorted(PROMPTS))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    prompt = args.prompt or PROMPTS[args.preset]

    info = jget(base + "/v1/info")
    cap0 = jget(base + "/v1/capacity")
    before = metrics(base)
    print("max_context=%s  free_kv_blocks before=%s  active=%s"
          % (info.get("max_context"), cap0.get("free_kv_blocks"), cap0.get("active_requests")), flush=True)

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(base + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})

    poller = Poller(base)
    poller.start()

    t0 = time.perf_counter()
    stamps = []
    finish = None
    usage = None
    next_mark = args.segment
    with urllib.request.urlopen(req, timeout=3600) as r:
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
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                if (d.get("content") or "") + (d.get("reasoning_content") or ""):
                    stamps.append(time.perf_counter())
            while len(stamps) >= next_mark:
                lo = next_mark - args.segment
                rate = args.segment / (stamps[next_mark - 1] - stamps[lo])
                print("  tokens %6d-%6d : %6.1f tok/s   (elapsed %6.1f s)"
                      % (lo + 1, next_mark, rate, stamps[next_mark - 1] - t0), flush=True)
                next_mark += args.segment
    end = time.perf_counter()
    poller.stop()
    after = metrics(base)

    n = len(stamps)
    gen = stamps[-1] - stamps[0] if n > 1 else 0.0
    segments = []
    for lo in range(0, n - 1, args.segment):
        hi = min(lo + args.segment, n)
        if hi - lo > 1:
            segments.append({"from": lo + 1, "to": hi,
                             "tps": round((hi - lo - 1) / (stamps[hi - 1] - stamps[lo]), 1)})
    if n > 1 and n % args.segment:
        lo = (n // args.segment) * args.segment
        if n - lo > 1:
            segments.append({"from": lo + 1, "to": n,
                             "tps": round((n - lo - 1) / (stamps[-1] - stamps[lo]), 1)})

    result = {
        "info": info,
        "max_tokens_requested": args.max_tokens,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens") or n,
        "finish_reason": finish,
        "ttft_ms": round((stamps[0] - t0) * 1000, 1) if n else None,
        "wall_s": round(end - t0, 1),
        "decode_tps_overall": round((n - 1) / gen, 1) if gen > 0 else None,
        "segments": segments,
        "capacity_before": cap0,
        "capacity_samples": poller.samples,
        "spec_delta": {k: after.get(k, 0) - before.get(k, 0)
                       for k in ("sparkinfer_speculative_runs_total",
                                 "sparkinfer_speculative_tokens_total")},
    }
    print("  === completion_tokens=%s finish_reason=%s wall=%ss overall=%.1f tok/s"
          % (result["completion_tokens"], finish, result["wall_s"], result["decode_tps_overall"] or 0), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()