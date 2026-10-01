"""Small local HTTP server for the nested 12B (standard library only).

| Endpoint | Job |
|---|---|
| GET  /v1/tiers            | tiers with size, RAM/stream split, measured scores |
| POST /v1/chat/completions | OpenAI-style body, plus "bits" (4/8/16) and "width" (1.0/0.9/0.85) |

One model in memory, one request at a time. Only the last user message is used
(no multi-turn history yet). The response carries the tier and timing in "nest".

usage: serve.py [--model models/g12-nest/model.nest] [--port 8765] [--ram 12] [--scores results/tier-scores.json]
"""
import argparse
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx

from nest.runtime import NestedGemma

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/g12-nest/model.nest")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--ram", type=float, default=12.0)
ap.add_argument("--scores", default="results/tier-scores.json")
args = ap.parse_args()

mx.set_cache_limit(1024**3)
M = NestedGemma.load(args.model, ram_budget_gb=args.ram)
SCORES = json.load(open(args.scores)) if os.path.exists(args.scores) else None
print(f"loaded in {M.load_s:.1f}s, {M.resident_bytes / 1e9:.1f} GB resident, port {args.port}", flush=True)


def last_user_text(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            return "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
    raise ValueError("no user message")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/tiers":
            return self._send(200, {"tiers": M.tiers(SCORES), "current": {"bits": M.bits, "width": M.width}})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            text = last_user_text(req["messages"])
            out = M.generate(text, max_tokens=int(req.get("max_tokens", 256)),
                             bits=req.get("bits"), width=req.get("width"))
        except Exception as e:      # report errors to the client, keep serving
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})
        st = out["stats"]
        self._send(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
            "created": int(time.time()), "model": f"g12-nest:{st['tier']}",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out["text"]},
                         "finish_reason": "stop" if st["tokens"] < int(req.get("max_tokens", 256)) else "length"}],
            "usage": {"prompt_tokens": st["prompt_tokens"], "completion_tokens": st["tokens"],
                      "total_tokens": st["prompt_tokens"] + st["tokens"]},
            "nest": st,
        })

    def log_message(self, fmt, *a):
        print(fmt % a, flush=True)


HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
