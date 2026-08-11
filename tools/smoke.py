#!/usr/bin/env python3
"""End-to-end smoke test — boots app.py on a synthetic rig and proves the
whole contract: auth flow, snapshot shape, SSE deltas, Prometheus export,
and webhook alert delivery. No GPU required; CI runs exactly this.

    BURNDECK_MOCK_GPUS=8 python tools/smoke.py

Exit 0 on PASS. Uses only the standard library plus the app's own deps.
"""
from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
N_GPUS = int(os.environ.get("BURNDECK_MOCK_GPUS", "8"))

_checks: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _checks.append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"smoke: FAILED at {name!r} {detail}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str, cookie: str = "", redirect: bool = True) -> tuple[int, dict, bytes]:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    opener = urllib.request.build_opener() if redirect else urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, headers={"Cookie": cookie} if cookie else {})
    try:
        with opener.open(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def main() -> None:
    port = free_port()
    sink_hits: list[bytes] = []

    # Webhook sink: alerts must arrive here (config below sets an absurdly low
    # gpu_hot_c so the mock rig trips it immediately).
    class Sink(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            sink_hits.append(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    sink = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Sink)
    threading.Thread(target=sink.serve_forever, daemon=True).start()

    with tempfile.TemporaryDirectory(prefix="burndeck-smoke-") as td:
        cfg = Path(td) / "config.toml"
        cfg.write_text(
            f'[alerts]\nwebhook = "http://127.0.0.1:{sink.server_address[1]}/hook"\n'
            'format = "json"\ngpu_hot_c = 10.0\ncooldown_seconds = 1.0\n'
        )
        env = dict(
            os.environ,
            BURNDECK_MOCK_GPUS=str(N_GPUS),
            BURNDECK_PORT=str(port),
            BURNDECK_HOST="127.0.0.1",
            BURNDECK_DATA_DIR=td,
            BURNDECK_CONFIG=str(cfg),
        )
        proc = subprocess.Popen(
            [sys.executable, str(REPO / "app.py")],
            env=env, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            # ── boot ──
            deadline = time.time() + 30
            health: dict = {}
            while time.time() < deadline:
                if proc.poll() is not None:
                    print(proc.stdout.read().decode(errors="replace"))
                    check("server stays up", False, f"exited {proc.returncode}")
                try:
                    _, _, body = get(f"{base}/healthz")
                    health = json.loads(body)
                    if health.get("gpus"):
                        break
                except OSError:
                    pass
                time.sleep(0.3)
            check("healthz reports mock rig", health.get("ok") is True and health.get("mock") is True)
            check(f"healthz sees {N_GPUS} GPUs", health.get("gpus") == N_GPUS, str(health.get("gpus")))

            # ── auth ──
            code, headers, _ = get(f"{base}/", redirect=False)
            loc = next((v for k, v in headers.items() if k.lower() == "location"), "")
            check("unauthenticated / redirects to login",
                  code == 302 and "/login" in loc, f"code={code} loc={loc!r}")
            code, _, _ = get(f"{base}/api/snapshot")
            check("unauthenticated api returns 401", code == 401, f"code={code}")

            user, _, pw = (Path(td) / ".auth-generated").read_text().strip().partition(":")
            check("first boot generated credentials", bool(user and pw))
            req = urllib.request.Request(
                f"{base}/api/auth",
                data=json.dumps({"login": user, "password": pw}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                cookie = r.headers.get("Set-Cookie", "").split(";")[0]
            check("login issues session cookie", cookie.startswith("burndeck_session="))

            # ── snapshot contract ──
            _, _, body = get(f"{base}/api/snapshot", cookie=cookie)
            snap = json.loads(body)
            gpus = snap.get("gpus", [])
            check(f"snapshot carries {N_GPUS} GPUs", len(gpus) == N_GPUS, str(len(gpus)))
            g = gpus[-1]
            for key in ("utilization_gpu_pct", "memory_used_mib", "memory_total_mib",
                        "temperature_c", "power_draw_w", "power_limit_w", "fan_pct",
                        "pcie_gen", "pcie_width", "pcie_aer", "processes", "history", "tps"):
                check(f"gpu[{g['index']}].{key} present", key in g)
            models = snap.get("models", [])
            check("mock models served", len(models) >= 1, str(len(models)))
            m = models[0]
            check("model spans all GPUs (TP)", m.get("gpu_indices") == list(range(N_GPUS)))
            check("model has live TPS", m.get("total_tps", 0) > 0, str(m.get("total_tps")))
            check("model intel attached", bool(m.get("intel", {}).get("family")))
            totals = snap.get("totals", {})
            check("totals aggregate power", abs(
                totals.get("power_draw_w", 0) - sum(x["power_draw_w"] for x in gpus)) < 1.0)

            # ── SSE stream: full snapshot, then a keyed delta ──
            req = urllib.request.Request(f"{base}/api/stream", headers={"Cookie": cookie})
            with urllib.request.urlopen(req, timeout=10) as r:
                first = json.loads(r.readline().decode()[len("data:"):])
                check("stream opens with full snapshot", len(first.get("gpus", [])) == N_GPUS)
                delta = None
                deadline = time.time() + 5
                while time.time() < deadline:
                    line = r.readline().decode().strip()
                    if line.startswith("data:"):
                        delta = json.loads(line[len("data:"):])
                        break
                check("stream sends update frames", bool(delta) and delta.get("type") == "update")

            # ── Prometheus ──
            _, _, body = get(f"{base}/metrics")
            text = body.decode()
            util_lines = [l for l in text.splitlines() if l.startswith("burndeck_gpu_utilization_percent")]
            check(f"/metrics exports {N_GPUS} GPU gauges", len(util_lines) == N_GPUS, str(len(util_lines)))
            check("/metrics exports model health", "burndeck_model_healthy" in text)

            # ── webhook alert (gpu_hot_c=10 guarantees a firing) ──
            deadline = time.time() + 10
            while time.time() < deadline and not sink_hits:
                time.sleep(0.3)
            check("webhook alert delivered", bool(sink_hits))
            payload = json.loads(sink_hits[0])
            check("alert payload well-formed",
                  payload.get("source") == "burndeck" and "GPU" in payload.get("text", ""))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            sink.shutdown()

    print(f"smoke: PASS — {len(_checks)} checks")


if __name__ == "__main__":
    main()
