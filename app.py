#!/usr/bin/env python3
"""burndeck — cinematic GPU + LLM telemetry for multi-GPU rigs.

Single-file FastAPI server: NVML (with an nvidia-smi fallback) for per-GPU /
per-process telemetry, OpenAI-compatible endpoint probes for model TPS, a
zero-config sniffer for anything holding weights on a GPU, and a snapshot
JSON + delta SSE stream feeding the single-page UI in static/.
"""
from __future__ import annotations

import asyncio
import importlib.util
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.request
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
DATA_DIR = Path(os.environ.get("BURNDECK_DATA_DIR", str(APP_ROOT)))

# ---------- Config ----------
# Precedence: BURNDECK_* environment > config.toml > built-in defaults.
# The file is optional — with no config at all the dashboard still comes up,
# auto-discovers local serving stacks via the sniffer, and generates
# credentials on first boot.


def _load_config_file() -> dict[str, Any]:
    import tomllib
    path = os.environ.get("BURNDECK_CONFIG") or str(APP_ROOT / "config.toml")
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # malformed config should be loud, not fatal
        print(f"burndeck: ignoring bad config {path}: {exc}", file=sys.stderr)
        return {}


CONFIG = _load_config_file()


def cfg(section: str, key: str, default: Any = None, env: str = "") -> Any:
    v = os.environ.get(env) if env else None
    if v not in (None, ""):
        if isinstance(default, bool):
            return v.strip().lower() not in ("0", "false", "no", "off")
        if isinstance(default, float):
            return float(v)
        if isinstance(default, int):
            return int(v)
        return v
    return CONFIG.get(section, {}).get(key, default)


# Fast GPU poll. NVML keeps this inexpensive; a subprocess-based nvidia-smi
# fallback preserves telemetry if the Python binding or driver is unavailable.
# LLM endpoint probes run on their own slower cadence so they never block it.
GPU_POLL_SECONDS = float(cfg("polling", "gpu_seconds", 0.25, "BURNDECK_GPU_POLL_SECONDS"))
LLM_POLL_SECONDS = float(cfg("polling", "llm_seconds", 1.0, "BURNDECK_LLM_POLL_SECONDS"))
# History (chart) sampling cadence — kept ~1s independent of the GPU push rate
# so the chart time-window and per-snapshot payload size stay constant even
# when the live tiles refresh several times per second.
HISTORY_INTERVAL = float(cfg("polling", "history_seconds", 1.0, "BURNDECK_HISTORY_INTERVAL"))
HISTORY_POINTS = int(cfg("polling", "history_points", 180, "BURNDECK_HISTORY_POINTS"))
# Synthetic rig: N > 0 replaces NVML and endpoint probes with a fake N-GPU
# tensor-parallel machine so the full stack can be previewed on GPU-less
# hardware. CI smoke-tests run this way.
MOCK_GPUS = int(cfg("demo", "mock_gpus", 0, "BURNDECK_MOCK_GPUS"))

MODEL_ROOTS: list[str] = [
    p for p in (
        os.environ.get("BURNDECK_MODEL_ROOTS")
        or ":".join(CONFIG.get("models", {}).get("roots", ["/models"]))
    ).split(":") if p
]

# Cards cycle through these when an endpoint doesn't pin its own accents.
_ACCENT_CYCLE = [("#45d1ff", "#8b7cff"), ("#4de6a8", "#57c8ff"),
                 ("#ff8c2e", "#ffc96b"), ("#fcee0a", "#ff8c2e")]


def _endpoints_from_config() -> list[dict[str, Any]]:
    """[[endpoints]] tables from config.toml. Zero entries is fine — the
    sniffer auto-discovers any local OpenAI-compatible server anyway."""
    out: list[dict[str, Any]] = []
    for i, e in enumerate(CONFIG.get("endpoints") or []):
        url = str(e.get("url") or e.get("endpoint") or "").rstrip("/")
        if not url:
            continue
        accent, accent2 = _ACCENT_CYCLE[i % len(_ACCENT_CYCLE)]
        out.append({
            "id": str(e.get("id") or f"ep-{i}-{url.rsplit(':', 1)[-1]}"),
            "endpoint": url,
            "label": str(e.get("label") or url.split("//", 1)[-1]),
            "gpu_indices": [int(x) for x in (e.get("gpus") or [])],
            "accent": str(e.get("accent") or accent),
            "accent2": str(e.get("accent2") or accent2),
        })
    return out


LLM_ENDPOINTS: list[dict[str, Any]] = _endpoints_from_config()

GPU_QUERY = (
    "index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "temperature.gpu,power.draw,power.limit,fan.speed,clocks.gr,clocks.mem,"
    "encoder.stats.averageFps,pstate,"
    "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,"
    "pci.bus_id"
)
PROC_QUERY = "gpu_uuid,pid,process_name,used_memory"

METRIC_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)\s*$")
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def f(v: Any, d: float = 0.0) -> float:
    if v in (None, "", "N/A", "[N/A]", "[Not Supported]"):
        return d
    try:
        return float(str(v).strip().split()[0])
    except Exception:
        return d


def run_smi(query: str, mode: str = "gpu") -> list[list[str]]:
    flag = "--query-gpu" if mode == "gpu" else "--query-compute-apps"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=False,
        )
        if out.returncode != 0:
            return []
        rows = []
        for line in out.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append([c.strip() for c in line.split(",")])
        return rows
    except Exception:
        return []


# NVML avoids spawning eight nvidia-smi processes per second at the default
# cadence. Arch installs the pure-Python binding for its system Python, while
# this service uses an isolated Python; load that binding directly when needed.
# Every path retains the subprocess implementation as a driver-safe fallback.
_NVML: Any = None
_NVML_HANDLES: tuple = ()
_NVML_POOL: ThreadPoolExecutor | None = None
_NVML_RETRY_AT = 0.0
_NVML_STATIC: dict[int, dict[str, Any]] = {}
_NVML_PROC_NAMES: dict[int, str] = {}
_NVML_SLOW: dict[int, tuple[float, dict[str, Any]]] = {}


def _load_nvml():
    global _NVML, _NVML_HANDLES, _NVML_POOL, _NVML_RETRY_AT
    if _NVML is not None:
        return _NVML
    if time.monotonic() < _NVML_RETRY_AT:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            try:
                import pynvml as nvml
            except ImportError:
                nvml = None
                paths = sorted(
                    Path("/usr/lib").glob("python*/site-packages/pynvml.py"),
                    reverse=True,
                )
                for path in paths:
                    spec = importlib.util.spec_from_file_location("_burndeck_pynvml", path)
                    if spec and spec.loader:
                        nvml = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = nvml
                        try:
                            spec.loader.exec_module(nvml)
                        except Exception:
                            sys.modules.pop(spec.name, None)
                            raise
                        break
                if nvml is None:
                    raise ImportError("pynvml binding not found")
        nvml.nvmlInit()
        _NVML_HANDLES = tuple(
            nvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(nvml.nvmlDeviceGetCount())
        )
        _NVML_POOL = ThreadPoolExecutor(
            max_workers=max(1, len(_NVML_HANDLES)),
            thread_name_prefix="nvml",
        )
        _NVML = nvml
        return nvml
    except Exception:
        _NVML_RETRY_AT = time.monotonic() + 30.0
        return None


def _drop_nvml() -> None:
    global _NVML, _NVML_HANDLES, _NVML_POOL, _NVML_RETRY_AT
    if _NVML_POOL is not None:
        _NVML_POOL.shutdown(wait=False, cancel_futures=True)
    if _NVML is not None:
        try:
            _NVML.nvmlShutdown()
        except Exception:
            pass
    _NVML = None
    _NVML_HANDLES = ()
    _NVML_POOL = None
    _NVML_STATIC.clear()
    _NVML_SLOW.clear()
    _NVML_PROC_NAMES.clear()
    _NVML_RETRY_AT = time.monotonic() + 30.0


def _nvml_call(call, default):
    try:
        return call()
    except Exception:
        return default


def _nvml_static_info(nvml, index: int, handle) -> dict[str, Any]:
    cached = _NVML_STATIC.get(index)
    if cached is not None:
        return cached
    pci = _nvml_call(lambda: nvml.nvmlDeviceGetPciInfo(handle), None)
    cached = {
        "index": index,
        "name": str(_nvml_call(lambda: nvml.nvmlDeviceGetName(handle), f"GPU {index}")).replace("NVIDIA ", ""),
        "uuid": str(_nvml_call(lambda: nvml.nvmlDeviceGetUUID(handle), f"GPU-{index}")),
        "pcie_gen_max": int(_nvml_call(lambda: nvml.nvmlDeviceGetMaxPcieLinkGeneration(handle), 0)),
        "pcie_width_max": int(_nvml_call(lambda: nvml.nvmlDeviceGetMaxPcieLinkWidth(handle), 0)),
        "bus_id": str(getattr(pci, "busId", "")),
        "power_limit_w": float(_nvml_call(lambda: nvml.nvmlDeviceGetPowerManagementLimit(handle), 0)) / 1000.0 or 1.0,
    }
    _NVML_STATIC[index] = cached
    return cached


def _nvml_device(nvml, index: int, handle) -> dict[str, Any]:
    static = _nvml_static_info(nvml, index, handle)
    util = _nvml_call(lambda: nvml.nvmlDeviceGetUtilizationRates(handle), None)
    memory = _nvml_call(lambda: nvml.nvmlDeviceGetMemoryInfo(handle), None)
    mem_used = float(getattr(memory, "used", 0)) / 2**20
    mem_total = float(getattr(memory, "total", 0)) / 2**20 or 1.0
    power_draw = float(_nvml_call(lambda: nvml.nvmlDeviceGetPowerUsage(handle), 0)) / 1000.0

    now = time.monotonic()
    cached_slow = _NVML_SLOW.get(index)
    if cached_slow is not None and now - cached_slow[0] < 1.0:
        slow = cached_slow[1]
    else:
        pstate = int(_nvml_call(lambda: nvml.nvmlDeviceGetPerformanceState(handle), -1))
        processes = []
        for proc in _nvml_call(lambda: nvml.nvmlDeviceGetComputeRunningProcesses(handle), ()):
            pid = int(proc.pid)
            name = _NVML_PROC_NAMES.get(pid)
            if name is None:
                raw_name = str(_nvml_call(lambda p=pid: nvml.nvmlSystemGetProcessName(p), f"pid {pid}"))
                executable = raw_name.split("\0", 1)[0].split(" ", 1)[0]
                name = (os.path.basename(executable) or raw_name)[:128]
                _NVML_PROC_NAMES[pid] = name
            used = getattr(proc, "usedGpuMemory", 0) or 0
            if used > 2**60:
                used = 0
            processes.append({
                "pid": pid,
                "process_name": name,
                "used_memory_mib": float(used) / 2**20,
            })
        slow = {
            "temperature_c": float(_nvml_call(lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU), 0)),
            "fan_pct": float(_nvml_call(lambda: nvml.nvmlDeviceGetFanSpeed(handle), 0)),
            "graphics_clock_mhz": float(_nvml_call(lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS), 0)),
            "memory_clock_mhz": float(_nvml_call(lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM), 0)),
            "pstate": f"P{pstate}" if pstate >= 0 else "",
            "pcie_gen": int(_nvml_call(lambda: nvml.nvmlDeviceGetCurrPcieLinkGeneration(handle), 0)),
            "pcie_width": int(_nvml_call(lambda: nvml.nvmlDeviceGetCurrPcieLinkWidth(handle), 0)),
            "pcie_aer": read_pcie_aer(static["bus_id"]),
            "processes": sorted(processes, key=lambda p: -p["used_memory_mib"])[:10],
        }
        _NVML_SLOW[index] = (now, slow)

    power_limit = static["power_limit_w"]
    return {
        **static,
        **slow,
        "utilization_gpu_pct": float(getattr(util, "gpu", 0)),
        "utilization_memory_pct": float(getattr(util, "memory", 0)),
        "memory_used_mib": mem_used,
        "memory_total_mib": mem_total,
        "memory_pct": (mem_used / mem_total) * 100.0,
        "power_draw_w": power_draw,
        "power_pct": (power_draw / power_limit) * 100.0,
    }

_aer_port_cache: dict[str, str] = {}


def _root_port_for(bdf: str) -> str:
    """Map a GPU PCI BDF to its parent root-port BDF (cached; stable per boot)."""
    if bdf in _aer_port_cache:
        return _aer_port_cache[bdf]
    port = ""
    try:
        port = os.path.basename(os.path.realpath(f"/sys/bus/pci/devices/{bdf}/.."))
    except Exception:
        port = ""
    _aer_port_cache[bdf] = port
    return port


def _read_aer_file(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open(path) as fh:
            for line in fh:
                key, _, val = line.partition(" ")
                key = key.strip()
                if key:
                    out[key] = int(val.strip() or "0")
    except Exception:
        pass
    return out


def read_pcie_aer(bus_id: str) -> dict[str, Any]:
    """Per-card PCIe AER error counters, read from the GPU's parent root port."""
    bdf = bus_id.strip().lower()
    parts = bdf.split(":")
    if len(parts) == 3 and len(parts[0]) > 4:
        bdf = f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"
    port = _root_port_for(bdf)
    if not port:
        return {}
    base = f"/sys/bus/pci/devices/{port}"
    cor = _read_aer_file(f"{base}/aer_dev_correctable")
    fatal = _read_aer_file(f"{base}/aer_dev_fatal")
    nonfatal = _read_aer_file(f"{base}/aer_dev_nonfatal")
    return {
        "port": port,
        "correctable": cor,
        "total_correctable": cor.get("TOTAL_ERR_COR", 0),
        "total_nonfatal": nonfatal.get("TOTAL_ERR_NONFATAL", 0),
        "total_fatal": fatal.get("TOTAL_ERR_FATAL", 0),
    }

def http_get(url: str, timeout: float = 1.5) -> tuple[int, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "burndeck/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as exc:
        return 0, f"__error__ {exc}"


def parse_prom(text: str) -> list[dict[str, Any]]:
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = METRIC_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        labels = {}
        raw = m.group("labels")
        if raw:
            for k, v in LABEL_RE.findall(raw[1:-1]):
                labels[k] = v
        try:
            value = float(m.group("value"))
        except Exception:
            continue
        samples.append({"name": name, "labels": labels, "value": value})
    return samples


# ---------- Model intelligence ----------
# Everything we can learn about a served model beyond raw TPS: identity (family /
# creator / origin blurb), architecture facts from its on-disk config.json, weight
# footprint, README model-card summary, engine + version, worker uptime and VRAM.

MODEL_KB: list[tuple[re.Pattern, dict[str, str]]] = [(re.compile(p), d) for p, d in [
    (r"glm", {"family": "GLM", "creator": "Z.ai (Zhipu AI)", "origin": "Beijing · CN",
              "blurb": "Z.ai's flagship open-weight MoE reasoning line — GLM descends from Tsinghua's General Language Model research."}),
    (r"qwen|qwq|qvq", {"family": "Qwen", "creator": "Alibaba Cloud", "origin": "Hangzhou · CN",
              "blurb": "Alibaba's Tongyi Qianwen family — open-weight dense and MoE models spanning chat, coding, math and vision."}),
    (r"deepseek", {"family": "DeepSeek", "creator": "DeepSeek AI", "origin": "Hangzhou · CN",
              "blurb": "DeepSeek's MoE + MLA pioneers — frontier reasoning at breakthrough training cost."}),
    (r"kimi", {"family": "Kimi", "creator": "Moonshot AI", "origin": "Beijing · CN",
              "blurb": "Moonshot AI's Kimi line — trillion-parameter-class MoE tuned for long context and agentic work."}),
    (r"llama", {"family": "Llama", "creator": "Meta AI", "origin": "Menlo Park · US",
              "blurb": "Meta's Llama herd — the open-weight family that kicked off the local-LLM era."}),
    (r"mixtral|mistral|codestral|devstral|magistral|ministral", {"family": "Mistral", "creator": "Mistral AI", "origin": "Paris · FR",
              "blurb": "Mistral AI's efficient European models — Apache-licensed dense workhorses and sparse MoE."}),
    (r"gemma", {"family": "Gemma", "creator": "Google DeepMind", "origin": "London · UK",
              "blurb": "Google DeepMind's open-weight siblings of Gemini, built for efficient serving."}),
    (r"phi-?\d", {"family": "Phi", "creator": "Microsoft Research", "origin": "Redmond · US",
              "blurb": "Microsoft's small-but-mighty Phi series — textbook-quality data over parameter count."}),
    (r"gpt-oss|gpt", {"family": "GPT", "creator": "OpenAI", "origin": "San Francisco · US",
              "blurb": "OpenAI's GPT lineage — gpt-oss brings the o-series reasoning stack to open weights."}),
    (r"nemotron", {"family": "Nemotron", "creator": "NVIDIA", "origin": "Santa Clara · US",
              "blurb": "NVIDIA's Nemotron models — hybrid-architecture, distillation-heavy, tuned for enterprise agents."}),
    (r"granite", {"family": "Granite", "creator": "IBM Research", "origin": "Yorktown Heights · US",
              "blurb": "IBM's Granite family — enterprise-focused, Apache-licensed, clean data provenance."}),
    (r"command", {"family": "Command", "creator": "Cohere", "origin": "Toronto · CA",
              "blurb": "Cohere's Command series — RAG-first enterprise models with strong tool use."}),
    (r"minimax", {"family": "MiniMax", "creator": "MiniMax", "origin": "Shanghai · CN",
              "blurb": "MiniMax's open MoE line — lightning attention and million-token context ambitions."}),
    (r"hunyuan", {"family": "Hunyuan", "creator": "Tencent", "origin": "Shenzhen · CN",
              "blurb": "Tencent's Hunyuan models — the WeChat giant's open-weight MoE line."}),
    (r"ernie", {"family": "ERNIE", "creator": "Baidu", "origin": "Beijing · CN",
              "blurb": "Baidu's ERNIE family — one of China's longest-running LLM programs, now open-weight."}),
    (r"internlm|intern-?vl", {"family": "InternLM", "creator": "Shanghai AI Lab", "origin": "Shanghai · CN",
              "blurb": "Shanghai AI Laboratory's open research models spanning language and vision."}),
    (r"falcon", {"family": "Falcon", "creator": "TII", "origin": "Abu Dhabi · AE",
              "blurb": "The Technology Innovation Institute's Falcon models — the Gulf's open-weight flagship."}),
    (r"exaone", {"family": "EXAONE", "creator": "LG AI Research", "origin": "Seoul · KR",
              "blurb": "LG's EXAONE — Korea's leading open-weight bilingual model line."}),
    (r"olmo|tulu", {"family": "OLMo", "creator": "Allen Institute for AI", "origin": "Seattle · US",
              "blurb": "AI2's fully-open models — weights, data and training code all published."}),
    (r"hermes", {"family": "Hermes", "creator": "Nous Research", "origin": "US · distributed",
              "blurb": "Nous Research's Hermes fine-tunes — community-driven and maximally steerable."}),
    (r"smollm", {"family": "SmolLM", "creator": "Hugging Face", "origin": "Paris · FR",
              "blurb": "Hugging Face's fully-open small models — trained in the open on curated data."}),
    (r"starcoder", {"family": "StarCoder", "creator": "BigCode (HF × ServiceNow)", "origin": "open consortium",
              "blurb": "The BigCode project's open code models with permissive, opt-out-respecting data."}),
]]


def kb_lookup(text: str) -> dict[str, str]:
    t = text.lower()
    for pat, d in MODEL_KB:
        if pat.search(t):
            return d
    return {}


_boot_ts: float = 0.0


def _boot_time() -> float:
    global _boot_ts
    if not _boot_ts:
        try:
            with open("/proc/stat") as fh:
                for ln in fh:
                    if ln.startswith("btime"):
                        _boot_ts = float(ln.split()[1])
                        break
        except Exception:
            _boot_ts = time.time() - uptime_seconds()
    return _boot_ts


_pid_start_cache: dict[int, float] = {}


def _pid_started(pid: int) -> float | None:
    """Epoch timestamp the process started (cached — it never changes)."""
    hit = _pid_start_cache.get(pid)
    if hit:
        return hit
    try:
        with open(f"/proc/{pid}/stat") as fh:
            parts = fh.read().rsplit(")", 1)[1].split()
        t = _boot_time() + float(parts[19]) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None
    if len(_pid_start_cache) > 4096:
        _pid_start_cache.clear()
    _pid_start_cache[pid] = t
    return t


_engine_cache: dict[str, dict[str, Any]] = {}


def probe_engine(cfg: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Engine name/version + native model facts (vLLM /version, SGLang
    /get_model_info + /get_server_info). Cached 10 min per endpoint once
    identified; retried every 20 s while unknown."""
    ep = cfg["endpoint"].rstrip("/")
    now = time.time()
    hit = _engine_cache.get(ep)
    if hit and now - hit["ts"] < (600 if hit["engine"] else 20):
        return hit
    info: dict[str, Any] = {"ts": now, "engine": "", "engine_version": "", "srv_model_path": "", "srv_ctx": None}
    if prefix != "vllm":  # sglang, or unknown: cheap native probe
        code, body = http_get(ep + "/get_model_info", timeout=1.0)
        if code == 200:
            try:
                j = json.loads(body)
                info["engine"] = "SGLang"
                info["srv_model_path"] = str(j.get("model_path") or "")
            except Exception:
                pass
        if info["engine"] == "SGLang":
            code, body = http_get(ep + "/get_server_info", timeout=1.5)
            if code == 200:
                try:
                    j = json.loads(body)
                    info["engine_version"] = str(j.get("version") or "")
                    info["srv_ctx"] = j.get("context_len") or (j.get("model_config") or {}).get("context_len")
                except Exception:
                    pass
    if not info["engine"]:
        code, body = http_get(ep + "/version", timeout=1.0)
        if code == 200:
            try:
                v = json.loads(body).get("version")
                if v:
                    info["engine"] = "vLLM"
                    info["engine_version"] = str(v)
            except Exception:
                pass
    if not info["engine"] and prefix:
        info["engine"] = {"vllm": "vLLM", "sglang": "SGLang"}[prefix]
    _engine_cache[ep] = info
    return info


def _fmt_params(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.2f}".rstrip("0").rstrip(".") + "T"
    if n >= 1e9:
        v = n / 1e9
        return (f"{v:.0f}" if v >= 100 else f"{v:.1f}") + "B"
    return f"{n / 1e6:.0f}M"


def _strip_md(s: str) -> str:
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = s.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", s).strip()


def _readme_intel(path: str) -> dict[str, Any]:
    """license / base_model front matter, first substantive paragraph as a short
    summary, and a stated parameter count ("753B") if the card leads with one."""
    try:
        with open(os.path.join(path, "README.md"), encoding="utf-8", errors="replace") as fh:
            text = fh.read(24576)
    except OSError:
        return {}
    out: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            fm, body = text[3:end], text[end + 4:]
            lm = re.search(r"^license(?:_name)?:\s*(\S[^\n]*)", fm, re.M)
            if lm:
                out["license"] = lm.group(1).strip().strip("\"'")
            bm = re.search(r"^base_model:\s*\n((?:\s*-\s*\S+\n?)+)", fm, re.M)
            if bm:
                out["base_model"] = [x.strip("- ").strip() for x in bm.group(1).strip().splitlines()][:3]
            else:
                bm1 = re.search(r"^base_model:\s*(\S+)", fm, re.M)
                if bm1:
                    out["base_model"] = [bm1.group(1)]
    for para in re.split(r"\n\s*\n", body):
        s = para.strip()
        if not s or s.startswith(("#", ">", "|", "![", "<", "```", "---")):
            continue
        s = _strip_md(s)
        if len(s) < 60:
            continue
        if len(s) > 300:
            s = s[:300].rsplit(" ", 1)[0] + " …"
        out["summary"] = s
        break
    sm = re.search(r"\b(\d{1,4}(?:\.\d+)?)\s?B\b", _strip_md(body)[:1500])
    if sm and 0.1 <= float(sm.group(1)) <= 5000:
        out["params"] = sm.group(1) + "B"
    return out


def _safetensor_params(path: str) -> int:
    """Exact tensor-element count from safetensors headers (8-byte length +
    JSON header per shard — no weight data is read)."""
    total = 0
    try:
        for name in os.listdir(path):
            if not name.endswith(".safetensors"):
                continue
            with open(os.path.join(path, name), "rb") as fh:
                hlen = int.from_bytes(fh.read(8), "little")
                if not 0 < hlen < 50_000_000:
                    return 0
                hdr = json.loads(fh.read(hlen))
            for k, v in hdr.items():
                if k == "__metadata__" or not isinstance(v, dict):
                    continue
                p = 1
                for s in v.get("shape") or []:
                    p *= s
                total += p
    except Exception:
        return 0
    return total


_disk_cache: dict[str, dict[str, Any]] = {}


def disk_intel(path: str) -> dict[str, Any]:
    """Architecture / quant / context / weight facts from a local model dir.
    Cached by config.json mtime; the safetensors header scan runs once."""
    cfg_p = os.path.join(path, "config.json")
    try:
        mtime = os.stat(cfg_p).st_mtime
    except OSError:
        return {}
    hit = _disk_cache.get(path)
    if hit is not None and hit.get("_mtime") == mtime:
        return hit
    out: dict[str, Any] = {"_mtime": mtime}
    try:
        with open(cfg_p) as fh:
            cfg = json.load(fh)
    except Exception:
        cfg = {}
    arch = (cfg.get("architectures") or [""])[0]
    out["arch"] = re.sub(r"For(CausalLM|ConditionalGeneration)$", "", arch)
    out["model_type"] = cfg.get("model_type") or ""
    out["layers"] = cfg.get("num_hidden_layers")
    out["hidden"] = cfg.get("hidden_size")
    out["heads"] = cfg.get("num_attention_heads")
    out["kv_heads"] = cfg.get("num_key_value_heads")
    out["vocab"] = cfg.get("vocab_size")
    out["ctx_native"] = cfg.get("max_position_embeddings")
    out["dtype"] = str(cfg.get("dtype") or cfg.get("torch_dtype") or "")
    out["attn"] = "MLA" if cfg.get("kv_lora_rank") else ""
    experts = cfg.get("n_routed_experts") or cfg.get("num_local_experts") or cfg.get("num_experts")
    if experts:
        out["moe"] = {"experts": experts,
                      "active": cfg.get("num_experts_per_tok") or cfg.get("moe_top_k"),
                      "shared": cfg.get("n_shared_experts") or 0}
    qc = cfg.get("quantization_config") or {}
    quant = str(qc.get("quant_algo") or qc.get("quant_method") or "")
    bits = qc.get("bits")
    if bits and quant:
        quant = f"{quant} {bits}-bit"
    out["quant"] = quant.upper()[:24]
    total = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.is_file() and e.name.endswith((".safetensors", ".bin", ".gguf", ".pt")):
                    total += e.stat().st_size
    except OSError:
        pass
    out["weights_gb"] = round(total / 1e9, 1) if total else None
    out.update(_readme_intel(path))
    if not out.get("params"):
        n = _safetensor_params(path)
        if n:
            # packed quant tensors under-count logical params → flag approximate
            out["params"] = ("~" if quant else "") + _fmt_params(n)
    _disk_cache[path] = out
    return out


def _flag(text: str, *flags: str) -> str:
    for fl in flags:
        mt = re.search(rf"(?:^|\s){re.escape(fl)}(?:[= ]\s*)(\S+)", text)
        if mt:
            return mt.group(1)
    return ""


def _int_flag(text: str, *flags: str) -> int | None:
    try:
        return int(_flag(text, *flags))
    except (TypeError, ValueError):
        return None


def _resolve_model_path(m: dict[str, Any], anc: str) -> str:
    cands: list[str] = []
    if m.get("srv_model_path"):
        cands.append(m["srv_model_path"])
    for meta in m.get("model_meta") or []:
        if meta.get("root"):
            cands.append(meta["root"])
    for fl in ("--model-path", "--model_path", "--model"):
        v = _flag(anc, fl)
        if v:
            cands.append(v)
    cands += list(m.get("model_names") or [])
    for c in cands:
        if c.startswith("/") and os.path.isdir(c):
            return c
    # container paths (e.g. /model inside docker) don't exist on the host —
    # fall back to fuzzy-matching the served name against local model dirs
    for n in m.get("model_names") or []:
        d = _match_local_dir(n)
        if d:
            return d
    return ""


def enrich_models(models: list[dict[str, Any]], now: float) -> None:
    """Attach m['intel'] — identity, architecture, provenance and serving facts.
    Cheap per tick: disk facts and engine probes are cached; /proc reads are
    behind the 60 s ancestor-cmdline cache."""
    for m in models:
        pids = m.get("worker_pids") or []
        anc = "\n".join(_ancestor_cmdlines(p) for p in pids[:3])
        path = _resolve_model_path(m, anc)
        disk = disk_intel(path) if path else {}
        served = next(iter(m.get("model_names") or []), "")
        display = served or (os.path.basename(path.rstrip("/")) if path else "")
        kb = kb_lookup(f"{served} {os.path.basename(path)}")
        meta = next(iter(m.get("model_meta") or []), {})
        ctx_srv = meta.get("max_model_len") or m.get("srv_ctx") or _int_flag(anc, "--max-model-len")
        tp = _int_flag(anc, "--tensor-parallel-size", "--tp-size", "--tp", "-tp") \
            or (len(m.get("gpu_indices") or []) or None)
        starts = [t for t in (_pid_started(p) for p in pids) if t]
        m["intel"] = {
            "display_name": display,
            "path": path,
            "family": kb.get("family", ""),
            "creator": kb.get("creator", ""),
            "origin": kb.get("origin", ""),
            "blurb": kb.get("blurb", ""),
            "summary": disk.get("summary", ""),
            "license": disk.get("license", ""),
            "base_model": disk.get("base_model") or [],
            "arch": disk.get("arch", ""),
            "model_type": disk.get("model_type", ""),
            "attn": disk.get("attn", ""),
            "params": disk.get("params", ""),
            "layers": disk.get("layers"),
            "hidden": disk.get("hidden"),
            "heads": disk.get("heads"),
            "kv_heads": disk.get("kv_heads"),
            "vocab": disk.get("vocab"),
            "moe": disk.get("moe"),
            "quant": disk.get("quant", ""),
            "dtype": disk.get("dtype", ""),
            "ctx_native": disk.get("ctx_native"),
            "ctx_serving": ctx_srv,
            "weights_gb": disk.get("weights_gb"),
            "engine": m.get("engine", ""),
            "engine_version": m.get("engine_version", ""),
            "tp": tp,
            "uptime_s": (now - min(starts)) if starts else None,
            "vram_mib": m.get("vram_mib"),
        }


# ---------- Universal model sniffer ----------
# Endpoint-agnostic detection: any GPU compute process holding model weights
# (mmap'd or open .safetensors/.gguf/.pt/... >= 64 MB) is reported, with the
# owning app, VRAM, GPUs, weight files, discovered API ports and — when the
# weights live in a local HF-style dir — full disk_intel enrichment.

_WEIGHT_MIN_BYTES = 64 * 1024 * 1024


def _is_weight_file(path: str) -> bool:
    if path.endswith((".safetensors", ".gguf", ".pt", ".pth", ".onnx", ".ckpt", ".engine")):
        return not path.startswith(("/tmp/torchinductor", "/dev", "/proc"))
    if path.endswith(".bin"):
        b = os.path.basename(path).lower()
        return "model" in b or "weight" in b or "consolidated" in b
    return False


_fsize_cache: dict[str, float] = {}


def _fsize(path: str) -> float:
    v = _fsize_cache.get(path)
    if v is None:
        try:
            v = float(os.stat(path).st_size)
        except OSError:
            v = 0.0
        if len(_fsize_cache) > 8192:
            _fsize_cache.clear()
        _fsize_cache[path] = v
    return v


_maps_cache: dict[int, tuple[float, frozenset]] = {}


def _pid_weight_files(pid: int) -> frozenset:
    """Weight files a process holds: mmap'd regions (survive fd close — how
    safetensors loads) plus open fds. Cached 10 s per pid."""
    now = time.time()
    hit = _maps_cache.get(pid)
    if hit and now - hit[0] < 10:
        return hit[1]
    found: set[str] = set()
    try:
        with open(f"/proc/{pid}/maps") as fh:
            for ln in fh:
                i = ln.find("/")
                if i < 0:
                    continue
                path = ln[i:].strip()
                if path.endswith(" (deleted)"):
                    path = path[:-10]
                if _is_weight_file(path):
                    found.add(path)
    except OSError:
        pass
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                tgt = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if tgt.startswith("/") and _is_weight_file(tgt):
                found.add(tgt)
    except OSError:
        pass
    fs = frozenset(p for p in found if _fsize(p) >= _WEIGHT_MIN_BYTES)
    if len(_maps_cache) > 512:
        _maps_cache.clear()
    _maps_cache[pid] = (now, fs)
    return fs


def _cmdline_model_paths(anc: str) -> set:
    out: set[str] = set()
    for fl in ("--model-path", "--model_path", "--model", "--checkpoint-path", "--distilled-checkpoint-path",
               "--spatial-upsampler-path", "--gemma-root", "--ckpt", "--weights", "--lora-path"):
        v = _flag(anc, fl)
        if v:
            out.add(v)
    for mt in re.finditer(r"(/\S+\.(?:safetensors|gguf|bin|pt|pth|onnx|ckpt))(?=\s|$)", anc):
        out.add(mt.group(1))
    return {p for p in out if p.startswith("/") and os.path.exists(p)}


_APP_SIGS = [
    ("vllm", "vLLM"), ("sglang", "SGLang"), ("llama-server", "llama.cpp"), ("llama.cpp", "llama.cpp"),
    ("ollama", "Ollama"), ("comfyui", "ComfyUI"), ("ComfyUI", "ComfyUI"), ("ltx", "LTX-2"),
    ("text-generation", "TGI"), ("exllama", "ExLlama"), ("ktransformers", "KTransformers"),
    ("diffusers", "Diffusers"), ("whisper", "Whisper"), ("torchrun", "torchrun"), ("accelerate", "Accelerate"),
]


def _detect_app(anc: str) -> str:
    for sig, name in _APP_SIGS:
        if sig in anc or sig.lower() in anc.lower():
            return name
    mt = re.search(r"-m\s+([\w\.]+)", anc)
    if mt:
        return mt.group(1).split(".")[0]
    mt = re.search(r"(\S+\.py)\b", anc)
    if mt:
        return os.path.basename(mt.group(1))
    return ""


def _model_key_name(path: str) -> tuple:
    """Map a weight path to a stable (group_key, display_name): sharded/HF
    checkpoints group under their directory."""
    p = path.rstrip("/")
    if os.path.isdir(p):
        return p, os.path.basename(p)
    d = os.path.dirname(p)
    base = os.path.basename(p)
    if re.search(r"-\d{5}-of-\d{5}", base) or os.path.exists(os.path.join(d, "config.json")):
        return d, os.path.basename(d)
    return p, re.sub(r"\.(safetensors|gguf|bin|pt|pth|onnx|ckpt|engine)$", "", base)


def _listen_ports(pids: set) -> dict:
    """pid -> sorted listening TCP ports, via socket-inode matching."""
    inodes: dict[str, int] = {}
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(f) as fh:
                next(fh)
                for ln in fh:
                    parts = ln.split()
                    if len(parts) > 9 and parts[3] == "0A":
                        inodes[parts[9]] = int(parts[1].rsplit(":", 1)[1], 16)
        except OSError:
            continue
    out: dict[int, list] = {}
    for pid in pids:
        ports = set()
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    tgt = os.readlink(f"/proc/{pid}/fd/{fd}")
                except OSError:
                    continue
                if tgt.startswith("socket:["):
                    port = inodes.get(tgt[8:-1])
                    if port:
                        ports.add(port)
        except OSError:
            continue
        if ports:
            out[pid] = sorted(ports)
    return out


_port_probe_cache: dict[int, tuple[float, list]] = {}


def _probe_port_models(port: int) -> list:
    """Served model ids if the port speaks OpenAI /v1/models. Cached 2 min."""
    now = time.time()
    hit = _port_probe_cache.get(port)
    if hit and now - hit[0] < 120:
        return hit[1]
    names: list[str] = []
    code, body = http_get(f"http://127.0.0.1:{port}/v1/models", timeout=0.8)
    if code == 200:
        try:
            names = [d.get("id", "") for d in json.loads(body).get("data", []) if d.get("id")]
        except Exception:
            pass
    if len(_port_probe_cache) > 256:
        _port_probe_cache.clear()
    _port_probe_cache[port] = (now, names)
    return names


_SERVE_SIG_RE = re.compile(
    r"(vllm(?:\.entrypoints\S*)?\s+serve|-m\s+vllm\.entrypoints|sglang\.launch_server|"
    r"llama-server|ollama\s+(?:serve|runner)|text-generation-launcher|lmdeploy\s+serve)", re.I)

_ENGINE_SIGS = [("vllm", "vLLM"), ("sglang", "SGLang"), ("llama-server", "llama.cpp"),
                ("ollama", "Ollama"), ("text-generation", "TGI"), ("lmdeploy", "LMDeploy")]


def _clean_arg(v: str) -> str:
    """Drop garbage scraped from wrapper-script argv (docker-init/bash hold the
    raw script text, so flags can surface as unexpanded '"${served_name}"')."""
    v = (v or "").strip().strip("'\"")
    return "" if "$" in v else v


def _scan_serving_procs() -> list:
    """System-wide sweep for model-serving processes by cmdline signature.
    Catches servers during weight load (before any CUDA context exists) and
    root / containerized servers whose /proc/<pid>/maps we can't read —
    /proc/<pid>/cmdline is world-readable either way."""
    out = []
    try:
        pids = [int(d) for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return out
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read(8192).replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if not cmd or not _SERVE_SIG_RE.search(cmd):
            continue
        engine = next((n for s, n in _ENGINE_SIGS if s in cmd.lower()), "")
        mpath = ""
        for fl in ("--model-path", "--model_path", "--model", "--ckpt", "--gguf"):
            v = _flag(cmd, fl)
            if v and not v.startswith("-"):
                mpath = v
                break
        if not mpath and engine == "llama.cpp":
            mpath = _flag(cmd, "-m")
        if not mpath:
            mt = re.search(r"\bserve\s+([^-\s]\S*)", cmd)
            if mt:
                mpath = mt.group(1)
        out.append({
            "pid": pid,
            "engine": engine,
            "port": _int_flag(cmd, "--port"),
            "name": _clean_arg(_flag(cmd, "--served-model-name", "--served_model_name", "--model-name", "--alias")),
            "mpath": _clean_arg(mpath),
        })
    return out


_local_dirs_cache: dict[str, Any] = {"ts": 0.0, "dirs": []}


def _local_model_dirs() -> list:
    """Model dirs (containing config.json) under [models].roots, cached 5 min."""
    now = time.time()
    if now - _local_dirs_cache["ts"] < 300:
        return _local_dirs_cache["dirs"]
    dirs: list[str] = []
    for root in MODEL_ROOTS:
        for pat in ("*", "*/*"):
            try:
                import glob as _glob
                for d in _glob.glob(os.path.join(root, pat)):
                    if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
                        dirs.append(d)
            except OSError:
                continue
    _local_dirs_cache.update(ts=now, dirs=dirs)
    return dirs


def _resolve_serving_path(sp: dict) -> str:
    """Best-effort model dir for a serving proc: host path → container path via
    /proc/<pid>/root (readable only with privilege) → fuzzy local-dir match on
    the served model name."""
    mp = sp.get("mpath") or ""
    if mp and os.path.isdir(mp):
        return mp
    if mp.startswith("/"):
        alt = f"/proc/{sp['pid']}/root{mp}"
        try:
            if os.path.isdir(alt) and os.path.exists(os.path.join(alt, "config.json")):
                return alt
        except OSError:
            pass
    return _match_local_dir(sp.get("name") or os.path.basename(mp.rstrip("/")))


def _match_local_dir(name: str) -> str:
    n = (name or "").lower().replace(" ", "")
    if not n:
        return ""
    for d in _local_model_dirs():
        b = os.path.basename(d).lower()
        if n in b or b in n:
            return d
    return ""


def _intel_lite(kb: dict, disk: dict) -> dict:
    return {
        "family": kb.get("family", ""),
        "creator": kb.get("creator", ""),
        "origin": kb.get("origin", ""),
        "blurb": kb.get("blurb", ""),
        "summary": disk.get("summary", ""),
        "license": disk.get("license", ""),
        "arch": disk.get("arch", ""),
        "params": disk.get("params", ""),
        "quant": disk.get("quant", ""),
        "dtype": disk.get("dtype", ""),
        "ctx_native": disk.get("ctx_native"),
        "moe": disk.get("moe"),
        "layers": disk.get("layers"),
        "hidden": disk.get("hidden"),
        "heads": disk.get("heads"),
        "attn": disk.get("attn", ""),
    }


# ---------- Model-load telemetry ----------
# Realtime progress for a model being loaded, from world-readable /proc only
# (works for root / containerized loaders): VRAM fill vs expected weights,
# disk read rate of the volume feeding the load, loader CPU, RAM staging, ETA.

_dev_cache: dict[str, str] = {}


def _dev_for_path(path: str) -> str:
    """maj:min of the block device backing path (for /proc/diskstats lookup).
    Anonymous-device filesystems (btrfs/bcachefs multi-device: st_dev major 0)
    resolve via /proc/mounts source; '*' = sum all physical disks."""
    dev = _dev_cache.get(path)
    if dev is not None:
        return dev
    dev = ""
    try:
        st = os.stat(path)
        if os.major(st.st_dev):
            dev = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
        else:
            best = ("", "")
            with open("/proc/mounts") as fh:
                for ln in fh:
                    parts = ln.split()
                    if len(parts) > 1 and parts[0].startswith("/dev/") \
                            and path.startswith(parts[1]) and len(parts[1]) > len(best[0]):
                        best = (parts[1], parts[0])
            if best[1]:
                rst = os.stat(best[1])
                dev = f"{os.major(rst.st_rdev)}:{os.minor(rst.st_rdev)}"
            else:
                dev = "*"
    except OSError:
        dev = ""
    _dev_cache[path] = dev
    return dev


_PHYS_DISK_RE = re.compile(r"^(nvme\d+n\d+|sd[a-z]+)$")


def _disk_read_bytes(dev: str):
    if not dev:
        return None
    try:
        with open("/proc/diskstats") as fh:
            if dev == "*":
                return sum(int(f[5]) * 512 for f in (ln.split() for ln in fh)
                           if len(f) > 5 and _PHYS_DISK_RE.match(f[2]))
            maj, minor = dev.split(":")
            for ln in fh:
                f = ln.split()
                if f[0] == maj and f[1] == minor:
                    return int(f[5]) * 512  # sectors read
    except (OSError, ValueError, IndexError):
        pass
    return None


def _pids_cpu_ticks(pids: list) -> float:
    total = 0.0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as fh:
                parts = fh.read().rsplit(")", 1)[1].split()
            total += float(parts[11]) + float(parts[12])  # utime + stime
        except (OSError, IndexError, ValueError):
            continue
    return total


def _pids_rss_mib(pids: list) -> float:
    total = 0
    page = os.sysconf("SC_PAGE_SIZE")
    for pid in pids:
        try:
            with open(f"/proc/{pid}/statm") as fh:
                total += int(fh.read().split()[1]) * page
        except (OSError, IndexError, ValueError):
            continue
    return total / 2**20


_LOAD_STAGES = ["init", "weights", "warmup", "api"]
_load_track: dict[str, dict[str, Any]] = {}


def _loading_telemetry(eid: str, pids: list, on_gpu: bool, vram_mib: float, path: str, weights_gb) -> dict:
    now = time.time()
    cpu_ticks = _pids_cpu_ticks(pids)
    dev = _dev_for_path(path) if path else ""
    disk_bytes = _disk_read_bytes(dev)
    tel: dict[str, Any] = {
        "stages": _LOAD_STAGES,
        "cpu_pct": None, "disk_read_mbps": None, "vram_rate_mibps": None,
        "rss_mib": round(_pids_rss_mib(pids), 1),
        "eta_s": None, "pct": None,
    }
    prev = _load_track.get(eid)
    if prev and now - prev["ts"] > 0.2:
        dt = now - prev["ts"]
        tel["cpu_pct"] = round(max(0.0, (cpu_ticks - prev["cpu"]) / os.sysconf("SC_CLK_TCK") / dt * 100.0), 1)
        if disk_bytes is not None and prev.get("disk") is not None:
            tel["disk_read_mbps"] = round(max(0.0, (disk_bytes - prev["disk"]) / dt / 1e6), 1)
        tel["vram_rate_mibps"] = round(max(0.0, (vram_mib - prev["vram"]) / dt), 1)
    _load_track[eid] = {"ts": now, "cpu": cpu_ticks, "disk": disk_bytes, "vram": vram_mib}
    if len(_load_track) > 64:
        for k in [k for k, v in _load_track.items() if now - v["ts"] > 600]:
            _load_track.pop(k, None)
    target_mib = weights_gb * 1e9 / 2**20 if weights_gb else None
    if not on_gpu:
        tel["stage"] = "init"
        tel["stage_idx"] = 0
        tel["detail"] = "spawning workers · staging weights"
    elif target_mib and vram_mib < target_mib * 0.92:
        tel["stage"] = "weights"
        tel["stage_idx"] = 1
        tel["pct"] = round(min(100.0, vram_mib / target_mib * 100.0), 1)
        tel["detail"] = f"{vram_mib / 953.7:.0f} / {weights_gb:.0f} GB into VRAM"
        rate = tel["vram_rate_mibps"]
        if rate and rate > 1:
            tel["eta_s"] = round((target_mib - vram_mib) / rate, 1)
    else:
        tel["stage"] = "warmup"
        tel["stage_idx"] = 2
        tel["pct"] = 100.0 if target_mib else None
        tel["detail"] = "kv cache · cuda graphs · api warmup"
    tel["vram_target_mib"] = round(target_mib, 1) if target_mib else None
    return tel


def sniff_models(gpus: list, models: list) -> list:
    """Detect every model on (or headed for) the GPUs, regardless of stack:
    weight files held by GPU procs, plus serving processes still in their
    weight-loading phase, plus root/container servers via cmdline evidence."""
    claimed_pids: set[int] = set()
    known_ports = {m.get("port") for m in models}
    healthy_ports = {m.get("port") for m in models if m.get("healthy")}
    for m in models:
        if m.get("healthy"):
            claimed_pids.update(m.get("worker_pids") or [])
    scan_by_pid = {sp["pid"]: sp for sp in _scan_serving_procs()}
    groups: dict[str, dict] = {}
    seen_pairs: dict[str, set] = {}
    all_pids: set[int] = set()
    for g in gpus:
        for p in g.get("processes") or []:
            pid = p["pid"]
            all_pids.add(pid)
            anc = _ancestor_cmdlines(pid)
            files = set(_pid_weight_files(pid)) | _cmdline_model_paths(anc)
            if not files:
                # unreadable /proc internals (root / container) — fall back to
                # cmdline evidence: this pid's own scan entry, or a scanned
                # server found in its ancestor chain (TP/spawned workers)
                sp = scan_by_pid.get(pid) or next(
                    (x for x in scan_by_pid.values()
                     if (x.get("name") and x["name"] in anc) or (x.get("mpath") and x["mpath"] in anc)),
                    None)
                if sp:
                    path = _resolve_serving_path(sp)
                    if path:
                        files = {path}
                        scan_by_pid.setdefault(pid, sp)
            if not files:
                continue
            for f in files:
                key, name = _model_key_name(f)
                if pid in scan_by_pid and scan_by_pid[pid].get("name"):
                    name = scan_by_pid[pid]["name"] if not os.path.isdir(f) else name
                grp = groups.setdefault(key, {
                    "key": key, "name": name, "path": key, "app": "",
                    "files": set(), "pids": set(), "gpu_indices": set(),
                    "vram_mib": 0.0, "claimed": False,
                })
                grp["files"].add(f)
                grp["pids"].add(pid)
                grp["gpu_indices"].add(g["index"])
                pair = (g["index"], pid)
                pairs = seen_pairs.setdefault(key, set())
                if pair not in pairs:
                    pairs.add(pair)
                    grp["vram_mib"] += p["used_memory_mib"]
                if not grp["app"]:
                    grp["app"] = (scan_by_pid.get(pid) or {}).get("engine") or _detect_app(anc)
                if pid in claimed_pids:
                    grp["claimed"] = True
    ports_by_pid = _listen_ports(all_pids)
    out = []
    for grp in groups.values():
        raw_ports = {pt for pid in grp["pids"] for pt in ports_by_pid.get(pid, [])} \
                  | {scan_by_pid[pid]["port"] for pid in grp["pids"]
                     if pid in scan_by_pid and scan_by_pid[pid].get("port")}
        if grp["claimed"] or (raw_ports & healthy_ports):
            continue  # already shown as a (configured or dynamic) endpoint card
        path = grp["path"]
        disk = disk_intel(path) if os.path.isdir(path) else {}
        kb = kb_lookup(grp["name"] + " " + os.path.basename(path))
        ports = sorted({pt for pid in grp["pids"] for pt in ports_by_pid.get(pid, []) if pt not in known_ports})
        for pid in grp["pids"]:
            sp = scan_by_pid.get(pid)
            if sp and sp.get("port") and sp["port"] not in known_ports:
                ports.append(sp["port"])
        ports = sorted(set(ports))
        served = []
        for pt in ports[:3]:
            served += _probe_port_models(pt)
        disk_gb = disk.get("weights_gb") or (round(sum(_fsize(f) for f in grp["files"] if os.path.isfile(f)) / 1e9, 1) or None)
        starts = [t for t in (_pid_started(p) for p in grp["pids"]) if t]
        out.append({
            "id": "sniff-" + hashlib.md5(grp["key"].encode()).hexdigest()[:10],
            "name": grp["name"],
            "path": path,
            "app": grp["app"],
            "state": "memory",
            "files": sorted(grp["files"])[:24],
            "n_files": len([f for f in grp["files"] if os.path.isfile(f)]),
            "pids": sorted(grp["pids"])[:16],
            "gpu_indices": sorted(grp["gpu_indices"]),
            "vram_mib": round(grp["vram_mib"], 1),
            "disk_gb": disk_gb,
            "ports": ports[:4],
            "served_names": served[:4],
            "uptime_s": (time.time() - min(starts)) if starts else None,
            "intel": _intel_lite(kb, disk),
        })
    # serving processes with no GPU presence yet = weight-loading phase.
    # Require actual model evidence (name / path) so idle daemons like a bare
    # `ollama serve` don't render as phantom loads; merge duplicate pids of the
    # same model+port (e.g. a container shim and the server inside it).
    loading: dict[tuple, dict] = {}
    for sp in scan_by_pid.values():
        if sp["pid"] in all_pids:
            continue
        if sp.get("port") and sp["port"] in healthy_ports:
            continue
        path = _resolve_serving_path(sp)
        if path and path in groups:
            continue  # its workers are already visible as an in-memory group
        name = sp.get("name") or (os.path.basename((sp.get("mpath") or "").rstrip("/"))) \
            or (os.path.basename(path.rstrip("/")) if path else "")
        if not name:
            continue  # no model evidence — an idle serving daemon, not a load
        lkey = (path or name.lower(), sp.get("port"))
        ent = loading.get(lkey)
        if ent:
            ent["pids"].append(sp["pid"])
            continue
        disk = disk_intel(path) if path else {}
        kb = kb_lookup(f"{name} {os.path.basename(path)}")
        started = _pid_started(sp["pid"])
        loading[lkey] = {
            "id": "sniff-load-" + hashlib.md5(f"{lkey}".encode()).hexdigest()[:10],
            "name": name,
            "path": path,
            "app": sp.get("engine", ""),
            "state": "loading",
            "files": [],
            "n_files": 0,
            "pids": [sp["pid"]],
            "gpu_indices": [],
            "vram_mib": None,
            "disk_gb": disk.get("weights_gb"),
            "ports": [sp["port"]] if sp.get("port") else [],
            "served_names": [],
            "uptime_s": (time.time() - started) if started else None,
            "intel": _intel_lite(kb, disk),
        }
    out.extend(loading.values())
    # attach realtime load telemetry: explicit loading entries, plus on-GPU
    # groups that expose an API port which isn't serving yet (still warming up)
    for e in out:
        api_pending = bool(e.get("ports")) and not e.get("served_names")
        if e["state"] == "loading" or api_pending:
            e["state"] = "loading"
            e["loading"] = _loading_telemetry(
                e["id"], e.get("pids") or [], bool(e.get("gpu_indices")),
                float(e.get("vram_mib") or 0.0), e.get("path") or "", e.get("disk_gb"))
    out.sort(key=lambda x: -(x["vram_mib"] or 0))
    return out[:10]


# Dynamic endpoints: any sniffed port that speaks OpenAI /v1/models gets scraped
# like a configured endpoint (TPS, kv-cache, queue) with zero configuration.
_DYN_ACCENTS = [("#4de6a8", "#57c8ff"), ("#ff8c2e", "#ffc96b"), ("#8b7cff", "#45d1ff"), ("#fcee0a", "#ff8c2e")]


def update_dynamic_endpoints(sniffed: list) -> None:
    now = time.time()
    static_ports = set()
    for c in LLM_ENDPOINTS:
        try:
            static_ports.add(int(c["endpoint"].rstrip("/").rsplit(":", 1)[1]))
        except Exception:
            pass
    cand: set[int] = set()
    for s in sniffed:
        cand.update(s.get("ports") or [])
    for pt in cand:
        if pt in static_ports:
            continue
        names = _probe_port_models(pt)
        if not names:
            continue
        cfg = STATE.dynamic_endpoints.get(pt)
        if cfg is None:
            acc = _DYN_ACCENTS[pt % len(_DYN_ACCENTS)]
            STATE.dynamic_endpoints[pt] = {
                "id": f"auto-{pt}", "endpoint": f"http://127.0.0.1:{pt}",
                "label": f"{names[0]} · auto", "gpu_indices": [],
                "accent": acc[0], "accent2": acc[1], "_dyn_ok": now,
            }
        else:
            cfg["_dyn_ok"] = now
    for m in STATE.models_cache:
        if str(m.get("id", "")).startswith("auto-") and m.get("healthy"):
            cfg = STATE.dynamic_endpoints.get(m.get("port"))
            if cfg:
                cfg["_dyn_ok"] = now
    for pt in list(STATE.dynamic_endpoints):
        if now - STATE.dynamic_endpoints[pt].get("_dyn_ok", 0) > 900:
            del STATE.dynamic_endpoints[pt]


# ---------- Polling state ----------

class State:
    def __init__(self):
        self.snapshot: dict[str, Any] = {"generated_at": utc_iso(), "gpus": [], "models": [], "history": {}}
        self.history: dict[int, dict[str, deque]] = {}
        self.global_hist: dict[str, deque] = {
            "total_tps": deque(maxlen=HISTORY_POINTS),
            "gpu_util_avg": deque(maxlen=HISTORY_POINTS),
            "mem_pct_avg": deque(maxlen=HISTORY_POINTS),
            "power_total": deque(maxlen=HISTORY_POINTS),
        }
        self.prev_counters: dict[str, dict[str, float]] = {}  # endpoint -> {prompt_total, gen_total, ts}
        self.subscribers: set[asyncio.Queue] = set()
        # Cache of the last LLM-endpoint scrape (refreshed on the slower
        # LLM_POLL_SECONDS cadence by its own background task). The fast GPU
        # poll reads this instead of doing blocking HTTP every tick.
        self.models_cache: list[dict[str, Any]] = []
        # Universal model sniffer output (refreshed by llm_poller off the fast path).
        self.sniffed_cache: list[dict[str, Any]] = []
        # Auto-discovered OpenAI-compatible endpoints (port -> endpoint cfg),
        # registered by the sniffer and scraped alongside LLM_ENDPOINTS.
        self.dynamic_endpoints: dict[int, dict[str, Any]] = {}
        # Revisions let the stream send model changes and one-point history
        # appends instead of repeating every 180-point series four times/second.
        self.last_history_ts: float = 0.0
        self.models_revision = 0
        self.history_revision = 0
        self.stream_models_revision = -1
        self.stream_history_revision = -1
        self.stream_snapshot: dict[str, Any] = {}

    def gpu_hist(self, idx: int) -> dict[str, deque]:
        h = self.history.get(idx)
        if h is None:
            h = {
                "util": deque(maxlen=HISTORY_POINTS),
                "mem": deque(maxlen=HISTORY_POINTS),
                "power": deque(maxlen=HISTORY_POINTS),
                "temp": deque(maxlen=HISTORY_POINTS),
            }
            self.history[idx] = h
        return h


STATE = State()


def scrape_endpoint(cfg: dict[str, Any], now: float) -> dict[str, Any]:
    try:
        ep_port = int(cfg["endpoint"].rstrip("/").rsplit(":", 1)[1])
    except Exception:
        ep_port = 0
    out = {
        "id": cfg["id"],
        "endpoint": cfg["endpoint"],
        "label": cfg["label"],
        "port": ep_port,
        "gpu_indices": list(cfg.get("gpu_indices") or []),
        "cfg_gpu_indices": list(cfg.get("gpu_indices") or []),
        "accent": cfg["accent"],
        "accent2": cfg["accent2"],
        "healthy": False,
        "error": "",
        "model_names": [],
        "max_model_len": None,
        "running": 0,
        "waiting": 0,
        "kv_cache_pct": 0.0,
        "prompt_tokens_total": 0,
        "generation_tokens_total": 0,
        "requests_total": 0,
        "input_tps": 0.0,
        "output_tps": 0.0,
        "total_tps": 0.0,
        "ttft_ms": None,
        "inter_token_ms": None,
        "request_rate": 0.0,
        "model_meta": [],
        "engine": "",
        "engine_version": "",
        "srv_model_path": "",
        "srv_ctx": None,
    }
    code, body = http_get(cfg["endpoint"].rstrip("/") + "/v1/models")
    if code == 200:
        try:
            j = json.loads(body)
            out["model_names"] = [m.get("id", "") for m in j.get("data", [])]
            out["model_meta"] = [{"id": d.get("id", ""), "root": d.get("root") or "",
                                  "max_model_len": d.get("max_model_len"), "created": d.get("created")}
                                 for d in j.get("data", [])]
            if out["model_names"]:
                out["healthy"] = True
        except Exception:
            pass
    else:
        out["error"] = body
    prefix = ""
    code2, body2 = http_get(cfg["endpoint"].rstrip("/") + "/metrics")
    if code2 == 200:
        out["healthy"] = True
        samples = parse_prom(body2)
        prefix = ("vllm" if any(s["name"].startswith("vllm:") for s in samples)
                  else "sglang" if any(s["name"].startswith("sglang:") for s in samples) else "")
        prompt_total = 0.0
        gen_total = 0.0
        req_total = 0.0
        running = 0.0
        waiting = 0.0
        kv = 0.0
        ttft = None
        itl = None
        for s in samples:
            n = s["name"]
            v = s["value"]
            # vLLM names
            if n in ("vllm:prompt_tokens_total", "sglang:prompt_tokens_total"):
                prompt_total += v
            elif n in ("vllm:generation_tokens_total", "sglang:generation_tokens_total"):
                gen_total += v
            elif n in ("vllm:num_requests_running", "sglang:num_running_reqs"):
                running = max(running, v)
            elif n in ("vllm:num_requests_waiting", "sglang:num_queue_reqs"):
                waiting = max(waiting, v)
            elif n in ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc", "sglang:token_usage"):
                kv = max(kv, v * (100.0 if v <= 1.0 else 1.0))
            elif n in ("vllm:request_success_total",):
                req_total += v
            elif n in ("vllm:time_to_first_token_seconds_sum",):
                # we won't compute avg here, leave as None
                pass
        out["prompt_tokens_total"] = int(prompt_total)
        out["generation_tokens_total"] = int(gen_total)
        out["requests_total"] = int(req_total)
        out["running"] = int(running)
        out["waiting"] = int(waiting)
        out["kv_cache_pct"] = float(kv)
        # delta TPS
        prev = STATE.prev_counters.get(cfg["id"])
        if prev:
            dt = max(now - prev["ts"], 1e-3)
            d_in = max(prompt_total - prev["prompt"], 0.0)
            d_out = max(gen_total - prev["gen"], 0.0)
            out["input_tps"] = d_in / dt
            out["output_tps"] = d_out / dt
            out["total_tps"] = out["input_tps"] + out["output_tps"]
        STATE.prev_counters[cfg["id"]] = {"prompt": prompt_total, "gen": gen_total, "ts": now}
    elif not out["healthy"]:
        if not out["error"]:
            out["error"] = body2
    if out["healthy"]:
        extras = probe_engine(cfg, prefix)
        lbl = (cfg.get("label") or "").lower()
        out["engine"] = extras["engine"] or ("vLLM" if "vllm" in lbl else "SGLang" if "sglang" in lbl else "")
        out["engine_version"] = extras["engine_version"]
        out["srv_model_path"] = extras["srv_model_path"]
        out["srv_ctx"] = extras["srv_ctx"]
    return out


def _collect_gpus_smi() -> list[dict[str, Any]]:
    # Fallback for hosts without a usable NVML Python binding.
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_rows = ex.submit(run_smi, GPU_QUERY, "gpu")
        fut_procs = ex.submit(run_smi, PROC_QUERY, "proc")
        rows = fut_rows.result()
        procs = fut_procs.result()
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    for r in procs:
        if len(r) < 4:
            continue
        uuid, pid, name, mem = r[0], r[1], r[2], r[3]
        by_uuid.setdefault(uuid, []).append({
            "pid": int(f(pid)),
            "process_name": name,
            "used_memory_mib": f(mem),
        })
    gpus = []
    for r in rows:
        if len(r) < 13:
            continue
        idx = int(f(r[0]))
        uuid = r[2]
        mem_used = f(r[5])
        mem_total = f(r[6]) or 1.0
        power_draw = f(r[8])
        power_limit = f(r[9]) or 1.0
        gpus.append({
            "index": idx,
            "name": r[1].replace("NVIDIA ", ""),
            "uuid": uuid,
            "utilization_gpu_pct": f(r[3]),
            "utilization_memory_pct": f(r[4]),
            "memory_used_mib": mem_used,
            "memory_total_mib": mem_total,
            "memory_pct": (mem_used / mem_total) * 100.0,
            "temperature_c": f(r[7]),
            "power_draw_w": power_draw,
            "power_limit_w": power_limit,
            "power_pct": (power_draw / power_limit) * 100.0,
            "fan_pct": f(r[10]),
            "graphics_clock_mhz": f(r[11]),
            "memory_clock_mhz": f(r[12]),
            "pstate": r[14] if len(r) > 14 else "",
            "pcie_gen": int(f(r[15])) if len(r) > 15 else 0,
            "pcie_gen_max": int(f(r[16])) if len(r) > 16 else 0,
            "pcie_width": int(f(r[17])) if len(r) > 17 else 0,
            "pcie_width_max": int(f(r[18])) if len(r) > 18 else 0,
            "pcie_aer": read_pcie_aer(r[19] if len(r) > 19 else ""),
            "processes": sorted(by_uuid.get(uuid, []), key=lambda p: -p["used_memory_mib"])[:10],
        })
    gpus.sort(key=lambda g: g["index"])
    return gpus


# ---------- Mock rig ----------
# BURNDECK_MOCK_GPUS=N (or [demo].mock_gpus) swaps NVML and the endpoint
# probes for a synthetic N-GPU tensor-parallel machine under a plausible
# serving load. Everything downstream — snapshot builder, SSE deltas, alerts,
# Prometheus export — runs the real code path, so this doubles as the CI
# smoke rig and as a full-fidelity preview on hardware that hasn't arrived.

_MOCK_T0 = time.time()
_MOCK_NAME = "RTX PRO 6000 Blackwell Workstation Edition"


def _wave(i: float, now: float, period: float, phase: float = 0.0) -> float:
    """Smooth 0..1 oscillation, phase-shifted per GPU so tiles don't lockstep."""
    return 0.5 + 0.5 * math.sin((now - _MOCK_T0) * 2.0 * math.pi / period + i * 0.9 + phase)


def mock_collect_gpus(n: int) -> list[dict[str, Any]]:
    now = time.time()
    gpus: list[dict[str, Any]] = []
    for i in range(n):
        # GPUs in a TP group ride one job envelope; per-GPU jitter keeps every
        # tile alive without the lockstep look of naive fakes.
        job = 0.42 + 0.5 * _wave(i // 8, now, 210.0)
        util = min(100.0, max(2.0, 100.0 * job * (0.86 + 0.14 * _wave(i, now, 7.0))
                              + 5.0 * _wave(i, now, 2.3, 1.7)))
        mem_total = 97887.0
        mem_used = mem_total * min(0.97, 0.55 + 0.32 * job + 0.02 * _wave(i, now, 31.0))
        power_limit = 600.0
        power = power_limit * (0.09 + 0.82 * (util / 100.0) ** 1.35)
        temp = 33.0 + 46.0 * (power / power_limit) + 3.0 * _wave(i, now, 47.0)
        gpus.append({
            "index": i,
            "name": _MOCK_NAME,
            "uuid": f"GPU-mock{i:02d}-0000-0000-0000-000000000000",
            "pcie_gen_max": 5,
            "pcie_width_max": 16,
            "bus_id": f"00000000:{0x18 + i:02X}:00.0",
            "power_limit_w": power_limit,
            "temperature_c": round(temp, 1),
            "fan_pct": round(min(100.0, max(0.0, (temp - 40.0) * 2.6)), 1),
            "graphics_clock_mhz": round(900.0 + 1900.0 * util / 100.0),
            "memory_clock_mhz": 14001.0,
            "pstate": "P0" if util > 40 else "P2" if util > 8 else "P8",
            "pcie_gen": 5,
            "pcie_width": 16,
            "pcie_aer": {"root_port": "", "correctable": {}, "total_correctable": 0,
                         "total_fatal": 0, "total_nonfatal": 0},
            "processes": [{"pid": 41000 + i, "process_name": "vllm::EngineCore",
                           "used_memory_mib": mem_used * 0.985}],
            "utilization_gpu_pct": round(util, 1),
            "utilization_memory_pct": round(util * 0.8, 1),
            "memory_used_mib": mem_used,
            "memory_total_mib": mem_total,
            "memory_pct": mem_used / mem_total * 100.0,
            "power_draw_w": round(power, 1),
            "power_pct": power / power_limit * 100.0,
        })
    return gpus


def _mock_model(now: float, *, mid: str, port: int, label: str, gpus: list[int],
                names: list[str], engine: str, version: str, out_tps: float,
                in_tps: float, kv: float, running: int, waiting: int, ctx: int,
                intel: dict[str, Any], uptime_s: float, vram_mib: float) -> dict[str, Any]:
    """A scrape_endpoint-shaped record with resolve/enrich fields pre-filled."""
    intel = dict(intel)
    intel.update({"engine": engine, "engine_version": version, "ctx_serving": ctx,
                  "tp": len(gpus), "uptime_s": uptime_s, "vram_mib": vram_mib})
    total = int((now - _MOCK_T0 + 3 * 86400) * (out_tps + in_tps) * 0.4)
    return {
        "id": mid, "endpoint": f"http://127.0.0.1:{port}", "label": label, "port": port,
        "gpu_indices": gpus, "cfg_gpu_indices": gpus,
        "accent": _ACCENT_CYCLE[0][0] if port == 8000 else _ACCENT_CYCLE[1][0],
        "accent2": _ACCENT_CYCLE[0][1] if port == 8000 else _ACCENT_CYCLE[1][1],
        "healthy": True, "error": "",
        "model_names": names,
        "model_meta": [{"id": names[0], "root": "", "max_model_len": ctx, "created": int(_MOCK_T0)}],
        "max_model_len": ctx,
        "running": running, "waiting": waiting, "kv_cache_pct": round(kv, 1),
        "prompt_tokens_total": int(total * 0.7), "generation_tokens_total": int(total * 0.3),
        "requests_total": int(total / 900), "request_rate": 0.0,
        "input_tps": round(in_tps, 2), "output_tps": round(out_tps, 2),
        "total_tps": round(in_tps + out_tps, 2),
        "ttft_ms": None, "inter_token_ms": None,
        "engine": engine, "engine_version": version,
        "srv_model_path": "", "srv_ctx": ctx,
        "worker_pids": [41000 + g for g in gpus],
        "vram_mib": vram_mib,
        "intel": intel,
    }


def mock_models(now: float) -> list[dict[str, Any]]:
    n = MOCK_GPUS
    load = 0.42 + 0.5 * _wave(0, now, 210.0)
    gust = 0.75 + 0.25 * _wave(0, now, 23.0)
    out_tps = (30.0 + 240.0 * load * gust) * max(n, 1) / 8.0
    in_tps = out_tps * (2.1 + 1.2 * _wave(0, now, 61.0))
    models = [_mock_model(
        now, mid="mock-vllm", port=8000,
        label="Qwen3-235B-A22B FP8 · vLLM",
        gpus=list(range(n)), names=["qwen3-235b-a22b-fp8"],
        engine="vLLM", version="0.10.2",
        out_tps=out_tps, in_tps=in_tps,
        kv=22.0 + 58.0 * load + 6.0 * _wave(1.0, now, 17.0),
        running=int(1 + 5 * load),
        waiting=max(0, int((_wave(0, now, 97.0) - 0.62) * 36.0)),
        ctx=262144,
        intel={"display_name": "Qwen3 235B A22B", "family": "Qwen",
               "creator": "Alibaba Cloud", "origin": "Hangzhou · CN",
               "blurb": "Alibaba's Tongyi Qianwen family — open-weight dense and "
                        "MoE models spanning chat, coding, math and vision.",
               "license": "apache-2.0", "params": "235B",
               "moe": {"experts": 128, "active": 8},
               "ctx_native": 262144, "quant": "FP8", "dtype": "bfloat16",
               "layers": 94, "hidden": 4096, "heads": 64, "attn": "GQA",
               "weights_gb": 235},
        uptime_s=now - _MOCK_T0 + 3 * 86400,
        vram_mib=61000.0 * max(n, 1),
    )]
    if n >= 2:
        draft_gpus = list(range(min(2, n)))
        models.append(_mock_model(
            now, mid="mock-sglang", port=30000,
            label="Llama-3.3-70B AWQ · SGLang",
            gpus=draft_gpus, names=["llama-3.3-70b-instruct-awq"],
            engine="SGLang", version="0.4.9",
            out_tps=out_tps * 0.28, in_tps=in_tps * 0.18,
            kv=30.0 + 40.0 * _wave(2.0, now, 41.0),
            running=int(2 * load), waiting=0, ctx=131072,
            intel={"display_name": "Llama 3.3 70B Instruct", "family": "Llama",
                   "creator": "Meta AI", "origin": "Menlo Park · US",
                   "blurb": "Meta's Llama herd — the open-weight family that "
                            "kicked off the local-LLM era.",
                   "license": "llama-3.3", "params": "70B",
                   "ctx_native": 131072, "quant": "AWQ INT4", "dtype": "bfloat16",
                   "layers": 80, "hidden": 8192, "heads": 64, "attn": "GQA",
                   "weights_gb": 40},
            uptime_s=now - _MOCK_T0 + 7200,
            vram_mib=21000.0 * len(draft_gpus),
        ))
    return models


def collect_gpus() -> list[dict[str, Any]]:
    if MOCK_GPUS:
        return mock_collect_gpus(MOCK_GPUS)
    nvml = _load_nvml()
    if nvml is not None and _NVML_POOL is not None:
        try:
            futures = [
                _NVML_POOL.submit(_nvml_device, nvml, index, handle)
                for index, handle in enumerate(_NVML_HANDLES)
            ]
            gpus = [future.result() for future in futures]
            if gpus:
                gpus.sort(key=lambda g: g["index"])
                return gpus
        except Exception:
            _drop_nvml()
    return _collect_gpus_smi()


# ---------- GPU ↔ endpoint attribution ----------

_anc_cache: dict[int, tuple[float, str]] = {}


def _ancestor_cmdlines(pid: int, max_depth: int = 12) -> str:
    """Joined cmdlines of pid and its ancestors (world-readable even for root procs)."""
    now = time.time()
    hit = _anc_cache.get(pid)
    if hit and now - hit[0] < 60:
        return hit[1]
    parts: list[str] = []
    cur = pid
    for _ in range(max_depth):
        if cur <= 1:
            break
        try:
            with open(f"/proc/{cur}/cmdline", "rb") as fh:
                parts.append(fh.read().replace(b"\0", b" ").decode("utf-8", "replace"))
        except Exception:
            pass
        try:
            with open(f"/proc/{cur}/stat") as fh:
                cur = int(fh.read().rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    text = "\n".join(parts)
    _anc_cache[pid] = (now, text)
    return text


def resolve_model_gpus(models: list[dict[str, Any]], gpus: list[dict[str, Any]]) -> None:
    """Set m['gpu_indices'] / m['worker_pids'] / m['vram_mib'] from the GPUs and
    compute processes actually running each endpoint's workers.

    nvidia-smi gives us each GPU's compute PIDs; the serving stack's launch args
    (--port / --served-model-name) live on an ancestor of those workers. Match
    order per endpoint: port in ancestor cmdline → served model name → engine
    keyword on the worker (only when that engine has a single healthy endpoint)
    → static config fallback. VRAM is the sum over matched (gpu, pid) entries,
    so a TP master with contexts on all cards is counted once per card.
    """
    gpu_procs = {
        g["index"]: [(p["pid"], p["process_name"] or "", p["used_memory_mib"], _ancestor_cmdlines(p["pid"]))
                     for p in g["processes"]]
        for g in gpus
    }

    def collect(pred) -> list[tuple[int, int, float]]:
        return [(i, pid, mem) for i, procs in gpu_procs.items()
                for pid, nm, mem, anc in procs if pred(nm, anc)]

    for m in models:
        matched: list[tuple[int, int, float]] = []
        if m.get("healthy"):
            port = m.get("port") or 0
            port_re = re.compile(rf"(?:--port[= ]\s*|:){port}(?!\d)") if port else None
            names = [n for n in (m.get("model_names") or []) if len(n) >= 3]
            if port_re:
                matched = collect(lambda nm, anc: port_re.search(anc))
            if not matched and names:
                matched = collect(lambda nm, anc: any(n in anc for n in names))
            if not matched:
                kw = next((k for k in ("vllm", "sglang") if k in (m.get("label") or "").lower()), None)
                rivals = [x for x in models if x is not m and x.get("healthy")
                          and kw and kw in (x.get("label") or "").lower()]
                if kw and not rivals:
                    matched = collect(lambda nm, anc: kw in nm.lower() or kw in anc.lower())
        idxs = sorted({i for i, _, _ in matched})
        m["gpu_indices"] = idxs if idxs else list(m.get("cfg_gpu_indices") or [])
        m["worker_pids"] = sorted({pid for _, pid, _ in matched})
        m["vram_mib"] = sum(mem for _, _, mem in matched) or None


def build_snapshot() -> dict[str, Any]:
    now = time.time()
    gpus = collect_gpus()
    # Read the cached LLM-endpoint scrape (refreshed by the slower llm_poller).
    # This keeps the fast GPU tick free of blocking HTTP to the serving stacks.
    models = STATE.models_cache
    if not MOCK_GPUS:
        resolve_model_gpus(models, gpus)
        enrich_models(models, now)
    # histories — gated to HISTORY_INTERVAL so the chart window and payload
    # size stay constant regardless of how fast the GPU tiles refresh.
    record_history = (now - STATE.last_history_ts) >= HISTORY_INTERVAL
    for g in gpus:
        h = STATE.gpu_hist(g["index"])
        if record_history:
            h["util"].append(g["utilization_gpu_pct"])
            h["mem"].append(g["memory_pct"])
            h["power"].append(g["power_pct"])
            h["temp"].append(g["temperature_c"])
        g["history"] = {k: list(v) for k, v in h.items()}
    total_tps = sum(m["total_tps"] for m in models)
    total_out_tps = sum(m["output_tps"] for m in models)
    total_in_tps = sum(m["input_tps"] for m in models)
    avg_util = sum(g["utilization_gpu_pct"] for g in gpus) / max(len(gpus), 1)
    avg_mem = sum(g["memory_pct"] for g in gpus) / max(len(gpus), 1)
    tot_power = sum(g["power_draw_w"] for g in gpus)
    tot_power_limit = sum(g["power_limit_w"] for g in gpus)
    if record_history:
        STATE.global_hist["total_tps"].append(total_tps)
        STATE.global_hist["gpu_util_avg"].append(avg_util)
        STATE.global_hist["mem_pct_avg"].append(avg_mem)
        STATE.global_hist["power_total"].append(tot_power)
        STATE.last_history_ts = now
        STATE.history_revision += 1
    # per-GPU TPS share: divide model TPS evenly across its gpu_indices
    per_gpu_tps = {g["index"]: {"input": 0.0, "output": 0.0, "total": 0.0, "models": []} for g in gpus}
    for m in models:
        idxs = [i for i in m["gpu_indices"] if i in per_gpu_tps]
        if not idxs or not m["healthy"]:
            continue
        n = len(idxs)
        for i in idxs:
            per_gpu_tps[i]["input"] += m["input_tps"] / n
            per_gpu_tps[i]["output"] += m["output_tps"] / n
            per_gpu_tps[i]["total"] += m["total_tps"] / n
            per_gpu_tps[i]["models"].append({"id": m["id"], "label": m["label"], "names": m["model_names"], "accent": m["accent"]})
    for g in gpus:
        g["tps"] = per_gpu_tps.get(g["index"], {"input": 0.0, "output": 0.0, "total": 0.0, "models": []})
    snap = {
        "generated_at": utc_iso(),
        "host": os.uname().nodename,
        "driver": "mock" if MOCK_GPUS else get_driver(),
        "cuda": "" if MOCK_GPUS else get_cuda(),
        "uptime_seconds": uptime_seconds(),
        "poll_seconds": GPU_POLL_SECONDS,
        "llm_poll_seconds": LLM_POLL_SECONDS,
        "gpus": gpus,
        "models": models,
        "sniffed": STATE.sniffed_cache,
        "totals": {
            "input_tps": total_in_tps,
            "output_tps": total_out_tps,
            "total_tps": total_tps,
            "gpu_util_avg": avg_util,
            "memory_pct_avg": avg_mem,
            "power_draw_w": tot_power,
            "power_limit_w": tot_power_limit,
            "power_pct": (tot_power / tot_power_limit * 100.0) if tot_power_limit else 0.0,
            "history": {k: list(v) for k, v in STATE.global_hist.items()},
        },
    }
    return snap


def build_stream_update(snap: dict[str, Any]) -> dict[str, Any]:
    """Build a keyed delta; new subscribers still receive a full snapshot."""
    previous = STATE.stream_snapshot
    frame: dict[str, Any] = {
        "type": "update",
        "generated_at": snap["generated_at"],
        "uptime_seconds": snap["uptime_seconds"],
    }
    for key in ("host", "driver", "cuda", "poll_seconds", "llm_poll_seconds"):
        if previous.get(key) != snap.get(key):
            frame[key] = snap.get(key)

    previous_gpus = {g["index"]: g for g in previous.get("gpus", [])}
    current_indices = set()
    gpu_updates = []
    for gpu in snap["gpus"]:
        index = gpu["index"]
        current_indices.add(index)
        old = previous_gpus.get(index, {})
        changed = {"index": index}
        changed.update({
            key: value
            for key, value in gpu.items()
            if key != "history" and old.get(key) != value
        })
        if len(changed) > 1:
            gpu_updates.append(changed)
    if gpu_updates:
        frame["gpus"] = gpu_updates
    removed = sorted(set(previous_gpus) - current_indices)
    if removed:
        frame["gpu_remove"] = removed

    previous_totals = previous.get("totals", {})
    totals = {
        key: value
        for key, value in snap["totals"].items()
        if key != "history" and previous_totals.get(key) != value
    }
    if totals:
        frame["totals"] = totals

    if STATE.stream_models_revision != STATE.models_revision:
        frame["models"] = snap["models"]
        frame["sniffed"] = snap["sniffed"]
        STATE.stream_models_revision = STATE.models_revision

    if STATE.stream_history_revision != STATE.history_revision:
        frame["history"] = {
            "max_points": HISTORY_POINTS,
            "gpus": {
                str(gpu["index"]): {
                    key: values[-1]
                    for key, values in gpu.get("history", {}).items()
                    if values
                }
                for gpu in snap["gpus"]
            },
            "totals": {
                key: values[-1]
                for key, values in snap["totals"].get("history", {}).items()
                if values
            },
        }
        STATE.stream_history_revision = STATE.history_revision

    STATE.stream_snapshot = snap
    return frame


_driver_cache: dict[str, Any] = {"ts": 0, "value": ""}


def get_driver() -> str:
    if time.time() - _driver_cache["ts"] < 3600 and _driver_cache["value"]:
        return _driver_cache["value"]
    nvml = _load_nvml()
    if nvml is not None:
        v = str(_nvml_call(nvml.nvmlSystemGetDriverVersion, ""))
    else:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            v = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        except Exception:
            v = ""
    _driver_cache.update(ts=time.time(), value=v)
    return v


_cuda_cache: dict[str, Any] = {"ts": 0, "value": ""}


def get_cuda() -> str:
    if time.time() - _cuda_cache["ts"] < 3600 and _cuda_cache["value"]:
        return _cuda_cache["value"]
    nvml = _load_nvml()
    if nvml is not None:
        get_version = getattr(
            nvml,
            "nvmlSystemGetCudaDriverVersion_v2",
            nvml.nvmlSystemGetCudaDriverVersion,
        )
        raw = int(_nvml_call(get_version, 0))
        v = f"{raw // 1000}.{(raw % 1000) // 10}" if raw else ""
    else:
        try:
            out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=2, check=False)
            m = re.search(r"CUDA Version:\s*([0-9.]+)", out.stdout or "")
            v = m.group(1) if m else ""
        except Exception:
            v = ""
    _cuda_cache.update(ts=time.time(), value=v)
    return v


def uptime_seconds() -> float:
    try:
        with open("/proc/uptime") as fp:
            return float(fp.read().split()[0])
    except Exception:
        return 0.0


# ---------- Auth ----------

SESSION_TTL = 60 * 60 * 24 * 30  # 30 days
COOKIE_NAME = "burndeck_session"
SECRET_FILE = DATA_DIR / ".session-secret"
GENERATED_AUTH_FILE = DATA_DIR / ".auth-generated"

AUTH_ENABLED = bool(cfg("auth", "enabled", True, "BURNDECK_AUTH"))
AUTH_USER = str(cfg("auth", "user", "operator", "BURNDECK_AUTH_USER"))
AUTH_PASS_SHA256 = str(cfg("auth", "password_sha256", "", "BURNDECK_AUTH_PASSWORD_SHA256") or "")
if not AUTH_PASS_SHA256:
    _plain = str(cfg("auth", "password", "", "BURNDECK_AUTH_PASSWORD") or "")
    if _plain:
        AUTH_PASS_SHA256 = hashlib.sha256(_plain.encode()).hexdigest()


def _ensure_credentials() -> None:
    """First boot with auth on and no password configured: mint a random one,
    print it once, and persist it (0600) so restarts keep the same secret."""
    global AUTH_USER, AUTH_PASS_SHA256
    if not AUTH_ENABLED or AUTH_PASS_SHA256:
        return
    try:
        user, _, pw = GENERATED_AUTH_FILE.read_text().strip().partition(":")
        if user and pw:
            AUTH_USER = user
            AUTH_PASS_SHA256 = hashlib.sha256(pw.encode()).hexdigest()
            print(f"burndeck: using generated credentials from {GENERATED_AUTH_FILE} (user {user!r})")
            return
    except FileNotFoundError:
        pass
    pw = secrets.token_urlsafe(12)
    GENERATED_AUTH_FILE.write_text(f"{AUTH_USER}:{pw}\n")
    try:
        GENERATED_AUTH_FILE.chmod(0o600)
    except OSError:
        pass
    AUTH_PASS_SHA256 = hashlib.sha256(pw.encode()).hexdigest()
    print(f"burndeck: no password configured — generated credentials  user: {AUTH_USER}  password: {pw}")
    print(f"burndeck: stored in {GENERATED_AUTH_FILE}; set [auth] in config.toml to choose your own")


_ensure_credentials()


def _session_secret() -> bytes:
    try:
        s = SECRET_FILE.read_text().strip()
        if s:
            return s.encode()
    except Exception:
        pass
    s = secrets.token_hex(32)
    SECRET_FILE.write_text(s)
    try:
        SECRET_FILE.chmod(0o600)
    except Exception:
        pass
    return s.encode()


SESSION_SECRET = _session_secret()


def sign_session(user: str, exp: int) -> str:
    msg = f"{user}.{exp}"
    sig = hmac.new(SESSION_SECRET, msg.encode(), hashlib.sha256).hexdigest()
    return f"{msg}.{sig}"


def verify_session(token: str) -> bool:
    try:
        user, exp, sig = token.rsplit(".", 2)
        expect = hmac.new(SESSION_SECRET, f"{user}.{exp}".encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expect) and user == AUTH_USER and int(exp) > time.time()
    except Exception:
        return False


APP_VERSION = "1.0.0"

METRICS_ENABLED = bool(cfg("metrics", "enabled", True, "BURNDECK_METRICS"))
METRICS_PUBLIC = bool(cfg("metrics", "public", True, "BURNDECK_METRICS_PUBLIC"))

PUBLIC_PREFIXES = ("/login", "/api/auth", "/healthz", "/static/themes/", "/static/login",
                   "/static/manifest", "/static/icon", "/static/demo", "/favicon") + (
                   ("/metrics",) if METRICS_PUBLIC else ())


# ---------- Alerts ----------
# Server-side counterpart of the UI event ticker: hysteresis edges over each
# snapshot, delivered to a webhook (Discord, Slack, or generic JSON).
# Detection is synchronous in the poll loop (pure arithmetic); delivery runs
# on a single-worker pool so a slow webhook can never stall telemetry.

class AlertEngine:
    def __init__(self) -> None:
        a = CONFIG.get("alerts", {})
        self.url = str(os.environ.get("BURNDECK_WEBHOOK") or a.get("webhook") or "")
        self.fmt = str(a.get("format", "discord"))
        self.events = set(a.get("events") or [
            "gpu_hot", "gpu_vram", "power_ceiling", "model_health",
            "pcie", "aer_fatal", "queue"])
        self.hot_c = float(a.get("gpu_hot_c", 83.0))
        self.vram_pct = float(a.get("vram_pct", 96.0))
        self.power_pct = float(a.get("power_pct", 95.0))
        self.cooldown = float(a.get("cooldown_seconds", 300.0))
        self.latch: dict[str, bool] = {}
        self.fired: dict[str, float] = {}
        self.health: dict[str, bool] = {}
        self.pcie: dict[int, bool] = {}
        self.aer: dict[int, float] = {}
        self.pool = ThreadPoolExecutor(max_workers=1)

    def edge(self, key: str, v: float, hi: float, lo: float) -> int:
        """Rising edge with hysteresis: 1 once at v >= hi, -1 once back below lo."""
        if self.latch.get(key):
            if v < lo:
                self.latch[key] = False
                return -1
            return 0
        if v >= hi:
            self.latch[key] = True
            return 1
        return 0

    def fire(self, key: str, sev: str, text: str) -> None:
        now = time.time()
        if now - self.fired.get(key, 0.0) < self.cooldown:
            return
        self.fired[key] = now
        host = STATE.snapshot.get("host", "")
        if self.fmt == "slack":
            payload: dict[str, Any] = {"text": f"[{sev}] burndeck @ {host} — {text}"}
        elif self.fmt == "json":
            payload = {"source": "burndeck", "host": host, "severity": sev,
                       "event": key, "text": text, "ts": utc_iso()}
        else:  # discord
            payload = {"content": f"**burndeck** @ `{host}` · **{sev.upper()}** — {text}"}
        body = json.dumps(payload).encode()

        def _post() -> None:
            try:
                req = urllib.request.Request(
                    self.url, data=body,
                    headers={"Content-Type": "application/json", "User-Agent": "burndeck"})
                urllib.request.urlopen(req, timeout=4).read()
            except Exception as exc:
                print("alert post err:", exc, file=sys.stderr)

        self.pool.submit(_post)

    def scan(self, snap: dict[str, Any]) -> None:
        if not self.url:
            return
        ev = self.events
        t = snap["totals"]
        if "power_ceiling" in ev and self.edge("clpwr", t["power_pct"], self.power_pct, self.power_pct - 13) > 0:
            self.fire("clpwr", "warn",
                      f"cluster at power ceiling — {t['power_draw_w']:.0f}W / {t['power_limit_w']:.0f}W")
        for g in snap["gpus"]:
            i = g["index"]
            if "gpu_hot" in ev:
                e = self.edge(f"hot{i}", g["temperature_c"], self.hot_c, self.hot_c - 9)
                if e > 0:
                    self.fire(f"hot{i}", "crit",
                              f"GPU{i} hot — {g['temperature_c']:.0f}°C at {g['power_draw_w']:.0f}W")
                elif e < 0:
                    self.fire(f"cool{i}", "good", f"GPU{i} back to {g['temperature_c']:.0f}°C")
            if "gpu_vram" in ev and self.edge(f"vram{i}", g["memory_pct"], self.vram_pct, self.vram_pct - 8) > 0:
                self.fire(f"vram{i}", "warn",
                          f"GPU{i} VRAM {g['memory_pct']:.0f}% — {g['memory_used_mib'] / 1024:.1f} GiB used")
            if "pcie" in ev:
                bad = ((g["pcie_gen"] >= 2 and g["pcie_gen"] < g["pcie_gen_max"]
                        and g["utilization_gpu_pct"] > 15)
                       or (g["pcie_width"] > 0 and g["pcie_width"] < g["pcie_width_max"]))
                prev = self.pcie.get(i)
                if prev is not None and bad != prev:
                    self.fire(f"pcie{i}-{bad}", "crit" if bad else "good",
                              f"GPU{i} PCIe {'DEGRADED' if bad else 'restored'} — "
                              f"Gen{g['pcie_gen']} ×{g['pcie_width']}")
                self.pcie[i] = bad
            if "aer_fatal" in ev:
                aer = g.get("pcie_aer") or {}
                fatal = float(aer.get("total_fatal") or 0) + float(aer.get("total_nonfatal") or 0)
                prev_f = self.aer.get(i)
                if prev_f is not None and fatal > prev_f:
                    self.fire(f"aer{i}", "crit",
                              f"GPU{i} PCIe AER +{fatal - prev_f:.0f} fatal/non-fatal errors")
                self.aer[i] = fatal
        for mrec in snap["models"]:
            mid = mrec["id"]
            if "model_health" in ev:
                prev = self.health.get(mid)
                if prev is not None and prev != mrec["healthy"]:
                    self.fire(f"mh-{mid}-{mrec['healthy']}", "good" if mrec["healthy"] else "crit",
                              f"{mrec['label']} {'back ONLINE' if mrec['healthy'] else 'OFFLINE'}")
                self.health[mid] = mrec["healthy"]
            if mrec["healthy"] and "queue" in ev and self.edge(f"q-{mid}", float(mrec["waiting"]), 8, 2) > 0:
                self.fire(f"q-{mid}", "warn", f"{mrec['label']} queue building — {mrec['waiting']} waiting")


ALERTS = AlertEngine()


# ---------- FastAPI ----------


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()

    def _scrape_all():
        now = time.time()
        if MOCK_GPUS:
            return mock_models(now)
        cfgs = LLM_ENDPOINTS + list(STATE.dynamic_endpoints.values())
        return [scrape_endpoint(c, now) for c in cfgs]

    async def llm_poller():
        # Slow cadence: probe the LLM serving stacks over HTTP and cache the
        # result. Runs off-thread so a slow/hung endpoint can never stall the
        # fast GPU feed. TPS deltas stay accurate because scrape_endpoint
        # timestamps against its own previous counters.
        while True:
            try:
                models = await loop.run_in_executor(None, _scrape_all)
                if MOCK_GPUS:
                    sniffed: list[dict[str, Any]] = []
                else:
                    sniffed = await loop.run_in_executor(
                        None, sniff_models, STATE.snapshot.get("gpus") or [], models)
                    await loop.run_in_executor(None, update_dynamic_endpoints, sniffed)
                STATE.models_cache = models
                STATE.sniffed_cache = sniffed
                STATE.models_revision += 1
            except Exception as exc:
                print("llm poll err:", exc)
            await asyncio.sleep(LLM_POLL_SECONDS)

    async def gpu_poller():
        # Fast cadence: NVML with an nvidia-smi fallback. Reads the cached LLM
        # scrape, so model endpoint latency never delays GPU telemetry.
        # Sleep is debited by the build time so the loop targets GPU_POLL_SECONDS
        # as the actual cadence rather than (build + interval).
        while True:
            t0 = time.time()
            try:
                snap = await loop.run_in_executor(None, build_snapshot)
                STATE.snapshot = snap
                frame = build_stream_update(snap)
                payload = f"data:{json.dumps(frame, separators=(',', ':'))}\n\n".encode()
                for q in list(STATE.subscribers):
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        # Slow client: drop its oldest frame and keep the newest.
                        # NEVER unsubscribe a live connection here — its generator
                        # blocks on q.get() forever while the socket stays open, so
                        # the browser freezes on stale data (e.g. 100% util) with no
                        # error event to trigger a reconnect. Dead connections are
                        # cleaned up by gen()'s finally on disconnect.
                        try:
                            q.get_nowait()
                            q.put_nowait(payload)
                        except (asyncio.QueueEmpty, asyncio.QueueFull):
                            pass
                try:
                    ALERTS.scan(snap)
                except Exception as exc:
                    print("alert scan err:", exc, file=sys.stderr)
            except Exception as exc:
                print("poll err:", exc)
            await asyncio.sleep(max(0.0, GPU_POLL_SECONDS - (time.time() - t0)))

    # Prime the LLM cache once before the GPU feed starts so the first frames
    # already carry TPS/model data.
    try:
        STATE.models_cache = await loop.run_in_executor(None, _scrape_all)
        STATE.models_revision += 1
    except Exception as exc:
        print("llm prime err:", exc)

    tasks = [asyncio.create_task(llm_poller()), asyncio.create_task(gpu_poller())]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()


app = FastAPI(lifespan=_lifespan)


@app.middleware("http")
async def auth_gate(request, call_next):
    path = request.url.path
    if AUTH_ENABLED and not any(path == p.rstrip("/") or path.startswith(p) for p in PUBLIC_PREFIXES):
        token = request.cookies.get(COOKIE_NAME, "")
        if not verify_session(token):
            if path.startswith("/api/"):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            from urllib.parse import quote
            return RedirectResponse(f"/login?redirect={quote(path)}", status_code=302)
    return await call_next(request)


# Version token for immutable static assets; bump when shipping UI changes.
# HTML is always no-cache (ETag revalidation), so a bump here propagates on
# the next page load through every layer (nginx, CDN, browser, direct :8801).
ASSET_VERSION = "20260811c"


@app.middleware("http")
async def cache_headers(request, call_next):
    resp = await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
    elif path.startswith("/static/"):
        if request.query_params.get("v"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            resp.headers["Cache-Control"] = "no-cache"
    else:
        resp.headers["Cache-Control"] = "no-cache"
    return resp




@app.get("/api/snapshot")
def api_snapshot():
    return JSONResponse(STATE.snapshot)


@app.get("/api/stream")
async def api_stream():
    q: asyncio.Queue = asyncio.Queue(maxsize=4)
    STATE.subscribers.add(q)

    async def gen():
        try:
            # Yield the baseline outside the bounded delta queue. Even if the
            # producer fills that queue during connection setup, every client
            # receives a complete snapshot before its first delta.
            yield f"data:{json.dumps(STATE.snapshot, separators=(',', ':'))}\n\n".encode()
            while True:
                chunk = await q.get()
                yield chunk
        finally:
            STATE.subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@app.get("/healthz")
def health():
    return {"ok": True, "ts": utc_iso(), "version": APP_VERSION,
            "mock": bool(MOCK_GPUS), "gpus": len(STATE.snapshot.get("gpus") or [])}


@app.get("/metrics")
def metrics():
    """Prometheus text exposition of the live snapshot — scrape straight into
    an existing Grafana stack alongside the built-in UI."""
    if not METRICS_ENABLED:
        return Response("metrics disabled\n", status_code=404, media_type="text/plain")
    snap = STATE.snapshot
    out: list[str] = []

    def emit(name: str, val: Any, labels: str = "") -> None:
        v = float(val)
        out.append(f"burndeck_{name}{{{labels}}} {v:.6g}" if labels else f"burndeck_{name} {v:.6g}")

    def lesc(s: Any) -> str:
        return str(s).replace("\\", "\\\\").replace('"', '\\"')

    emit("up", 1)
    for g in snap.get("gpus", []):
        lb = f'gpu="{g["index"]}",name="{lesc(g["name"])}",uuid="{lesc(g["uuid"])}"'
        for key, metric in (
            ("utilization_gpu_pct", "gpu_utilization_percent"),
            ("memory_used_mib", "gpu_memory_used_mib"),
            ("memory_total_mib", "gpu_memory_total_mib"),
            ("temperature_c", "gpu_temperature_celsius"),
            ("power_draw_w", "gpu_power_watts"),
            ("power_limit_w", "gpu_power_limit_watts"),
            ("fan_pct", "gpu_fan_percent"),
            ("graphics_clock_mhz", "gpu_sm_clock_mhz"),
            ("memory_clock_mhz", "gpu_memory_clock_mhz"),
            ("pcie_gen", "gpu_pcie_gen"),
            ("pcie_width", "gpu_pcie_width"),
        ):
            emit(metric, g.get(key) or 0, lb)
        aer = g.get("pcie_aer") or {}
        emit("gpu_pcie_aer_correctable_total", aer.get("total_correctable") or 0, lb)
        emit("gpu_pcie_aer_fatal_total",
             (aer.get("total_fatal") or 0) + (aer.get("total_nonfatal") or 0), lb)
    for mrec in snap.get("models", []):
        lb = f'id="{lesc(mrec["id"])}",label="{lesc(mrec["label"])}"'
        emit("model_healthy", 1 if mrec.get("healthy") else 0, lb)
        for key, metric in (
            ("input_tps", "model_input_tps"),
            ("output_tps", "model_output_tps"),
            ("kv_cache_pct", "model_kv_cache_percent"),
            ("running", "model_requests_running"),
            ("waiting", "model_requests_waiting"),
            ("prompt_tokens_total", "model_prompt_tokens_total"),
            ("generation_tokens_total", "model_generation_tokens_total"),
        ):
            emit(metric, mrec.get(key) or 0, lb)
    t = snap.get("totals", {})
    for key, metric in (
        ("total_tps", "cluster_tps"),
        ("gpu_util_avg", "cluster_utilization_percent"),
        ("power_draw_w", "cluster_power_watts"),
        ("power_limit_w", "cluster_power_limit_watts"),
    ):
        emit(metric, t.get(key) or 0)
    return Response("\n".join(out) + "\n",
                    media_type="text/plain; version=0.0.4; charset=utf-8")


def _html_file(request: Request, name: str):
    """FileResponse with conditional-request support: bare FileResponse never
    answers 304, so no-cache HTML would re-ship the full body on every load.
    ETag matches Starlette's stat-based algorithm for consistency."""
    path = STATIC_ROOT / name
    stat = path.stat()
    etag = f'"{hashlib.md5(f"{stat.st_mtime}-{stat.st_size}".encode(), usedforsecurity=False).hexdigest()}"'
    if etag in (request.headers.get("if-none-match") or ""):
        return Response(status_code=304, headers={"ETag": etag})
    return FileResponse(path, stat_result=stat)


@app.get("/login")
def login_page(request: Request):
    return _html_file(request, "login.html")


@app.post("/api/auth")
async def api_auth(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    user = str(body.get("login", "")).strip()
    pw = str(body.get("password", ""))
    pw_hash = hashlib.sha256(pw.encode()).hexdigest()
    ok = hmac.compare_digest(user, AUTH_USER) and hmac.compare_digest(pw_hash, AUTH_PASS_SHA256)
    if not ok:
        return JSONResponse({"error": "denied"}, status_code=401)
    exp = int(time.time()) + SESSION_TTL
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        COOKIE_NAME, sign_session(AUTH_USER, exp),
        max_age=SESSION_TTL, httponly=True, samesite="lax", path="/",
    )
    return resp


@app.get("/api/logout")
def api_logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


# static
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@app.get("/")
def root(request: Request):
    return _html_file(request, "index.html")


if __name__ == "__main__":
    import uvicorn

    host = str(cfg("server", "host", "127.0.0.1", "BURNDECK_HOST"))
    port = int(cfg("server", "port", 8801, "BURNDECK_PORT"))
    if MOCK_GPUS:
        print(f"burndeck: mock mode — synthetic {MOCK_GPUS}-GPU rig, no NVML or endpoint probes")
    print(f"burndeck: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
