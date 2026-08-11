# burndeck

Cinematic GPU + LLM telemetry for multi-GPU rigs.

One Python file watches your GPUs over NVML and your model servers over their
own APIs. One HTML file renders it like a film-grade ops console — animated
WebGL wallpapers, analog gauges, live tok/s, and an event ticker that
narrates your cluster's day. Built on a 4× RTX PRO 6000 Blackwell box,
designed to scale to TP8 rigs and beyond.

**Try it right now, no install — the demo runs a synthetic rig in your browser:**
[8 GPUs](https://jcartu.github.io/burndeck/?gpus=8) ·
[16 GPUs](https://jcartu.github.io/burndeck/?gpus=16) ·
[single card](https://jcartu.github.io/burndeck/?gpus=1)

## Features

**GPU telemetry**
- 4 Hz per-GPU feed: utilization speedometer with a peak needle and a red
  zone, an animated mercury thermometer (minus the mercury), memory / power /
  fan bars, SM and MEM clocks, P-states, per-process VRAM tables.
- NVML first, `nvidia-smi` subprocess fallback, any GPU count, hot-unplug safe.
- PCIe link watch: current vs. max generation and width, degradation badges,
  and per-card AER error counters read from the root port — the stuff you
  normally only discover after three days of "why is TP slow".
- SPIN-WAIT chip: flags a GPU at 100% utilization on idle wattage — the NCCL
  barrier workout. Maximum effort, zero progress. The wallpaper knows the
  difference too, and won't summon the storm for it.

**LLM telemetry**
- Input / output tok/s, KV-cache %, running and queued requests from vLLM and
  SGLang metrics; cluster totals; three-minute history charts; session
  throughput records in the ticker.
- Per-GPU TPS share attributed by walking process ancestry — TP workers,
  launcher scripts, and dockerized engines all land on the right card.
- Zero-config discovery: any process holding ≥64 MB of model weights on a GPU
  is detected and identified; any local port speaking the OpenAI API gets
  scraped without being declared. burndeck is nosy so you don't have to be.
- Model intelligence on every card: family and creator, parameter count read
  straight from safetensors headers, MoE layout, quant, context, license,
  engine and version.
- Live load telemetry while a model boots: VRAM fill rate, disk read rate,
  staging RAM, loader CPU, and an ETA — watching 235B of weights mmap in is
  the new progress bar.

**The console**
- 13 WebGL shader themes with their own palettes, fonts, and fully themed
  login screens (operator labels, typed boot logs, in-universe small print).
  The wallpaper reacts to cluster load, heat, and events. [Gallery.](docs/THEMES.md)
- Event ticker with hysteresis thresholds: hot GPU, VRAM pressure, power
  ceiling, queue depth, PCIe degradation, AER deltas, process spawn/exit,
  model online/offline, TPS records.
- Density mode: compact tiles engage automatically at 6+ GPUs. Eight
  full-size cards is a scrolling documentary; density mode is the trailer.
- Installable PWA, wake-lock in fullscreen, touch and foldable friendly —
  candybar, unfolded, portrait, or wall-mounted TV.

**The plumbing**
- Webhook alerts (Discord / Slack / JSON) from server-side detectors — your
  rig complains even when nobody has the tab open.
- Prometheus `/metrics` for the Grafana crowd.
- Mock rig (`BURNDECK_MOCK_GPUS=8`) and a browser-only demo feed — preview
  the console before the hardware clears customs.

## Quickstart

Python 3.11+, an NVIDIA driver, and a browser:

```sh
git clone https://github.com/jcartu/burndeck && cd burndeck
./scripts/install.sh          # venv + deps + config.toml, prompts for a password
.venv/bin/python app.py       # http://127.0.0.1:8801
```

No GPUs yet? Boot the synthetic rig — same server, same UI, imaginary
silicon that never thermal-throttles:

```sh
BURNDECK_MOCK_GPUS=8 .venv/bin/python app.py
```

No password configured? One is generated on first boot, printed once to the
log, and kept in `.auth-generated` (0600).

## Using the deck

| Key / control | Does |
|---|---|
| `W` (or right-click the theme button) | shift to the next theme |
| `S` (or click the theme button) | open the theme menu |
| `1`–`9`, `0` | jump straight to a theme |
| `D` (or ▦) | toggle density mode |
| `F` (or ⛶) | fullscreen + screen wake-lock |
| `Esc` | close the theme menu |

Reading the dials:

- **Speedometer** — GPU utilization. The thin outer tick is the session peak;
  the arc turns amber past 85% and red past 95%.
- **Thermometer** — core temperature, banded cool → warm → hot → fire. The
  trend arrow appears when it moved more than 1.5 °C in 30 s.
- **Bars** — memory, power (draw vs. limit), fan, and this GPU's share of
  cluster TPS.
- **PCIe line** — expected link vs. live link. If a card negotiates down
  under load you get a red badge, a ticker event, and (if configured) a
  webhook. Reseat your risers.
- **Model cards** — configured endpoints and discovered ones. Blue-flagged
  cards were sniffed from VRAM without any configuration; a loading card
  shows stage, fill rate, and ETA in real time.
- **Ticker** (bottom right) — severity-coded events, ten-second lifespan, no
  scrollback, no guilt. Anything worth keeping should go to your webhook.

The login screen is themed too — every theme brings its own operator labels
and boot log, so logging in feels like clocking into a different facility
each time.

## Deploy

| Path | Command |
|---|---|
| systemd | `./scripts/install.sh --systemd` — renders `deploy/burndeck.service`, enables it, health-checks it |
| Docker | `docker compose -f deploy/compose.yaml up -d --build` — needs nvidia-container-toolkit; `pid: host` lets the sniffer see host processes |
| Reverse proxy | `deploy/nginx.conf` — keep `proxy_buffering off` on `/api/stream`, or enjoy a dashboard frozen in time |

The server binds `127.0.0.1` and speaks plain HTTP; terminate TLS in front
(nginx, caddy, cloudflared). Step-by-step recipes for humans and coding
agents live in [AGENTS.md](AGENTS.md) — point your agent at this repo and it
has everything it needs.

## Configure

Copy `config.example.toml` to `config.toml` — or configure nothing and let
discovery do the work. Every key has a `BURNDECK_*` environment override
(env > file > defaults). Highlights:

```toml
[[endpoints]]                       # optional: pin labels/accents/attribution
url = "http://127.0.0.1:8000"
label = "Qwen3-235B · vLLM"
gpus = [0, 1, 2, 3, 4, 5, 6, 7]

[alerts]
webhook = "https://discord.com/api/webhooks/..."

[demo]
mock_gpus = 0                       # 8 = synthetic TP8 rig
```

Full reference with every knob explained: [docs/CONFIG.md](docs/CONFIG.md).

## Scaling to TP8 and beyond

Everything is per-index with no fixed fan-out: the backend polls N devices
concurrently off the event loop, the stream sends keyed deltas (an idle GPU
costs almost nothing on the wire), history appends are single points after
the initial window, and the UI skips redrawing cards you've scrolled past.
At 6+ GPUs the grid drops into density mode on its own. CI boots the mock
rig at 8 GPUs and asserts the full contract on every push; the Pages demo
runs it in your browser at any `?gpus=1..32`.

## Verification

```sh
BURNDECK_MOCK_GPUS=8 .venv/bin/python tools/smoke.py
```

31 end-to-end checks: boot, auth redirect and session cookie, snapshot
contract, SSE full-snapshot + delta frames, Prometheus gauges, and an actual
webhook delivery into a local sink. If it prints `smoke: PASS`, the whole
pipeline works on your machine. How it all fits together:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Security model

Read-only telemetry, no control plane — burndeck cannot touch clocks, power
limits, or processes. It watches. That's all it does. Session auth is an
HMAC-signed cookie (secret generated at `.session-secret`, 0600);
credentials live in your config or are generated on first boot. `/healthz`
is unauthenticated; `/metrics` is public by default (set `[metrics] public =
false` to gate it). Nothing phones home.

## Themes

`oblivion · ares · k2049 · muthur · jarvis · nightcity · arrakis · lumon ·
apex · construct · gargantua · thegrid · tokyo`

Palettes and flavor text in the [gallery](docs/THEMES.md), which also
documents the shader contract (`u_drive`, `u_heat`, `u_energy`, …) if you
want to write your own. A theme is a single JS file; PRs welcome, luma
budgets enforced — this is a dashboard, not a rave.

## Roadmap

- **Fleet view** — one pane aggregating several rigs, so the group can
  compare burn rates without screenshots.
- **Persistent history** — optional SQLite ring for 24 h charts.
- NVLink / C2C counters where the silicon has them.

## Related

[Burndeck Cloud](https://github.com/jcartu/burndeck-cloud) — the sibling
instrument deck for API quota, burn rate, and cooldowns on the hosted-model
side. This repo is the deck for the silicon you own.

## Credits

Interface inspired by a decade of great film UIs and by
[nvtop](https://github.com/Syllo/nvtop), the terminal original. burndeck is
the deck you watch while the Blackwells burn. MIT licensed.

Built with love for the **RTX6KPro Discord Family**.
