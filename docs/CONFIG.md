# Configuration reference

Precedence for every setting: **`BURNDECK_*` environment > `config.toml` >
built-in default**. The file is optional — burndeck with zero configuration
still finds your GPUs, your model servers, and your models. Configuration is
for pinning names and colors, opening alerts, and saying who gets to look.

The server looks for `./config.toml` next to `app.py` unless
`BURNDECK_CONFIG` points elsewhere. `config.example.toml` mirrors this page
with comments.

Booleans from the environment accept `1/0`, `true/false`, `yes/no`, `on/off`.

## [server]

| Key | Env | Default | Notes |
|---|---|---|---|
| `host` | `BURNDECK_HOST` | `127.0.0.1` | Bind address. Keep loopback; expose via reverse proxy. |
| `port` | `BURNDECK_PORT` | `8801` | |

## [auth]

| Key | Env | Default | Notes |
|---|---|---|---|
| `enabled` | `BURNDECK_AUTH` | `true` | `off` only makes sense on a trusted LAN. |
| `user` | `BURNDECK_AUTH_USER` | `operator` | |
| `password_sha256` | `BURNDECK_AUTH_PASSWORD_SHA256` | — | Produce with `scripts/hash_password.py`. |
| `password` | `BURNDECK_AUTH_PASSWORD` | — | Plain-text alternative; `password_sha256` wins if both set. |

With auth enabled and no password configured, first boot generates a random
password, prints it once to the log, and persists `user:password` to
`.auth-generated` (0600) so restarts keep the same secret. Delete that file
to force regeneration; set real credentials in config to retire it.

Sessions are HMAC-signed cookies (`burndeck_session`, 30 days), keyed by an
auto-generated `.session-secret`. Deleting that file invalidates all
sessions.

## [polling]

| Key | Env | Default | Notes |
|---|---|---|---|
| `gpu_seconds` | `BURNDECK_GPU_POLL_SECONDS` | `0.25` | Live-tile cadence. NVML keeps 4 Hz cheap even at 8 GPUs. |
| `llm_seconds` | `BURNDECK_LLM_POLL_SECONDS` | `1.0` | Endpoint probe cadence; never blocks the GPU feed. |
| `history_seconds` | `BURNDECK_HISTORY_INTERVAL` | `1.0` | Chart sampling cadence, independent of tile refresh. |
| `history_points` | `BURNDECK_HISTORY_POINTS` | `180` | Chart window (180 × 1 s = 3 min). |

## [models]

| Key | Env | Default | Notes |
|---|---|---|---|
| `roots` | `BURNDECK_MODEL_ROOTS` (colon-separated) | `["/models"]` | Scanned (2 levels) for HF-style dirs with `config.json`; feeds the model-intel cards. |

## [[endpoints]]

Optional array of tables — model servers to probe for TPS, KV cache, and
queue depth. Zero entries is normal: any local port speaking the OpenAI API
is discovered and scraped automatically. Declare an endpoint only to pin its
label, accent colors, or GPU attribution.

| Key | Default | Notes |
|---|---|---|
| `url` | — (required) | e.g. `http://127.0.0.1:8000` |
| `id` | derived | Stable id used in the stream and metrics labels. |
| `label` | host:port | Card title. |
| `gpus` | `[]` | GPU indices; empty = attribute by process/port ownership. |
| `accent`, `accent2` | palette cycle | Card accent colors. |

Works with anything exposing `/v1/models`; TPS/KV/queue additionally need
vLLM- or SGLang-style `/metrics`.

## [metrics]

| Key | Env | Default | Notes |
|---|---|---|---|
| `enabled` | `BURNDECK_METRICS` | `true` | `GET /metrics`, Prometheus text format, `burndeck_*` namespace. |
| `public` | `BURNDECK_METRICS_PUBLIC` | `true` | `false` requires the dashboard session cookie (browser scrapes only). |

Exported: per-GPU utilization, memory, temperature, power, fan, clocks, PCIe
gen/width, AER counters; per-model health, in/out TPS, KV %, running/waiting,
token totals; cluster totals.

## [alerts]

| Key | Env | Default | Notes |
|---|---|---|---|
| `webhook` | `BURNDECK_WEBHOOK` | `""` (off) | Discord webhook URL, Slack incoming webhook, or any JSON sink. |
| `format` | — | `discord` | `discord` \| `slack` \| `json`. |
| `events` | — | all | Subset of `gpu_hot gpu_vram power_ceiling model_health pcie aer_fatal queue`. |
| `gpu_hot_c` | — | `83.0` | Fires at ≥ threshold, clears 9 °C below (hysteresis). |
| `vram_pct` | — | `96.0` | Clears 8 points below. |
| `power_pct` | — | `95.0` | Cluster power ceiling; clears 13 points below. |
| `cooldown_seconds` | — | `300.0` | Per-event floor between deliveries. |

Detection runs in the poll loop (pure arithmetic); delivery happens on a
single-worker pool so a slow webhook can never stall telemetry.

## [demo]

| Key | Env | Default | Notes |
|---|---|---|---|
| `mock_gpus` | `BURNDECK_MOCK_GPUS` | `0` | N > 0 boots a synthetic N-GPU rig: no NVML, no probes, full UI/stream/alerts/metrics. CI uses 8. |

## Other environment variables

| Env | Default | Notes |
|---|---|---|
| `BURNDECK_CONFIG` | `./config.toml` | Alternate config path. |
| `BURNDECK_DATA_DIR` | app dir | Where `.session-secret` / `.auth-generated` live (containers: a volume). |
