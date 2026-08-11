# burndeck — agent deployment notes

Instructions for coding agents (and terse humans) deploying or modifying this
repo. Everything here is deterministic and verifiable.

## Repo map

```
app.py                  # entire backend: FastAPI, NVML poller, endpoint scraper,
                        # weight sniffer, mock rig, alerts, /metrics — one file
static/index.html       # entire dashboard UI (CSS + HTML + JS, one file)
static/login.html       # themed login page
static/demo.js          # synthetic rig for backend-less demo (Pages, ?gpus=N)
static/themes/*.js      # 13 WebGL themes + runner.js (shared shader runner)
config.example.toml     # every setting, commented; copy to config.toml
scripts/install.sh      # idempotent installer (venv, config, optional systemd)
scripts/hash_password.py# getpass/env -> password_sha256 line for config.toml
deploy/                 # burndeck.service, nginx.conf, Dockerfile, compose.yaml
tools/smoke.py          # 31-check end-to-end proof, no GPU needed
tools/validate_theme.py # per-theme shader validation (optional, Playwright)
docs/                   # CONFIG.md, ARCHITECTURE.md, THEMES.md
```

## Deploy — bare metal + systemd (recommended)

```sh
./scripts/install.sh --non-interactive --systemd
curl -fsS http://127.0.0.1:8801/healthz   # -> {"ok":true,...,"gpus":<N>}
```

- Requires: python3 >= 3.11, sudo for the unit install, NVIDIA driver for
  real telemetry (warns and continues without; add `--mock 8` for synthetic).
- `--non-interactive` skips the password prompt; the server generates
  credentials on first boot and prints them once. Retrieve them later from
  `.auth-generated` (0600, `user:password`).
- To set credentials explicitly instead:
  `BURNDECK_PASSWORD='...' python scripts/hash_password.py` and paste the
  printed line into `config.toml` under `[auth]`.

## Deploy — Docker

```sh
docker compose -f deploy/compose.yaml up -d --build
```

Requires nvidia-container-toolkit. The compose file uses `gpus: all` and
`pid: host` (the model sniffer reads host /proc; without it you lose model
discovery, not GPU telemetry). Mock variant needs neither — set
`BURNDECK_MOCK_GPUS=8` on the service and drop both keys.

## Verify (always do this)

```sh
BURNDECK_MOCK_GPUS=8 .venv/bin/python tools/smoke.py   # must end: smoke: PASS — 31 checks
curl -fsS http://127.0.0.1:8801/healthz                 # {"ok":true,"version":...,"mock":...,"gpus":N}
curl -fsS http://127.0.0.1:8801/metrics | head          # burndeck_* gauges
```

`GET /` unauthenticated returns 302 → `/login`. `GET /api/snapshot` without a
session cookie returns 401. That is correct behavior, not a failure.

## Configuration contract

- Precedence: `BURNDECK_*` env > `config.toml` > defaults. File is optional.
- Key env vars: `BURNDECK_HOST`, `BURNDECK_PORT`, `BURNDECK_CONFIG`,
  `BURNDECK_DATA_DIR`, `BURNDECK_MOCK_GPUS`, `BURNDECK_AUTH` (on/off),
  `BURNDECK_AUTH_USER`, `BURNDECK_AUTH_PASSWORD_SHA256`,
  `BURNDECK_MODEL_ROOTS` (colon-separated), `BURNDECK_WEBHOOK`,
  `BURNDECK_GPU_POLL_SECONDS`, `BURNDECK_LLM_POLL_SECONDS`.
- Full schema: docs/CONFIG.md. TOML tables: `[server] [auth] [polling]
  [models] [[endpoints]] [metrics] [alerts] [demo]`.

## Operational invariants

- The server binds 127.0.0.1 by default. Expose via reverse proxy only;
  `deploy/nginx.conf` is canonical. On `/api/stream` (SSE) buffering MUST
  stay off and read timeouts long — a buffering proxy freezes the dashboard.
- Runtime state lives next to app.py (or `BURNDECK_DATA_DIR`):
  `.session-secret`, `.auth-generated`. Both 0600, both gitignored. Never
  commit them; never regenerate on a live install unless rotating creds.
- `config.toml` is gitignored by design — only `config.example.toml` ships.
- Static assets are cache-busted by the `?v=` token (`ASSET_VERSION` in
  app.py, `ASSET_VER` in both HTML files). Bump all three together when
  shipping UI changes.
- The frontend uses only RELATIVE paths (`static/...`, `api/...`) so it works
  at `/`, behind sub-path proxies, and on GitHub Pages. Keep it that way.
- "Dashboard is slow" reports: the client has an adaptive lite tier
  (`?lite=1` to force, `?lite=0` to forbid, `L` key, localStorage key
  `burndeck-perf`). Rendering is already GPU-backed (Metal/D3D/GL via the
  browser); lite disables backdrop blur and shrinks the wallpaper budget.

## CI / Pages

- `.github/workflows/ci.yml` — runs `tools/smoke.py` at mock-8 and
  `node --check` over every theme. Must pass before merging.
- `.github/workflows/pages.yml` — publishes `static/` to GitHub Pages. The UI
  detects `*.github.io` (or `?demo` / `?gpus=N`) and feeds itself from
  `static/demo.js` instead of the API. No server involved.

## Editing guidance

- Backend is deliberately one file with `# ----------` section markers; add
  to the matching section rather than creating modules.
- New themes: follow docs/THEMES.md (registration shape, shader contract,
  luma budget), add the id to `THEME_ORDER` in BOTH static/index.html and
  static/login.html, validate with `tools/validate_theme.py <id>`.
- Any change to snapshot/stream shape must update all four consumers of the
  contract: `build_stream_update` (app.py), `mergeStreamFrame`
  (static/index.html), `static/demo.js`, and `tools/smoke.py`.
