# CAD Review Web

A small, LAN-first Web interface for reviewing and organizing CAD exports. It renders STL and 3MF files in the browser with Three.js; STL files additionally expose model dimensions, facet count, volume, and a basic watertight check.

The bundled Python API indexes a configurable model directory and supports three recoverable states: active, archive, and trash. It does not permanently delete files.

## Features

- Interactive STL and 3MF viewing with orbit, standard views, wireframe, and auto-rotation
- Persistent SQLite/WAL preview queue with content-addressed GLB artifacts
- Independent, serial preview worker so model conversion never blocks the API
- Split/production-extension 3MF loading by package path and object ID, with shared geometry instances instead of repeated mesh expansion
- STL audit data: dimensions, facets, volume, and watertight status
- Download support for STL, 3MF, STEP/STP, FCStd, and PNG files
- Archive, trash, and restore workflows implemented as local file moves
- No third-party service required; preview state uses Python's bundled SQLite

## Requirements

- Node.js 20 or newer
- Python 3.10 or newer

## Run locally

Install the Web dependencies:

```bash
npm install
```

Create the API environment and install its preview dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r server/requirements.txt
```

Start the model API. By default it reads `./models` and only listens on localhost:

```bash
.venv/bin/python server/model_server.py
```

In another terminal, start the preview worker. It reconciles existing and
externally uploaded 3MF files, queues smaller files first, and runs each
conversion in a short-lived subprocess:

```bash
.venv/bin/python server/preview_worker.py
```

In a third terminal, start the Web interface:

```bash
npm run dev
```

Open `http://localhost:5173`. Put model exports in `./models`, then refresh the page.

Archiving a 3MF through the API enqueues it immediately. Files copied directly
into the active or archive directory are discovered by the worker's periodic
reconciliation. The browser API only serves a GLB after its state is `ready`;
it returns HTTP 202 for `pending`/`processing` and HTTP 422 for `failed`, and
never converts a model in a request thread.

The browser accepts deep links from an MCP uploader. For example,
`/?scope=archive&file=finished-part.stl` opens the archive tab and selects that
file when it is present.

## LAN configuration

To review models from another computer on a trusted LAN, bind both services to the LAN interface:

```bash
CAD_OUTPUT_DIR=/path/to/cad/exports \
CAD_VIEWER_API_HOST=0.0.0.0 \
CAD_VIEWER_API_PORT=8091 \
CAD_VIEWER_ALLOWED_ORIGIN=http://cad-host.local:5173 \
.venv/bin/python server/model_server.py

CAD_OUTPUT_DIR=/path/to/cad/exports \
.venv/bin/python server/preview_worker.py

npm run dev
```

The Web interface uses port `8091` on the same hostname by default. To point it somewhere else, copy `.env.example` to `.env` and set `VITE_API_BASE_URL`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAD_OUTPUT_DIR` | `./models` | Directory containing model exports |
| `CAD_VIEWER_API_HOST` | `127.0.0.1` | API bind address |
| `CAD_VIEWER_API_PORT` | `8091` | API port |
| `CAD_VIEWER_ALLOWED_ORIGIN` | `http://localhost:5173` | Web origin allowed to call the API |
| `CAD_VIEWER_PREVIEW_DIR` | `<CAD_OUTPUT_DIR>/.preview-cache` | SQLite/WAL queue and content-addressed GLB directory |
| `CAD_VIEWER_PREVIEW_CACHE_DIR` | `<CAD_OUTPUT_DIR>/.preview-cache` | Backward-compatible alias for `CAD_VIEWER_PREVIEW_DIR` |
| `CAD_VIEWER_PREVIEW_PIPELINE_VERSION` | `three-mf-glb-v3` | Converter version included in each cache key |
| `CAD_VIEWER_PREVIEW_PROFILE` | `fast` | Preview conversion profile included in each cache key |
| `CAD_VIEWER_PREVIEW_FACE_BUDGET` | `100000` | Fast-preview triangle budget |
| `CAD_VIEWER_PREVIEW_TIMEOUT` | `300` | Per-model converter timeout in seconds |
| `CAD_VIEWER_PREVIEW_MAX_ATTEMPTS` | `3` | Automatic attempts before a job becomes `failed` |
| `CAD_VIEWER_PREVIEW_SCAN_INTERVAL` | `30` | External-file reconciliation interval in seconds |
| `CAD_VIEWER_PREVIEW_NUMERIC_THREADS` | `1` | BLAS/OpenMP threads allowed in a converter subprocess |
| `VITE_API_BASE_URL` | Same Web hostname, port `8091` | Browser-visible API URL |

## Preview API

- `POST /api/previews/enqueue` with `{"scope":"archive","name":"part.3mf"}`
  indexes a source and returns `preview_status`, `preview_revision`, and a
  relative `preview_url`. An optional `sha256` verifies the source content.
- `GET /api/previews/status?scope=archive&name=part.3mf` reads durable state.
- `POST /api/previews/retry` with the same scope/name body resets a failed job.
- `GET /api/models?scope=archive` returns per-source preview fields and
  `preview_counts`. Generated artifacts and SQLite files never affect the
  active/archive/trash model counts.

## Verify

```bash
npm run check
```

## Security

This project is designed for localhost or a trusted LAN. The file-management API has no authentication and allows archive/trash/restore operations inside `CAD_OUTPUT_DIR`. Do not expose the API directly to the public internet. Use a firewall, VPN, or an authenticated reverse proxy when access crosses a trusted network boundary.

No CAD model files, screenshots, machine addresses, credentials, or printing data are included in this repository.

## License

Released under the [MIT License](LICENSE).
