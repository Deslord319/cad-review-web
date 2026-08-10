# CAD Review Web

A small, LAN-first Web interface for reviewing and organizing CAD exports. It renders STL files in the browser with Three.js and exposes model dimensions, facet count, volume, and a basic watertight check.

The bundled Python API indexes a configurable model directory and supports three recoverable states: active, archive, and trash. It does not permanently delete files.

## Features

- Interactive STL viewing with orbit, standard views, wireframe, and auto-rotation
- STL audit data: dimensions, facets, volume, and watertight status
- Download support for STL, STEP/STP, FCStd, and PNG files
- Archive, trash, and restore workflows implemented as local file moves
- No database and no third-party service required

## Requirements

- Node.js 20 or newer
- Python 3.9 or newer

## Run locally

Install the Web dependencies:

```bash
npm install
```

Start the model API. By default it reads `./models` and only listens on localhost:

```bash
python3 server/model_server.py
```

In another terminal, start the Web interface:

```bash
npm run dev
```

Open `http://localhost:5173`. Put model exports in `./models`, then refresh the page.

## LAN configuration

To review models from another computer on a trusted LAN, bind both services to the LAN interface:

```bash
CAD_OUTPUT_DIR=/path/to/cad/exports \
CAD_VIEWER_API_HOST=0.0.0.0 \
CAD_VIEWER_API_PORT=8091 \
CAD_VIEWER_ALLOWED_ORIGIN=http://cad-host.local:5173 \
python3 server/model_server.py

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
| `VITE_API_BASE_URL` | Same Web hostname, port `8091` | Browser-visible API URL |

## Verify

```bash
npm run check
```

## Security

This project is designed for localhost or a trusted LAN. The file-management API has no authentication and allows archive/trash/restore operations inside `CAD_OUTPUT_DIR`. Do not expose the API directly to the public internet. Use a firewall, VPN, or an authenticated reverse proxy when access crosses a trusted network boundary.

No CAD model files, screenshots, machine addresses, credentials, or printing data are included in this repository.

## License

Released under the [MIT License](LICENSE).
