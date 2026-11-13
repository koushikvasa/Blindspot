# Blindspot UI: how to run it

No build step. FastAPI serves the API and the pages.

```powershell
# from the repo root, with .venv active
pip install -r requirements.txt
uvicorn blindspot.api.main:app --host 0.0.0.0 --port 8000
```

- Story (presenter laptop): http://localhost:8000/
- Phone page: http://localhost:8000/join
- Stage (projector): http://localhost:8000/stage (optional `?map=map_gss_agent_r4` to pick the background map)

## Keys on the story page
- Up / Down / Page Up / Page Down / Space: move between scenes
- Left / Right: switch crowds (scene 2) or agent rounds (scene 4)
- R: back to the start

## Data
The pages read `results/ui/*.json` (real results, written by `python -m blindspot.eval.export_ui`, see `handovers/H8-export-ui-data.md`). Any missing file falls back to `web/mock/` and a yellow "Sample data in use" badge appears. Rebuild sample data with `python scripts/make_mock_ui.py`. File formats: `docs/UI_DATA_CONTRACT.md`.

## Live mode on the day
Phones must reach the laptop. Two options:
1. Same Wi-Fi or your phone hotspot: find the laptop's IP (`ipconfig`) and open the story page as `http://<laptop-ip>:8000/?join=http://<laptop-ip>:8000/join` so the QR code points to the right address. Allow Python through Windows Firewall when asked.
2. A tunnel (e.g. `ngrok http 8000`), then `http://localhost:8000/?join=https://<your-tunnel>/join`.

Answers are scored in memory and never written to disk. `/api/live/reset` clears the stage feed.

## Files
- `blindspot/api/main.py`, `blindspot/api/live.py`: API and live scoring
- `web/index.html`, `web/js/story.js`: the six-scene story
- `web/join.html`, `web/js/join.js`: phone flow
- `web/stage.html`: projector view
- `web/js/map.js`: the constellation renderer
- `web/css/app.css`: all styles
- `web/vendor/qrcode.js`: QR library (qrcode-generator 1.4.4, MIT)
- `tests/test_api.py`: API tests
