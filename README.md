# Tech Companion

Tech Companion is an Android guidance assistant for people with low digital literacy. Its web interface guides users through supported phone tasks, including making text larger.

## How it works

Pioneer classifies requests into intents; the orchestrator checks risk before guidance begins. Holo uses Android screenshots to suggest the next on-screen step. Explicit public Android-information lookups can use Tavily search when configured. In normal mode, ADB connects to an Android device for screenshots and actions; demo mode replaces the device, Pioneer routing, and Holo guidance with local simulations (no provider keys or Android device required). Tavily is not used in demo mode.

## Setup and run

Requires Python 3.12+ and `uv`. From this checkout:

```sh
uv sync --dev
cp .env.example .env
TECH_COMPANION_DEMO_MODE=1 uv run uvicorn app.main:app
```

Open `http://127.0.0.1:8000/` to use the demo. For a real Android device, configure the provider keys in `.env` and use ADB; see `.env.example` for the available settings. Do not set demo mode for that path.

## Tests

```sh
uv run pytest -q
```

## Known issue

`pyproject.toml` declares a `droid` console script (`app.android.cli:main`), but that module is not in this tree yet, so `uv run droid` will fail until the CLI lands.
