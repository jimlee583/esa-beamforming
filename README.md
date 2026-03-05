# ESA Beamforming Explorer

Interactive tool for designing and visualising electronically steered array (ESA) beamforming patterns. Includes a NumPy-based array engine, a FastAPI backend, and a React/Vite frontend.

---

## Conventions

All math conventions are documented in detail at the top of [`array_engine/steering.py`](array_engine/steering.py). A summary:

### Coordinate frame (array / body frame)

| Axis | Meaning |
|------|---------|
| **x** | Horizontal across the panel |
| **y** | Vertical across the panel |
| **z** | Boresight (outward normal) |

ENU and ECEF frames are defined in the models but not yet implemented.

### Direction representation

All directions are internally represented as **unit vectors** `u = [ux, uy, uz]`.

### Azimuth / elevation mapping

```
ux = sin(az) · cos(el)
uy = sin(el)
uz = cos(az) · cos(el)
```

| Parameter | Zero | Positive direction |
|-----------|------|--------------------|
| **az** (azimuth) | Boresight (+z) | Toward +x |
| **el** (elevation) | x–z plane | Toward +y |

At broadside: `az = 0, el = 0 → u = [0, 0, 1]`

### Steering vector phase convention

```
φ_n = −k · (r_n · u)       where k = 2π / λ
w_n = a_n · exp(j · φ_n)
```

- `r_n` — element position vector
- `u` — desired look-direction unit vector
- `a_n` — amplitude taper (1.0 for uniform)
- **Broadside sanity check:** `u = [0,0,1]`, all `r_n` have `z = 0` (planar) → `r_n · u = 0` → all phases = 0. ✓

---

## Repository structure

```
esa-beamforming/
├── array_engine/              # Pure-Python + NumPy beamforming library
│   ├── __init__.py
│   ├── geometry.py            # Rectangular & triangular lattice generators
│   ├── steering.py            # az/el → unit vector, steering phases & weights
│   ├── pattern.py             # Array factor & pattern cuts
│   └── models.py              # Pydantic models (shared with backend)
├── backend/                   # FastAPI service
│   ├── pyproject.toml         # uv deps + ruff/mypy/pytest config
│   └── app/
│       ├── main.py
│       └── routes.py
├── frontend/                  # React + Vite + Plotly UI
│   ├── package.json
│   ├── eslint.config.js
│   └── src/
│       ├── App.tsx            # Main component
│       ├── api.ts             # API client
│       └── types.ts           # TS types mirroring backend models
├── tests/                     # pytest test suite (32 tests, <1s)
│   ├── test_steering.py       # Convention & sign tests
│   ├── test_pattern.py        # Pattern peak tests
│   ├── test_golden.py         # Hand-calculated golden-reference tests
│   ├── test_geometry.py       # Lattice generator tests
│   └── test_api.py            # API integration tests
├── .github/workflows/ci.yml   # GitHub Actions (lint + type-check + test + build)
├── .pre-commit-config.yaml    # Pre-commit hooks
├── .editorconfig
├── Makefile
└── README.md
```

---

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Node.js >= 18
- (Optional) [pre-commit](https://pre-commit.com/) for git hooks

---

## Quick start

### Install everything

```bash
make install
```

This runs `uv sync` + dev tool installs in `backend/`, and `npm install` in `frontend/`.

### Run the backend

```bash
make backend
# or: cd backend && uv run uvicorn app.main:app --reload --port 8000
```

### Run the frontend

```bash
make frontend
# or: cd frontend && npm run dev
```

The Vite dev server proxies `/api/*` to `http://localhost:8000`. Open http://localhost:5173.

---

## Quality commands

| Command | What it does |
|---------|--------------|
| `make test` | Run pytest (32 tests, <1s) |
| `make lint` | Ruff lint (Python) + ESLint (frontend) |
| `make format` | Auto-format Python with ruff |
| `make format-check` | Check formatting without changing files |
| `make typecheck` | Run mypy on `array_engine` + `app` |
| `make ci` | Run all checks (lint + format-check + typecheck + test + frontend build) |

### Running checks individually

```bash
# Python lint
cd backend && uv run ruff check ../array_engine ../tests app

# Python format check
cd backend && uv run ruff format --check ../array_engine ../tests app

# Python type check
cd backend && uv run mypy

# Python tests
cd backend && uv run pytest -v

# Frontend lint
cd frontend && npm run lint

# Frontend build
cd frontend && npm run build
```

---

## Pre-commit hooks

Install pre-commit and enable the hooks:

```bash
# Install pre-commit (pick one)
uv tool install pre-commit
# or: pipx install pre-commit
# or: brew install pre-commit

# Enable hooks in this repo
pre-commit install

# Run all hooks once (to verify)
pre-commit run --all-files
```

Hooks configured:
- **trailing-whitespace**, **end-of-file-fixer**, **check-yaml** — basic hygiene
- **ruff** — lint with auto-fix
- **ruff-format** — code formatting
- **mypy** — type checking
- **pytest** — runs on `pre-push` (not on every commit, to stay fast)

---

## CI (GitHub Actions)

The workflow at `.github/workflows/ci.yml` runs on every push to `main` and on pull requests. It has two jobs:

**Backend** — `ubuntu-latest`, Python 3.12:
1. `uv sync` + install dev tools
2. `ruff check` (lint)
3. `ruff format --check` (formatting)
4. `mypy` (type checking)
5. `pytest` (32 tests)

**Frontend** — `ubuntu-latest`, Node 22:
1. `npm ci`
2. `npm run lint` (ESLint)
3. `npm run build` (TypeScript + Vite)

---

## API reference

### `GET /api/health`

Returns `{"status": "ok"}`.

### `POST /api/weights`

Compute steering weights for a planar array.

```bash
curl -s http://localhost:8000/api/weights \
  -H 'Content-Type: application/json' \
  -d '{
    "freq_hz": 10e9,
    "panel_size_m": 0.3,
    "lattice": "rectangular",
    "element_k_lambda": 0.5,
    "steer_az_deg": 15,
    "steer_el_deg": 0,
    "taper": "uniform"
  }' | python3 -m json.tool | head -20
```

**Response fields:** `n_elements`, `positions` (m), `phases_rad`, `weights_re_im` ([[re,im],...]), `spacing_m`, `spacing_lambda`.

### `POST /api/pattern`

Compute array factor pattern cuts.

```bash
curl -s http://localhost:8000/api/pattern \
  -H 'Content-Type: application/json' \
  -d '{
    "freq_hz": 10e9,
    "panel_size_m": 0.3,
    "lattice": "rectangular",
    "element_k_lambda": 0.5,
    "steer_az_deg": 15,
    "steer_el_deg": 0,
    "taper": "uniform",
    "az_range_deg": 90,
    "el_range_deg": 90,
    "n_points": 361
  }' | python3 -m json.tool | head -20
```

**Response fields:** `az_cut`, `el_cut` (each with `angles_deg`, `gain_db`, `label`), `peak_az_deg`, `peak_el_deg`, `peak_gain_db`, `n_elements`, `spacing_lambda`.
