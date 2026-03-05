"""FastAPI application for the ESA beamforming backend."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Make array_engine importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.routes import router

app = FastAPI(
    title="ESA Beamforming API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
