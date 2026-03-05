"""Shared fixtures for beamforming tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure array_engine is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
