"""PGN file ingestion.

Lazily streams chess.pgn.Game objects from a PGN file without loading the
entire file into memory. Call sites are responsible for error handling around
individual game reads.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

try:
    import chess.pgn
except ImportError as exc:
    raise ImportError(
        "python-chess is required. Install training extras: uv sync --extra training"
    ) from exc


def iter_pgn_games(pgn_path: Path) -> Iterator[chess.pgn.Game]:
    """Yield chess.pgn.Game objects one at a time from pgn_path.

    Silently skips games that python-chess cannot parse at all (returns None).
    The caller decides what to do with games that parse but fail domain filters.
    """
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                # Skip games that cause exceptions during read (e.g. malformed tokens).
                continue
            if game is None:
                break
            yield game


def pgn_sha256(pgn_path: Path) -> str:
    """Return the SHA-256 hex digest of a PGN file."""
    import hashlib

    h = hashlib.sha256()
    with open(pgn_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
