"""PGN file ingestion.

Lazily streams chess.pgn.Game objects from a PGN file without loading the
entire file into memory. Call sites are responsible for error handling around
individual game reads.
"""

from __future__ import annotations

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
    with _open_text_pgn(pgn_path) as f:
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


def _open_text_pgn(pgn_path: Path):
    """Open plain or lightly-compressed PGN as text.

    Lichess monthly archives are commonly distributed as .zst. We do not add a
    zstandard dependency here because this pipeline is intentionally small; pass
    a decompressed .pgn, or a .gz/.bz2 file handled by the Python stdlib.
    """
    suffixes = pgn_path.suffixes
    if suffixes[-2:] == [".pgn", ".gz"] or pgn_path.suffix == ".gz":
        import gzip

        return gzip.open(pgn_path, mode="rt", encoding="utf-8", errors="replace")
    if suffixes[-2:] == [".pgn", ".bz2"] or pgn_path.suffix == ".bz2":
        import bz2

        return bz2.open(pgn_path, mode="rt", encoding="utf-8", errors="replace")
    if suffixes[-2:] == [".pgn", ".zst"] or pgn_path.suffix == ".zst":
        raise ValueError(
            "Reading .zst PGN files requires an extra decompression step. "
            "Decompress the archive to .pgn first, or recompress as .gz/.bz2."
        )
    return open(pgn_path, encoding="utf-8", errors="replace")
