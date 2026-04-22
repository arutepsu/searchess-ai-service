"""Position feature encoding — the single source of truth for board representation.

This module encodes chess positions (FEN strings) into fixed-size float32
feature vectors used by both training and inference.

Encoder stability contract
--------------------------
_REFERENCE_NONZERO_INDICES lists every feature index that must be set (=1.0)
when encoding the standard starting position. Any change to encode_fen() that
shifts or removes a feature will cause tests and the runtime fingerprint check
to fail immediately.

compute_encoder_fingerprint() produces a short deterministic string that is
embedded in every artifact's encoder_config. At load time, the runtime
recomputes this fingerprint and compares — if encode_fen() changed but
ENCODER_VERSION was not bumped, the fingerprint check still catches the drift.

Move encoding lives in move_encoder.py, not here.

Updating the encoder
--------------------
If encode_fen() is intentionally changed:
  1. Verify the change is correct.
  2. Update _REFERENCE_NONZERO_INDICES to match the new reference output.
  3. Bump ENCODER_VERSION (e.g. "1.0" → "2.0").
  4. Re-run the full dataset → train pipeline to produce a new artifact.
  All existing artifacts will be rejected at load time until re-trained.
"""

from __future__ import annotations

import hashlib

import numpy as np

try:
    import chess
except ImportError as exc:
    raise ImportError(
        "python-chess is required. Install training extras: uv sync --extra training"
    ) from exc

# ---------------------------------------------------------------------------
# Encoder constants
# ---------------------------------------------------------------------------

# Feature vector layout:
#   [0:768]   12 piece-type planes × 64 squares (white/black P N B R Q K)
#   [768:772] castling rights (WK, WQ, BK, BQ)
#   [772]     side to move (1.0 = white, 0.0 = black)
FEATURE_SIZE: int = 773

# Bump this when encode_fen() changes in a way that alters the feature vector.
# Must also update _REFERENCE_NONZERO_INDICES when bumping.
ENCODER_VERSION: str = "1.0"

_PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]
_COLORS = [chess.WHITE, chess.BLACK]

# ---------------------------------------------------------------------------
# Encoder stability reference data
# ---------------------------------------------------------------------------

# Standard starting position — stable across python-chess versions.
REFERENCE_FEN: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Expected non-zero feature indices when encoding REFERENCE_FEN.
#
# Plane layout: plane = piece_type_idx * 2 + color_idx
#   Plane  0 – white pawns   (pt=PAWN,   color=WHITE)
#   Plane  1 – black pawns   (pt=PAWN,   color=BLACK)
#   Plane  2 – white knights (pt=KNIGHT, color=WHITE)
#   Plane  3 – black knights (pt=KNIGHT, color=BLACK)
#   Plane  4 – white bishops (pt=BISHOP, color=WHITE)
#   Plane  5 – black bishops (pt=BISHOP, color=BLACK)
#   Plane  6 – white rooks   (pt=ROOK,   color=WHITE)
#   Plane  7 – black rooks   (pt=ROOK,   color=BLACK)
#   Plane  8 – white queen   (pt=QUEEN,  color=WHITE)
#   Plane  9 – black queen   (pt=QUEEN,  color=BLACK)
#   Plane 10 – white king    (pt=KING,   color=WHITE)
#   Plane 11 – black king    (pt=KING,   color=BLACK)
#
# Square numbering: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
#
# To regenerate: python -m searchess_ai.model_runtime.encoder
_REFERENCE_NONZERO_INDICES: frozenset[int] = frozenset(
    {
        # Plane 0: white pawns at a2–h2 (squares 8–15)
        8, 9, 10, 11, 12, 13, 14, 15,
        # Plane 1: black pawns at a7–h7 (squares 48–55)
        112, 113, 114, 115, 116, 117, 118, 119,
        # Plane 2: white knights at b1=1, g1=6
        129, 134,
        # Plane 3: black knights at b8=57, g8=62
        249, 254,
        # Plane 4: white bishops at c1=2, f1=5
        258, 261,
        # Plane 5: black bishops at c8=58, f8=61
        378, 381,
        # Plane 6: white rooks at a1=0, h1=7
        384, 391,
        # Plane 7: black rooks at a8=56, h8=63
        504, 511,
        # Plane 8: white queen at d1=3
        515,
        # Plane 9: black queen at d8=59
        635,
        # Plane 10: white king at e1=4
        644,
        # Plane 11: black king at e8=60
        764,
        # Castling rights: WK=768, WQ=769, BK=770, BQ=771
        768, 769, 770, 771,
        # Side to move: 772 (1.0 = white)
        772,
    }
)

# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------


def encode_fen(fen: str) -> np.ndarray:
    """FEN string → float32 feature vector of shape (FEATURE_SIZE,).

    Raises ValueError for invalid FEN strings.
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {fen!r}") from exc

    features = np.zeros(FEATURE_SIZE, dtype=np.float32)

    # Piece planes: 6 piece types × 2 colors × 64 squares
    for pt_idx, piece_type in enumerate(_PIECE_TYPES):
        for col_idx, color in enumerate(_COLORS):
            plane = pt_idx * 2 + col_idx
            for sq in board.pieces(piece_type, color):
                features[plane * 64 + sq] = 1.0

    # Castling rights
    features[768] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[769] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[770] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[771] = float(board.has_queenside_castling_rights(chess.BLACK))

    # Side to move
    features[772] = 1.0 if board.turn == chess.WHITE else 0.0

    return features


# ---------------------------------------------------------------------------
# Encoder stability checks
# ---------------------------------------------------------------------------


def compute_encoder_fingerprint(fen: str = REFERENCE_FEN) -> str:
    """Compute a deterministic fingerprint of the encoder's output on fen.

    Uses sorted non-zero indices rather than raw float bytes so the result is
    platform-independent (no endianness or float-representation variation).

    This fingerprint is stored in every artifact's encoder_config and
    re-verified at load time to catch encoder drift even if ENCODER_VERSION
    was not bumped.
    """
    vec = encode_fen(fen)
    nonzero_indices = sorted(int(i) for i in np.nonzero(vec)[0])
    raw = ",".join(str(i) for i in nonzero_indices)
    return hashlib.sha256(raw.encode()).hexdigest()


def verify_encoder_stable() -> None:
    """Assert that encode_fen(REFERENCE_FEN) still matches _REFERENCE_NONZERO_INDICES.

    Raises AssertionError on drift with a human-readable diff showing which
    indices were added or removed, making root-cause analysis immediate.

    Call from tests and CI. Also called at encoder module validation time.
    """
    vec = encode_fen(REFERENCE_FEN)
    current = frozenset(int(i) for i in np.nonzero(vec)[0])
    if current == _REFERENCE_NONZERO_INDICES:
        return
    added = sorted(current - _REFERENCE_NONZERO_INDICES)
    removed = sorted(_REFERENCE_NONZERO_INDICES - current)
    raise AssertionError(
        f"Encoder drift detected on reference FEN '{REFERENCE_FEN}'.\n"
        f"  Unexpected non-zero indices (added): {added}\n"
        f"  Missing non-zero indices (removed):  {removed}\n"
        "If encode_fen() was intentionally changed:\n"
        "  1. Update _REFERENCE_NONZERO_INDICES to match the new output.\n"
        "  2. Bump ENCODER_VERSION.\n"
        "  3. Re-run: python -m searchess_ai.model_runtime.encoder  (to verify).\n"
        "  4. Re-train all artifacts — existing ones will be rejected at load time."
    )


def encoder_config() -> dict:
    """Return a serialisable snapshot of position encoding parameters.

    Includes the live fingerprint so artifacts carry behavioral identity,
    not just a version string.
    """
    return {
        "version": ENCODER_VERSION,
        "feature_size": FEATURE_SIZE,
        "fingerprint": compute_encoder_fingerprint(),
        "piece_plane_order": ["P", "p", "N", "n", "B", "b", "R", "r", "Q", "q", "K", "k"],
        "castling_bits": ["WK", "WQ", "BK", "BQ"],
        "side_to_move_index": 772,
    }


# ---------------------------------------------------------------------------
# Developer utility
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Position encoder self-check")
    print(f"  ENCODER_VERSION : {ENCODER_VERSION}")
    print(f"  FEATURE_SIZE    : {FEATURE_SIZE}")
    print(f"  REFERENCE_FEN   : {REFERENCE_FEN}")
    fp = compute_encoder_fingerprint()
    print(f"  fingerprint     : {fp}")
    try:
        verify_encoder_stable()
        print("  stability check : PASSED")
    except AssertionError as e:
        print(f"  stability check : FAILED\n{e}")
