"""Deterministic UCI move encoding for the move-scoring policy network.

Move representation
-------------------
Each legal move is encoded as a fixed-size float32 vector of size
MOVE_FEATURE_SIZE (133):

  [  0 ..  63]  from_square — one-hot over 64 squares (a1=0 ... h8=63)
  [ 64 .. 127]  to_square   — one-hot over 64 squares
  [128 .. 132]  promotion   — one-hot over 5 classes:
                                0 = none  (non-promotion move)
                                1 = queen
                                2 = rook
                                3 = bishop
                                4 = knight

Underpromotions are preserved exactly.  Every (from, to, promotion) triple
maps to a unique 133-dimensional vector.

Governance
----------
MOVE_ENCODER_VERSION must be bumped whenever encode_move() changes in a way
that alters the output.  A behavioral fingerprint (SHA-256 of a fixed
reference encoding) is stored in every artifact's move_encoder_config and
re-verified at load time — the same two-layer protection used for the position
encoder.

Updating the move encoder
--------------------------
If encode_move() is intentionally changed:
  1. Update MOVE_ENCODER_VERSION.
  2. Re-run the full dataset → train pipeline.
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
# Constants
# ---------------------------------------------------------------------------

# Total feature dimensions: 64 (from) + 64 (to) + 5 (promotion) = 133
MOVE_FEATURE_SIZE: int = 133

MOVE_ENCODER_VERSION: str = "1.0"

# Promotion class indices (matches the last 5 features of the vector)
PROMOTION_NONE: int = 0
PROMOTION_QUEEN: int = 1
PROMOTION_ROOK: int = 2
PROMOTION_BISHOP: int = 3
PROMOTION_KNIGHT: int = 4

_PROMOTION_MAP: dict[int | None, int] = {
    None: PROMOTION_NONE,
    chess.QUEEN: PROMOTION_QUEEN,
    chess.ROOK: PROMOTION_ROOK,
    chess.BISHOP: PROMOTION_BISHOP,
    chess.KNIGHT: PROMOTION_KNIGHT,
}

# Offsets into the feature vector
_FROM_OFFSET: int = 0
_TO_OFFSET: int = 64
_PROMO_OFFSET: int = 128

# Reference move used to compute the behavioral fingerprint.
# Must be a stable, unambiguous UCI string.
REFERENCE_MOVE: str = "e2e4"


# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------


def encode_move(uci: str) -> np.ndarray:
    """UCI string → float32 feature vector of shape (MOVE_FEATURE_SIZE,).

    Raises ValueError for invalid or ambiguous UCI strings.
    """
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as exc:
        raise ValueError(f"Invalid UCI move: {uci!r}") from exc

    vec = np.zeros(MOVE_FEATURE_SIZE, dtype=np.float32)
    vec[_FROM_OFFSET + move.from_square] = 1.0
    vec[_TO_OFFSET + move.to_square] = 1.0
    promo_class = _PROMOTION_MAP.get(move.promotion, PROMOTION_NONE)
    vec[_PROMO_OFFSET + promo_class] = 1.0
    return vec


def encode_moves(ucis: list[str]) -> np.ndarray:
    """Encode a list of UCI moves to shape (N, MOVE_FEATURE_SIZE).

    Raises ValueError if any UCI string is invalid.
    """
    if not ucis:
        raise ValueError("Cannot encode an empty move list")
    return np.stack([encode_move(uci) for uci in ucis])


# ---------------------------------------------------------------------------
# Governance: fingerprint and metadata
# ---------------------------------------------------------------------------


def compute_move_encoder_fingerprint(uci: str = REFERENCE_MOVE) -> str:
    """SHA-256 of the sorted non-zero indices of encode_move(uci).

    Platform-independent (no endianness/float representation variation).
    Stored in every artifact and re-verified at load time.
    """
    vec = encode_move(uci)
    nonzero_indices = sorted(int(i) for i in np.nonzero(vec)[0])
    raw = ",".join(str(i) for i in nonzero_indices)
    return hashlib.sha256(raw.encode()).hexdigest()


def move_encoder_config() -> dict:
    """Serialisable snapshot of move encoding parameters for artifact packaging."""
    return {
        "version": MOVE_ENCODER_VERSION,
        "move_feature_size": MOVE_FEATURE_SIZE,
        "num_promotion_classes": 5,
        "fingerprint": compute_move_encoder_fingerprint(),
        "from_square_offset": _FROM_OFFSET,
        "to_square_offset": _TO_OFFSET,
        "promotion_offset": _PROMO_OFFSET,
        "promotion_classes": ["none", "queen", "rook", "bishop", "knight"],
    }


# ---------------------------------------------------------------------------
# Developer utility
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Move encoder self-check")
    print(f"  MOVE_ENCODER_VERSION : {MOVE_ENCODER_VERSION}")
    print(f"  MOVE_FEATURE_SIZE    : {MOVE_FEATURE_SIZE}")
    print(f"  REFERENCE_MOVE       : {REFERENCE_MOVE}")
    fp = compute_move_encoder_fingerprint()
    print(f"  fingerprint          : {fp}")
    print(f"  encode_move('e2e4')  : {np.nonzero(encode_move('e2e4'))[0].tolist()}")
    print(f"  encode_move('e7e8q') : {np.nonzero(encode_move('e7e8q'))[0].tolist()}")
    print(f"  encode_move('e7e8n') : {np.nonzero(encode_move('e7e8n'))[0].tolist()}")
