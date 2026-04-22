"""Tests for move_encoder.py.

Enforces:
- correct one-hot layout for normal moves
- all promotion types are distinct and correctly placed
- determinism (same UCI → same vector, always)
- fingerprint stability
- encoding of an invalid UCI raises ValueError
"""

from __future__ import annotations

import pytest

chess = pytest.importorskip("chess", reason="python-chess not installed (run: uv sync --extra training)")
np = pytest.importorskip("numpy", reason="numpy not installed (run: uv sync --extra training)")

from searchess_ai.model_runtime.move_encoder import (
    MOVE_ENCODER_VERSION,
    MOVE_FEATURE_SIZE,
    PROMOTION_BISHOP,
    PROMOTION_KNIGHT,
    PROMOTION_NONE,
    PROMOTION_QUEEN,
    PROMOTION_ROOK,
    REFERENCE_MOVE,
    compute_move_encoder_fingerprint,
    encode_move,
    encode_moves,
    move_encoder_config,
)

_FROM_OFFSET = 0
_TO_OFFSET = 64
_PROMO_OFFSET = 128


# ---------------------------------------------------------------------------
# Output shape and value constraints
# ---------------------------------------------------------------------------


class TestMoveEncoderShape:
    def test_output_shape(self):
        vec = encode_move("e2e4")
        assert vec.shape == (MOVE_FEATURE_SIZE,)

    def test_output_dtype(self):
        vec = encode_move("e2e4")
        assert vec.dtype == np.float32

    def test_feature_size_is_133(self):
        assert MOVE_FEATURE_SIZE == 133

    def test_values_are_binary(self):
        for uci in ["e2e4", "d7d5", "e7e8q", "e7e8r", "e7e8b", "e7e8n", "e1g1"]:
            vec = encode_move(uci)
            assert set(vec.tolist()).issubset({0.0, 1.0}), f"Non-binary values for {uci}"

    def test_exactly_three_nonzero_bits(self):
        """Every encoded move must set exactly 3 bits: from, to, promotion class."""
        for uci in ["e2e4", "d7d5", "e7e8q", "e7e8r", "e7e8b", "e7e8n"]:
            vec = encode_move(uci)
            count = int((vec != 0).sum())
            assert count == 3, f"Expected 3 non-zero bits for {uci}, got {count}"


# ---------------------------------------------------------------------------
# Normal move layout
# ---------------------------------------------------------------------------


class TestNormalMoveLayout:
    def test_e2e4_from_square(self):
        # e2 = square 12
        vec = encode_move("e2e4")
        assert vec[_FROM_OFFSET + 12] == 1.0

    def test_e2e4_to_square(self):
        # e4 = square 28
        vec = encode_move("e2e4")
        assert vec[_TO_OFFSET + 28] == 1.0

    def test_e2e4_no_promotion(self):
        vec = encode_move("e2e4")
        assert vec[_PROMO_OFFSET + PROMOTION_NONE] == 1.0
        for cls in (PROMOTION_QUEEN, PROMOTION_ROOK, PROMOTION_BISHOP, PROMOTION_KNIGHT):
            assert vec[_PROMO_OFFSET + cls] == 0.0

    def test_a1h8_move(self):
        # a1=0, h8=63
        vec = encode_move("a1h8")
        assert vec[_FROM_OFFSET + 0] == 1.0
        assert vec[_TO_OFFSET + 63] == 1.0
        assert vec[_PROMO_OFFSET + PROMOTION_NONE] == 1.0


# ---------------------------------------------------------------------------
# Promotion moves — all four types are distinct
# ---------------------------------------------------------------------------


class TestPromotionMoves:
    def test_queen_promotion(self):
        vec = encode_move("e7e8q")
        assert vec[_PROMO_OFFSET + PROMOTION_QUEEN] == 1.0
        for cls in (PROMOTION_NONE, PROMOTION_ROOK, PROMOTION_BISHOP, PROMOTION_KNIGHT):
            assert vec[_PROMO_OFFSET + cls] == 0.0

    def test_rook_promotion(self):
        vec = encode_move("e7e8r")
        assert vec[_PROMO_OFFSET + PROMOTION_ROOK] == 1.0
        for cls in (PROMOTION_NONE, PROMOTION_QUEEN, PROMOTION_BISHOP, PROMOTION_KNIGHT):
            assert vec[_PROMO_OFFSET + cls] == 0.0

    def test_bishop_promotion(self):
        vec = encode_move("e7e8b")
        assert vec[_PROMO_OFFSET + PROMOTION_BISHOP] == 1.0
        for cls in (PROMOTION_NONE, PROMOTION_QUEEN, PROMOTION_ROOK, PROMOTION_KNIGHT):
            assert vec[_PROMO_OFFSET + cls] == 0.0

    def test_knight_promotion(self):
        vec = encode_move("e7e8n")
        assert vec[_PROMO_OFFSET + PROMOTION_KNIGHT] == 1.0
        for cls in (PROMOTION_NONE, PROMOTION_QUEEN, PROMOTION_ROOK, PROMOTION_BISHOP):
            assert vec[_PROMO_OFFSET + cls] == 0.0

    def test_all_promotions_share_same_squares(self):
        """Same from/to square bits; only promotion class differs."""
        vq = encode_move("e7e8q")
        vr = encode_move("e7e8r")
        vb = encode_move("e7e8b")
        vn = encode_move("e7e8n")
        # from square bits must be equal
        assert (vq[:64] == vr[:64]).all()
        assert (vq[:64] == vb[:64]).all()
        assert (vq[:64] == vn[:64]).all()
        # to square bits must be equal
        assert (vq[64:128] == vr[64:128]).all()
        # promotion class bits must differ
        assert not (vq[128:] == vr[128:]).all()
        assert not (vq[128:] == vb[128:]).all()
        assert not (vq[128:] == vn[128:]).all()

    def test_all_promotions_are_distinct(self):
        """No two promotion types must produce identical vectors."""
        vq = encode_move("e7e8q").tobytes()
        vr = encode_move("e7e8r").tobytes()
        vb = encode_move("e7e8b").tobytes()
        vn = encode_move("e7e8n").tobytes()
        assert len({vq, vr, vb, vn}) == 4, "All four promotion vectors must be distinct"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestMoveEncoderDeterminism:
    def test_same_uci_always_same_vector(self):
        for uci in ["e2e4", "d7d5", "e7e8q", "e7e8n", "g1f3"]:
            v1 = encode_move(uci)
            v2 = encode_move(uci)
            assert (v1 == v2).all(), f"Non-deterministic encoding for {uci}"

    def test_encode_moves_stacks_correctly(self):
        ucis = ["e2e4", "d2d4", "g1f3"]
        batch = encode_moves(ucis)
        assert batch.shape == (3, MOVE_FEATURE_SIZE)
        for i, uci in enumerate(ucis):
            assert (batch[i] == encode_move(uci)).all()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestMoveEncoderErrors:
    def test_invalid_uci_raises(self):
        with pytest.raises(ValueError, match="Invalid UCI move"):
            encode_move("notvalid")

    def test_empty_move_list_raises(self):
        with pytest.raises(ValueError):
            encode_moves([])


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestMoveEncoderFingerprint:
    def test_fingerprint_is_deterministic(self):
        assert compute_move_encoder_fingerprint() == compute_move_encoder_fingerprint()

    def test_fingerprint_is_hex_string_of_length_64(self):
        fp = compute_move_encoder_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_different_moves_have_different_fingerprints(self):
        fp1 = compute_move_encoder_fingerprint("e2e4")
        fp2 = compute_move_encoder_fingerprint("d2d4")
        assert fp1 != fp2

    def test_config_contains_fingerprint(self):
        cfg = move_encoder_config()
        assert "fingerprint" in cfg
        assert len(cfg["fingerprint"]) == 64

    def test_config_version_matches_constant(self):
        cfg = move_encoder_config()
        assert cfg["version"] == MOVE_ENCODER_VERSION

    def test_config_move_feature_size_matches_constant(self):
        cfg = move_encoder_config()
        assert cfg["move_feature_size"] == MOVE_FEATURE_SIZE
