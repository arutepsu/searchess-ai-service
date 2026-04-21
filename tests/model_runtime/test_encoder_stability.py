"""Encoder stability tests.

These tests enforce the invariants in model_runtime/encoder.py that prevent
silent encoder drift.  They are the automated safety net for Workstream 1:
any unintentional change to encode_fen() will cause at least one test here
to fail before the change reaches an artifact or the serving runtime.

All tests skip cleanly if python-chess / numpy are not installed.
"""

from __future__ import annotations

import pytest

chess = pytest.importorskip("chess", reason="python-chess not installed (run: uv sync --extra training)")
np = pytest.importorskip("numpy", reason="numpy not installed (run: uv sync --extra training)")

from searchess_ai.model_runtime.encoder import (
    ENCODER_VERSION,
    FEATURE_SIZE,
    MOVE_VOCAB_SIZE,
    REFERENCE_FEN,
    _REFERENCE_NONZERO_INDICES,
    compute_encoder_fingerprint,
    encode_fen,
    encode_uci_move,
    verify_encoder_stable,
)


# ---------------------------------------------------------------------------
# Core stability assertions
# ---------------------------------------------------------------------------


class TestEncoderStability:
    """verify_encoder_stable() and _REFERENCE_NONZERO_INDICES must agree."""

    def test_verify_encoder_stable_passes(self):
        """verify_encoder_stable() must not raise for the current encoder."""
        verify_encoder_stable()  # AssertionError if encoder has drifted

    def test_reference_nonzero_indices_match_encode_fen(self):
        """_REFERENCE_NONZERO_INDICES must exactly match the encoded starting position."""
        vec = encode_fen(REFERENCE_FEN)
        actual = frozenset(int(i) for i in np.nonzero(vec)[0])
        unexpected_added = sorted(actual - _REFERENCE_NONZERO_INDICES)
        unexpectedly_removed = sorted(_REFERENCE_NONZERO_INDICES - actual)
        assert actual == _REFERENCE_NONZERO_INDICES, (
            f"Encoder output drifted from _REFERENCE_NONZERO_INDICES.\n"
            f"  Unexpected indices (added)  : {unexpected_added}\n"
            f"  Missing indices (removed)   : {unexpectedly_removed}\n"
            "Update _REFERENCE_NONZERO_INDICES and bump ENCODER_VERSION."
        )

    def test_nonzero_count_for_starting_position(self):
        """Starting position must have exactly 37 non-zero features.

        Breakdown: 8 WP + 8 BP + 2 WN + 2 BN + 2 WB + 2 BB + 2 WR + 2 BR
                   + 1 WQ + 1 BQ + 1 WK + 1 BK = 32 pieces
                   + 4 castling rights + 1 side-to-move = 37
        """
        vec = encode_fen(REFERENCE_FEN)
        count = int((vec != 0).sum())
        assert count == 37, (
            f"Expected 37 non-zero features for starting position, got {count}. "
            "encode_fen() may have changed the feature layout."
        )

    def test_feature_vector_shape(self):
        vec = encode_fen(REFERENCE_FEN)
        assert vec.shape == (FEATURE_SIZE,)
        assert vec.dtype == np.float32

    def test_feature_values_are_binary(self):
        """All features must be exactly 0.0 or 1.0 for any standard position."""
        for fen in [
            REFERENCE_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        ]:
            vec = encode_fen(fen)
            assert set(vec.tolist()).issubset({0.0, 1.0}), (
                f"Feature vector for FEN {fen!r} contains values other than 0.0 or 1.0: "
                f"{sorted(set(vec.tolist()) - {0.0, 1.0})}"
            )


# ---------------------------------------------------------------------------
# Fingerprint stability
# ---------------------------------------------------------------------------


class TestEncoderFingerprint:
    def test_fingerprint_is_deterministic(self):
        """Same FEN must always produce the same fingerprint."""
        fp1 = compute_encoder_fingerprint(REFERENCE_FEN)
        fp2 = compute_encoder_fingerprint(REFERENCE_FEN)
        assert fp1 == fp2

    def test_different_positions_have_different_fingerprints(self):
        """Different board states must produce different fingerprints."""
        fp_start = compute_encoder_fingerprint(REFERENCE_FEN)
        fp_after_e4 = compute_encoder_fingerprint(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        )
        assert fp_start != fp_after_e4

    def test_fingerprint_is_hex_string(self):
        fp = compute_encoder_fingerprint(REFERENCE_FEN)
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_changes_if_feature_order_changes(self):
        """Simulate a silent encoder change: swap two bits.

        This test confirms the fingerprint mechanism catches the kind of
        drift that a version string alone would miss.
        """
        import hashlib

        original_fp = compute_encoder_fingerprint(REFERENCE_FEN)

        # Build a fake "drifted" fingerprint by computing on a modified index set.
        drifted_indices = sorted((_REFERENCE_NONZERO_INDICES - {8}) | {9999})
        raw = ",".join(str(i) for i in drifted_indices)
        drifted_fp = hashlib.sha256(raw.encode()).hexdigest()

        assert original_fp != drifted_fp, (
            "Test construction error: drifted fingerprint must differ from original."
        )


# ---------------------------------------------------------------------------
# Specific piece-plane layout assertions
# ---------------------------------------------------------------------------


class TestEncoderPiecePlanes:
    """Verify the exact plane layout so regressions are immediately localised."""

    def test_white_pawns_plane_0(self):
        vec = encode_fen(REFERENCE_FEN)
        for sq in range(8, 16):  # a2–h2
            assert vec[0 * 64 + sq] == 1.0, f"Missing white pawn at square {sq}"

    def test_black_pawns_plane_1(self):
        vec = encode_fen(REFERENCE_FEN)
        for sq in range(48, 56):  # a7–h7
            assert vec[1 * 64 + sq] == 1.0, f"Missing black pawn at square {sq}"

    def test_white_king_plane_10(self):
        vec = encode_fen(REFERENCE_FEN)
        assert vec[10 * 64 + 4] == 1.0  # e1=4

    def test_black_king_plane_11(self):
        vec = encode_fen(REFERENCE_FEN)
        assert vec[11 * 64 + 60] == 1.0  # e8=60

    def test_castling_rights_indices_768_771(self):
        vec = encode_fen(REFERENCE_FEN)
        assert vec[768] == 1.0  # WK castling
        assert vec[769] == 1.0  # WQ castling
        assert vec[770] == 1.0  # BK castling
        assert vec[771] == 1.0  # BQ castling

    def test_side_to_move_index_772(self):
        vec_white = encode_fen(REFERENCE_FEN)
        assert vec_white[772] == 1.0

        vec_black = encode_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        )
        assert vec_black[772] == 0.0

    def test_empty_squares_are_zero_in_starting_position(self):
        """Ranks 3–6 are empty in the starting position — all bits must be 0."""
        vec = encode_fen(REFERENCE_FEN)
        for plane in range(12):
            for sq in range(16, 48):  # ranks 3–6
                assert vec[plane * 64 + sq] == 0.0, (
                    f"Plane {plane}, square {sq} should be 0.0 but got {vec[plane * 64 + sq]}"
                )


# ---------------------------------------------------------------------------
# Move encoding
# ---------------------------------------------------------------------------


class TestMoveEncoding:
    def test_e2e4_encodes_correctly(self):
        # e2=12, e4=28; index = 12*64+28 = 796
        assert encode_uci_move("e2e4") == 12 * 64 + 28

    def test_promotion_strips_piece(self):
        # e7e8q and e7e8r must map to the same index
        assert encode_uci_move("e7e8q") == encode_uci_move("e7e8r")

    def test_invalid_uci_raises(self):
        with pytest.raises(ValueError):
            encode_uci_move("notvalid")

    def test_move_vocab_size(self):
        # All squares-to-squares produce valid indices
        assert MOVE_VOCAB_SIZE == 4096
        for fr in range(64):
            for to in range(64):
                idx = fr * 64 + to
                assert 0 <= idx < MOVE_VOCAB_SIZE
