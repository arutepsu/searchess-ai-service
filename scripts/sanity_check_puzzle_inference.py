"""Sanity-check runtime inference on puzzle positions.

After training on the puzzle dataset, run this script to verify that the
produced artifact loads correctly and produces sensible MODEL-mode decisions.

Usage:
  uv run python scripts/sanity_check_puzzle_inference.py \\
      --artifact-dir <path/to/artifact/run_...> \\
      --dataset-dir  <path/to/puzzle/dataset>   \\
      [--n-positions 10]

Exit codes:
  0 — all checks passed
  1 — one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

_PASS = "PASS"
_FAIL = "FAIL"


def _check(label: str, condition: bool, detail: str = "") -> bool:
    mark = _PASS if condition else _FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return condition


def run_checks(artifact_dir: Path, dataset_dir: Path, n_positions: int) -> bool:
    """Load the artifact and run inference on n_positions from the dataset.

    Returns True if all checks pass.
    """
    try:
        import pandas as pd
        import chess
    except ImportError:
        print("ERROR: pandas and python-chess required. uv sync --extra training", file=sys.stderr)
        sys.exit(1)

    try:
        from searchess_ai.model_runtime.loader import load_artifact
        from searchess_ai.model_runtime.runtime import SupervisedModelRuntime
        from searchess_ai.model_runtime.decision import DecisionMode
    except ImportError as exc:
        print(f"ERROR: could not import runtime: {exc}", file=sys.stderr)
        sys.exit(1)

    all_ok = True

    # ------------------------------------------------------------------
    # 1. Artifact loading
    # ------------------------------------------------------------------
    print("\n=== 1. Artifact loading ===")
    try:
        loaded = load_artifact(artifact_dir)
        runtime = SupervisedModelRuntime(loaded)
        all_ok &= _check("load_artifact() succeeded", True)
        all_ok &= _check(
            "model_version present",
            bool(loaded.model_version),
            loaded.model_version,
        )
        all_ok &= _check(
            "artifact_id present",
            bool(loaded.artifact_id),
            loaded.artifact_id[:8] + "…",
        )
        all_ok &= _check(
            "encoder_version present",
            bool(loaded.encoder_version),
            loaded.encoder_version,
        )
        all_ok &= _check(
            "move_encoder_version present",
            bool(loaded.move_encoder_version),
            loaded.move_encoder_version,
        )
    except Exception as exc:
        _check("load_artifact() succeeded", False, str(exc))
        print("\nArtifact failed to load — aborting remaining checks.")
        return False

    # ------------------------------------------------------------------
    # 2. Dataset presence
    # ------------------------------------------------------------------
    print("\n=== 2. Dataset ===")
    parquet_path = dataset_dir / "prepared_samples.parquet"
    metadata_path = dataset_dir / "dataset_metadata.json"

    parquet_ok = _check("prepared_samples.parquet exists", parquet_path.exists())
    meta_ok = _check("dataset_metadata.json exists", metadata_path.exists())
    all_ok &= parquet_ok and meta_ok

    if not parquet_ok:
        print("Dataset not found — cannot run inference checks.")
        return False

    df = pd.read_parquet(parquet_path)
    all_ok &= _check("parquet has rows", len(df) > 0, f"{len(df):,} rows")

    if meta_ok:
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        all_ok &= _check(
            "dataset_type is lichess_puzzles",
            meta.get("dataset_type") == "lichess_puzzles",
            meta.get("dataset_type", "<missing>"),
        )

    # ------------------------------------------------------------------
    # 3. Inference on sample positions
    # ------------------------------------------------------------------
    print(f"\n=== 3. Inference on {n_positions} positions ===")

    sample_df = df.sample(n=min(n_positions, len(df)), random_state=0)
    decision_modes: list[str] = []
    confidences: list[float] = []
    fallback_rows: list[str] = []

    for _, row in sample_df.iterrows():
        fen = str(row["position_fen"])
        legal_json = row.get("legal_moves_uci")
        played = str(row["played_move_uci"])

        # Recompute legal moves from FEN (ground-truth validation)
        try:
            board = chess.Board(fen)
            legal_uci = [m.uci() for m in board.legal_moves]
        except Exception:
            legal_uci = json.loads(legal_json) if legal_json else []

        decision = runtime.select_move(fen, legal_uci)
        decision_modes.append(decision.decision_mode.value)

        if decision.decision_mode == DecisionMode.MODEL:
            confidences.append(decision.confidence or 0.0)
            # Verify selected move is legal
            if decision.selected_uci not in legal_uci:
                fallback_rows.append(f"{fen[:30]}… → {decision.selected_uci} not in legal")
        else:
            fallback_rows.append(
                f"{fen[:30]}… fallback={decision.fallback_reason} "
                f"detail={decision.error_detail}"
            )

    model_decisions = decision_modes.count("model")
    fallback_decisions = len(decision_modes) - model_decisions

    all_ok &= _check(
        "all decisions are MODEL mode",
        fallback_decisions == 0,
        f"{model_decisions} model / {fallback_decisions} fallback",
    )
    if fallback_rows:
        for msg in fallback_rows:
            print(f"       ! {msg}")

    all_ok &= _check(
        "no illegal moves returned",
        len(fallback_rows) == 0,
        f"{len(fallback_rows)} issues",
    )

    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        all_ok &= _check(
            "confidence in (0, 1]",
            all(0.0 < c <= 1.0 for c in confidences),
            f"avg={avg_conf:.3f}",
        )

    # ------------------------------------------------------------------
    # 4. Decision transparency — spot check one position verbosely
    # ------------------------------------------------------------------
    print("\n=== 4. Decision transparency (spot check) ===")
    row = sample_df.iloc[0]
    fen = str(row["position_fen"])
    try:
        board = chess.Board(fen)
        legal_uci = [m.uci() for m in board.legal_moves]
    except Exception:
        legal_uci = []

    decision = runtime.select_move(fen, legal_uci)
    print(f"  FEN            : {fen}")
    print(f"  Legal moves    : {len(legal_uci)} moves")
    print(f"  Selected       : {decision.selected_uci}")
    print(f"  Decision mode  : {decision.decision_mode.value}")
    print(f"  Confidence     : {decision.confidence}")
    print(f"  Fallback reason: {decision.fallback_reason}")

    all_ok &= _check(
        "transparency fields populated",
        decision.selected_uci != "" and decision.decision_mode is not None,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + ("=" * 40))
    if all_ok:
        print(f"[{_PASS}] All checks passed.")
    else:
        print(f"[{_FAIL}] Some checks FAILED - review output above.")
    print("=" * 40)

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check a trained artifact on puzzle dataset positions."
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Path to the artifact directory (e.g. artifacts/run_...)",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the puzzle dataset directory (contains prepared_samples.parquet)",
    )
    parser.add_argument(
        "--n-positions",
        type=int,
        default=10,
        help="Number of positions to run inference on (default: 10)",
    )
    args = parser.parse_args(argv)

    ok = run_checks(
        artifact_dir=Path(args.artifact_dir),
        dataset_dir=Path(args.dataset_dir),
        n_positions=args.n_positions,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
