from __future__ import annotations

import json
from io import StringIO

import pytest

pandas = pytest.importorskip("pandas", reason="pandas not installed")
pytest.importorskip("pyarrow", reason="pyarrow not installed")
chess = pytest.importorskip("chess", reason="python-chess not installed")

from searchess_ai.data_pipeline.full_game_pipeline import (
    FullGameDatasetConfig,
    extract_full_game_samples,
    run_full_game_pipeline,
)
from searchess_ai.data_pipeline.writer import METADATA_FILENAME, PARQUET_FILENAME


PGN_TEXT = """
[Event "Rated Blitz game"]
[Site "https://lichess.org/test1"]
[Date "2024.01.01"]
[Round "-"]
[White "WhiteA"]
[Black "BlackA"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1750"]
[TimeControl "300+0"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/test2"]
[Date "2024.01.01"]
[Round "-"]
[White "WhiteB"]
[Black "BlackB"]
[Result "0-1"]
[WhiteElo "1900"]
[BlackElo "1850"]
[TimeControl "300+0"]
[Termination "Normal"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 0-1
""".strip()


def test_extract_full_game_samples_honors_min_ply_and_remaining_cap():
    game = chess.pgn.read_game(StringIO(PGN_TEXT))

    samples = extract_full_game_samples(
        game,
        game_id="game_1",
        include_legal_moves=True,
        min_sample_ply_index=4,
        max_remaining_samples=3,
    )

    assert len(samples) == 3
    assert [sample.ply_index for sample in samples] == [4, 5, 6]
    assert all(sample.legal_moves_uci for sample in samples)
    assert samples[0].played_move_uci in samples[0].legal_moves_uci


def test_run_full_game_pipeline_writes_bounded_dataset(tmp_path):
    pgn_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "dataset"
    pgn_path.write_text(PGN_TEXT, encoding="utf-8")

    metadata = run_full_game_pipeline(
        FullGameDatasetConfig(
            source_pgn=pgn_path,
            output_dir=output_dir,
            dataset_name="test_full_game",
            max_games=10,
            max_samples=5,
            min_game_ply_count=6,
            min_sample_ply_index=2,
        )
    )

    parquet_path = output_dir / PARQUET_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    assert parquet_path.exists()
    assert metadata_path.exists()
    assert metadata["total_samples"] == 5
    assert metadata["games_accepted"] == 1
    assert metadata["pipeline_config"]["max_samples"] == 5
    assert metadata["pipeline_config"]["min_sample_ply_index"] == 2

    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    df = pandas.read_parquet(parquet_path)

    assert persisted["total_samples"] == 5
    assert len(df) == 5
    assert set(df["source_game_id"]) == {"full_game_00000000"}
    assert df["ply_index"].min() >= 2
