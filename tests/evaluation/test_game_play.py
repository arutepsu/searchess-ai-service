from __future__ import annotations

import random
from pathlib import Path

import pytest

chess = pytest.importorskip("chess", reason="python-chess not installed")

from searchess_ai.evaluation.game_play import (
    AgentMove,
    EvaluationConfig,
    GameResult,
    RandomAgent,
    play_game,
    summarize_results,
)


class FirstLegalAgent:
    name = "model"

    def choose_move(self, board: chess.Board) -> AgentMove:
        return AgentMove(next(iter(board.legal_moves)))


def test_play_game_progresses_until_max_ply_cap():
    result = play_game(
        game_index=1,
        white=FirstLegalAgent(),
        black=FirstLegalAgent(),
        model_color="white",
        max_plies=4,
    )

    assert result.plies == 4
    assert result.termination == "max_ply"
    assert result.illegal_moves == 0
    assert result.model_moves == 4


def test_random_agent_is_reproducible():
    board = chess.Board()

    first = RandomAgent(random.Random(7)).choose_move(board).move
    second = RandomAgent(random.Random(7)).choose_move(board).move

    assert first == second


def test_summarize_results_tracks_model_perspective_and_side_breakdown():
    config = EvaluationConfig(artifact_dir=Path("artifacts/example"), games=3)
    summary = summarize_results(
        config=config,
        artifact_id="artifact-1",
        model_version="model-v1",
        game_results=[
            GameResult(
                game_index=1,
                model_color="white",
                result="1-0",
                winner="white",
                plies=20,
                termination="checkmate",
                model_fallbacks=0,
                model_moves=10,
            ),
            GameResult(
                game_index=2,
                model_color="black",
                result="1-0",
                winner="white",
                plies=30,
                termination="checkmate",
                model_fallbacks=1,
                model_moves=15,
            ),
            GameResult(
                game_index=3,
                model_color="white",
                result="1/2-1/2",
                winner=None,
                plies=40,
                termination="max_ply",
                model_fallbacks=0,
                model_moves=20,
            ),
        ],
    )

    assert summary.total_games == 3
    assert summary.wins == 1
    assert summary.losses == 1
    assert summary.draws == 1
    assert summary.win_rate == 0.3333
    assert summary.average_game_length_plies == 30.0
    assert summary.fallback_count == 1
    assert summary.fallback_rate == 0.0222
    assert summary.by_side["white"] == {
        "games": 2,
        "wins": 1,
        "losses": 0,
        "draws": 1,
    }
    assert summary.by_side["black"] == {
        "games": 1,
        "wins": 0,
        "losses": 1,
        "draws": 0,
    }
    assert summary.terminations == {"checkmate": 2, "max_ply": 1}
