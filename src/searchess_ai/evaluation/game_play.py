"""Lightweight full-game evaluation for trained move-scoring artifacts.

This module intentionally stays small: it runs local standard-chess games
between the real supervised runtime and a random legal-move baseline. It is not
an Elo benchmark or a replacement for stronger engine-based evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

try:
    import chess
except ImportError as exc:  # pragma: no cover - exercised when extras are absent
    raise ImportError(
        "python-chess is required for game-play evaluation. "
        "Install training extras: uv sync --extra training"
    ) from exc

from searchess_ai.model_runtime.decision import DecisionMode, RuntimeDecision
from searchess_ai.model_runtime.loader import load_artifact
from searchess_ai.model_runtime.runtime import SupervisedModelRuntime


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Explicit, serialisable configuration for a model-vs-random match."""

    artifact_dir: Path
    games: int = 20
    max_plies: int = 300
    seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class AgentMove:
    """A move selected by an evaluation agent plus optional runtime metadata."""

    move: chess.Move
    runtime_decision: RuntimeDecision | None = None


class EvaluationAgent(Protocol):
    """Minimal agent interface for local full-game evaluation."""

    name: str

    def choose_move(self, board: chess.Board) -> AgentMove:
        """Choose one move from board.legal_moves."""


class RandomAgent:
    """Uniformly samples one legal move using an injected deterministic RNG."""

    name = "random"

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose_move(self, board: chess.Board) -> AgentMove:
        legal_moves = list(board.legal_moves)
        return AgentMove(self._rng.choice(legal_moves))


class ModelAgent:
    """Chooses moves through the real trained-artifact runtime path."""

    name = "model"

    def __init__(self, runtime: SupervisedModelRuntime) -> None:
        self._runtime = runtime

    @classmethod
    def from_artifact(cls, artifact_dir: Path, *, device: str = "cpu") -> "ModelAgent":
        loaded = load_artifact(artifact_dir, device=device)
        return cls(SupervisedModelRuntime(loaded))

    def choose_move(self, board: chess.Board) -> AgentMove:
        legal_ucis = [move.uci() for move in board.legal_moves]
        decision = self._runtime.select_move(board.fen(), legal_ucis)
        return AgentMove(chess.Move.from_uci(decision.selected_uci), decision)


@dataclass(slots=True)
class GameResult:
    """Result of one completed or capped evaluation game."""

    game_index: int
    model_color: str
    result: str
    winner: str | None
    plies: int
    termination: str
    model_fallbacks: int = 0
    model_moves: int = 0
    illegal_moves: int = 0


@dataclass(slots=True)
class MatchSummary:
    """Aggregate model-vs-random evaluation result."""

    config: EvaluationConfig
    artifact_id: str
    model_version: str
    total_games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    average_game_length_plies: float
    model_moves: int
    fallback_count: int
    fallback_rate: float
    illegal_move_count: int
    terminations: dict[str, int]
    by_side: dict[str, dict[str, int]]
    games: list[GameResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["config"]["artifact_dir"] = str(self.config.artifact_dir)
        return data


def find_latest_artifact(artifacts_root: Path = Path("artifacts")) -> Path:
    """Return the newest complete-looking artifact directory below artifacts_root."""

    candidates = [
        path
        for path in artifacts_root.iterdir()
        if path.is_dir()
        and (path / "manifest.json").exists()
        and (path / "model.pt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No trained artifacts found below {artifacts_root}. "
            "Pass --artifact-dir explicitly after training a model."
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def play_game(
    *,
    game_index: int,
    white: EvaluationAgent,
    black: EvaluationAgent,
    model_color: str,
    max_plies: int,
) -> GameResult:
    """Play one standard chess game from the initial position."""

    board = chess.Board()
    model_fallbacks = 0
    model_moves = 0
    illegal_moves = 0

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        agent = white if board.turn == chess.WHITE else black
        agent_move = agent.choose_move(board)
        move = agent_move.move

        if agent.name == "model":
            model_moves += 1
            if (
                agent_move.runtime_decision is not None
                and agent_move.runtime_decision.decision_mode != DecisionMode.MODEL
            ):
                model_fallbacks += 1

        if move not in board.legal_moves:
            illegal_moves += 1
            move = next(iter(board.legal_moves))

        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result = "1/2-1/2"
        winner = None
        termination = "max_ply"
    else:
        result = outcome.result()
        winner = _winner_name(outcome.winner)
        termination = outcome.termination.name.lower()

    return GameResult(
        game_index=game_index,
        model_color=model_color,
        result=result,
        winner=winner,
        plies=board.ply(),
        termination=termination,
        model_fallbacks=model_fallbacks,
        model_moves=model_moves,
        illegal_moves=illegal_moves,
    )


def evaluate_model_vs_random(config: EvaluationConfig) -> MatchSummary:
    """Run alternating-color full games between the artifact model and random."""

    if config.games <= 0:
        raise ValueError("games must be greater than zero")
    if config.max_plies <= 0:
        raise ValueError("max_plies must be greater than zero")

    artifact_dir = config.artifact_dir
    loaded = load_artifact(artifact_dir, device=config.device)
    model_agent = ModelAgent(SupervisedModelRuntime(loaded))
    random_agent = RandomAgent(random.Random(config.seed))

    game_results: list[GameResult] = []
    for game_index in range(config.games):
        model_is_white = game_index % 2 == 0
        result = play_game(
            game_index=game_index + 1,
            white=model_agent if model_is_white else random_agent,
            black=random_agent if model_is_white else model_agent,
            model_color="white" if model_is_white else "black",
            max_plies=config.max_plies,
        )
        game_results.append(result)

    return summarize_results(
        config=config,
        artifact_id=loaded.artifact_id,
        model_version=loaded.model_version,
        game_results=game_results,
    )


def summarize_results(
    *,
    config: EvaluationConfig,
    artifact_id: str,
    model_version: str,
    game_results: list[GameResult],
) -> MatchSummary:
    """Aggregate per-game results from the model perspective."""

    wins = losses = draws = 0
    by_side = {
        "white": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        "black": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
    }
    terminations: Counter[str] = Counter()

    for result in game_results:
        side = by_side[result.model_color]
        side["games"] += 1
        terminations[result.termination] += 1

        if result.winner == result.model_color:
            wins += 1
            side["wins"] += 1
        elif result.winner is None:
            draws += 1
            side["draws"] += 1
        else:
            losses += 1
            side["losses"] += 1

    total_games = len(game_results)
    total_model_moves = sum(result.model_moves for result in game_results)
    total_fallbacks = sum(result.model_fallbacks for result in game_results)
    total_illegal = sum(result.illegal_moves for result in game_results)
    total_plies = sum(result.plies for result in game_results)

    return MatchSummary(
        config=config,
        artifact_id=artifact_id,
        model_version=model_version,
        total_games=total_games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=round(wins / total_games, 4) if total_games else 0.0,
        average_game_length_plies=round(total_plies / total_games, 2)
        if total_games
        else 0.0,
        model_moves=total_model_moves,
        fallback_count=total_fallbacks,
        fallback_rate=round(total_fallbacks / total_model_moves, 4)
        if total_model_moves
        else 0.0,
        illegal_move_count=total_illegal,
        terminations=dict(sorted(terminations.items())),
        by_side=by_side,
        games=game_results,
    )


def _winner_name(winner: bool | None) -> str | None:
    if winner is None:
        return None
    return "white" if winner else "black"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Searchess model against random legal moves."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Trained artifact directory. Defaults to latest directory below artifacts/.",
    )
    parser.add_argument("--games", type=int, default=20, help="Number of games to run.")
    parser.add_argument(
        "--max-plies",
        type=int,
        default=300,
        help="Ply cap per game; capped games count as draws.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random baseline seed.")
    parser.add_argument("--device", default="cpu", help="Torch device for artifact loading.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full machine-readable result JSON.",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or find_latest_artifact()
    config = EvaluationConfig(
        artifact_dir=artifact_dir,
        games=args.games,
        max_plies=args.max_plies,
        seed=args.seed,
        device=args.device,
    )
    summary = evaluate_model_vs_random(config)

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(_format_summary(summary))


def _format_summary(summary: MatchSummary) -> str:
    side_lines = [
        f"  model as {side}: {stats['wins']}W/{stats['losses']}L/{stats['draws']}D "
        f"over {stats['games']} games"
        for side, stats in summary.by_side.items()
    ]
    termination_lines = [
        f"  {termination}: {count}"
        for termination, count in summary.terminations.items()
    ]
    return "\n".join(
        [
            "Model vs random evaluation",
            f"artifact_dir: {summary.config.artifact_dir}",
            f"artifact_id: {summary.artifact_id}",
            f"model_version: {summary.model_version}",
            f"games: {summary.total_games}",
            f"score: {summary.wins}W/{summary.losses}L/{summary.draws}D",
            f"win_rate: {summary.win_rate:.4f}",
            f"average_game_length_plies: {summary.average_game_length_plies:.2f}",
            f"model_moves: {summary.model_moves}",
            f"fallbacks: {summary.fallback_count} ({summary.fallback_rate:.4f})",
            f"illegal_moves: {summary.illegal_move_count}",
            "by_side:",
            *side_lines,
            "terminations:",
            *termination_lines,
        ]
    )


if __name__ == "__main__":
    main()
