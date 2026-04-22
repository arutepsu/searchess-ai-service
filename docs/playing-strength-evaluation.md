# Playing-Strength Evaluation

This is a lightweight local check for the current trained move-scoring model.
It answers a narrow question: can the trained artifact play complete standard
chess games and beat a uniformly random legal-move baseline?

It is intentionally not a benchmark suite. It does not estimate Elo, use
Stockfish, run self-play training, or prove that the model is strong against
human or engine opponents.

## What It Runs

The evaluator plays full games from the normal initial chess position using
`python-chess`.

Agents:

- `ModelAgent`: loads the trained artifact through the real artifact loader and
  `SupervisedModelRuntime`.
- `RandomAgent`: samples uniformly from legal moves with a fixed seed.

Games terminate on normal `python-chess` game-over conditions. A configurable
max-ply cap prevents pathological long games; capped games are counted as
draws with termination `max_ply`.

## How To Run

Install the training extras first because the evaluator needs `python-chess`
and the trained-artifact runtime needs `torch`:

```powershell
uv sync --extra training
```

Run against the latest complete artifact under `artifacts/`:

```powershell
uv run evaluate-trained-model --games 20 --max-plies 300 --seed 42
```

Or pass an explicit artifact:

```powershell
uv run evaluate-trained-model --artifact-dir artifacts/run_20260422_102148_a5ee4efa --games 20
```

For machine-readable output:

```powershell
uv run evaluate-trained-model --games 20 --json
```

## Reported Metrics

The command reports:

- total games
- wins, losses, draws from the model perspective
- model win rate
- average game length in plies
- model runtime fallback count and rate
- illegal move count
- termination breakdown
- color split for model as White and model as Black

## Interpretation

A positive result means the artifact can survive repeated full-game runtime
inference and performs better than random legal moves in this small local setup.

It does not prove general chess strength. Random is a trivial baseline, the
sample size is usually small, and the model was trained from puzzle positions
rather than complete-game outcomes.
