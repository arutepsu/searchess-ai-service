Title: Replace Fixed 4096 Move Vocabulary with Explicit Move Encoding

Status: Proposed

Context
--------

The current system encodes moves using a fixed-size vocabulary (size = 4096),
mapping UCI moves to integer indices.

Limitations of this approach:

1. Promotion ambiguity
   - UCI promotions (e.g. e7e8q, e7e8r, etc.) are collapsed
   - runtime assumes queen promotion
   - model cannot learn underpromotion strategies

2. Wasted output space
   - many of the 4096 slots are unused or invalid for a given position
   - inefficiency in learning and inference

3. Poor generalization
   - fixed vocabulary ties representation to a specific encoding scheme
   - harder to evolve or extend

4. Masking complexity
   - requires mapping legal moves → indices → masking logits
   - adds coupling between encoder and runtime

Given future goals:
- stronger supervised learning
- potential RL/self-play
- engine-assisted training

The current representation is insufficient.

Decision
--------

We replace the fixed vocabulary with an **explicit move representation**:

Each move is encoded as structured components:

- from_square: int (0–63)
- to_square: int (0–63)
- promotion: int (0–4)
    0 = none
    1 = queen
    2 = rook
    3 = bishop
    4 = knight

This results in:

- 64 × 64 × 5 = 20,480 possible moves
- but only legal moves are considered at runtime

Model output is no longer a flat 4096 vector.
Instead:

- The model produces a **score for each candidate move**
- Candidates are provided explicitly (legal moves)

This shifts from:
    "predict over fixed vocabulary"
to:
    "score a set of valid actions"

Architecture change:

Old:
    input → model → logits[4096] → mask → argmax

New:
    input + move_features → model → scores[N] → argmax

Where:
- N = number of legal moves (~20–40 typical)

--------------------------------
Model Interface
--------------------------------

New forward signature:

    score_moves(position_features, move_features) → scores

Where:
- position_features: tensor (773,)
- move_features: tensor (N, move_feature_dim)

--------------------------------
Move Feature Encoding
--------------------------------

Each move is encoded as a vector containing:

Required:
- from_square (one-hot or embedding)
- to_square (one-hot or embedding)
- promotion type (one-hot)

Optional extensions (future):
- piece type
- capture flag
- check flag

--------------------------------
Runtime Changes
--------------------------------

Runtime flow becomes:

1. encode_fen → position_features
2. legal_moves → list of UCI
3. encode each move → move_features
4. model scores all moves
5. select argmax

No vocabulary mapping required.
No global mask required.

--------------------------------
Artifact Changes
--------------------------------

Artifacts must now include:

- encoder_config:
    - feature_size (position)
    - move_feature_dim
    - encoder_version

- model_config:
    - architecture compatible with move scoring

--------------------------------
Backward Compatibility
--------------------------------

Breaking change.

Old artifacts (4096 vocab) are NOT compatible.

Migration strategy:

- introduce new encoder_version
- loader rejects old artifacts
- optionally keep legacy runtime behind a flag (short-term only)

--------------------------------
Consequences
--------------------------------

Positive:

- correct handling of promotions
- smaller effective decision space
- better learning signal (only legal moves)
- simpler masking logic (removed)
- more flexible for future models (RL, MCTS, etc.)

Negative:

- requires model redesign
- changes training pipeline
- breaks compatibility with existing artifacts

--------------------------------
Alternatives Considered
--------------------------------

1. Expand vocabulary to include promotions
   → rejected: still inefficient and rigid

2. Keep vocabulary and improve masking
   → rejected: does not solve representation limitations

3. Use policy head with full board tensor output
   → rejected: unnecessary complexity for current stage

--------------------------------
Decision Outcome
--------------------------------

Adopt explicit move encoding with per-move scoring.

This becomes the new standard representation for:
- training
- inference
- artifact format

--------------------------------
Next Steps
--------------------------------

1. Implement move encoder module
2. Refactor model to accept move features
3. Update training pipeline
4. Update runtime scoring logic
5. Add compatibility checks (encoder_version bump)