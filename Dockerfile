FROM python:3.14-slim

WORKDIR /app

# Copy only what pip needs to install the package and its dependencies.
# Copying pyproject.toml + src in two steps keeps the dependency layer
# cacheable when only source code changes (not metadata).
COPY pyproject.toml README.md ./
COPY src/ src/

# Install production dependencies only (no pytest/pylint/httpx).
RUN pip install --no-cache-dir .

EXPOSE 8765

# INFERENCE_BACKEND controls which backend answers move-suggestion requests.
# "random" — picks a legal move at random; safe for integration testing.
# "fake"   — always picks legalMoves[0]; deterministic, used in unit tests.
# "openspiel" — requires open_spiel installed; not bundled in this image.
ENV INFERENCE_BACKEND=random

CMD ["uvicorn", "searchess_ai.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8765"]
