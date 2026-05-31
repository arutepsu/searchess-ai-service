FROM python:3.12-slim

WORKDIR /app

# Copy package metadata before source so the torch layer is cached independently.
COPY pyproject.toml README.md ./

# Install the serving extras (python-chess, numpy) from PyPI and torch CPU-only wheel
# before copying source so this heavy layer is cached across source-only changes.
# torch is installed from the pytorch CPU index to avoid the 2 GB CUDA wheel on PyPI.
# numpy and python-chess are NOT bundled in the CPU torch wheel, so they are listed
# explicitly; they match the [project.optional-dependencies.serving] set in pyproject.toml.
RUN pip install --no-cache-dir "python-chess>=1.999" "numpy>=1.24" \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install OpenTelemetry packages in a separate layer before copying source so
# they are cached independently of application code changes.
RUN pip install --no-cache-dir \
    "opentelemetry-sdk>=1.20.0" \
    "opentelemetry-exporter-otlp-proto-grpc>=1.20.0" \
    "opentelemetry-instrumentation-fastapi>=0.41b0"

COPY src/ src/

# Install the package (fastapi, uvicorn) and its declared dependencies.
# torch is already present; pip resolves it from the local cache.
RUN pip install --no-cache-dir .

EXPOSE 8765

# INFERENCE_BACKEND selects the move-selection engine at startup:
#   "fake"       — always picks legalMoves[0]; deterministic, safe default
#   "random"     — uniformly random selection from legal moves
#   "supervised" — trained neural-network policy; MODEL_ARTIFACT_DIR must be set
#
# When INFERENCE_BACKEND=supervised, mount the artifact directory into the container
# and set MODEL_ARTIFACT_DIR to the specific run directory inside it:
#   docker run \
#     -v /host/artifacts:/artifacts:ro \
#     -e INFERENCE_BACKEND=supervised \
#     -e MODEL_ARTIFACT_DIR=/artifacts/run_<id> \
#     searchess-ai-service
ENV INFERENCE_BACKEND=fake

CMD ["uvicorn", "searchess_ai.main:app", "--host", "0.0.0.0", "--port", "8765"]
