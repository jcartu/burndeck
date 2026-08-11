# Burndeck dashboard image.
#
# NVML (nvidia-ml-py) can only enumerate host GPUs when the container is run
# with the NVIDIA container toolkit -- i.e. `docker run --gpus all` or the
# `gpus: all` compose knob. Without it the app falls back to nvidia-smi and,
# ultimately, an empty GPU set. For GPU-less CI/demo hosts run with
# BURNDECK_MOCK_GPUS instead; see deploy/compose.yaml.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BURNDECK_HOST=0.0.0.0

# Non-root runtime user.
RUN useradd --create-home --uid 1000 burndeck

WORKDIR /app

# Install dependencies first for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application assets. config.toml is gitignored; the image ships the example
# copy, and operators override it with a bind mount (see deploy/compose.yaml).
COPY app.py ./
COPY static/ ./static/
COPY config.example.toml ./

EXPOSE 8801

USER burndeck

CMD ["python", "app.py"]
