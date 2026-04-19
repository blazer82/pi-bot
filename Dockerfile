FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && apt-get install -y \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --depth 1 https://github.com/rhasspy/piper && \
    cd piper/src/python && \
    pip install -e ".[train]"

COPY voice_trainer/requirements-voice-trainer.txt .
RUN pip install -r requirements-voice-trainer.txt

COPY . /workspace/pi-bot

WORKDIR /workspace/pi-bot

ENV NUMBA_CACHE_DIR=/tmp/.numba_cache
