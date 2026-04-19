FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && apt-get install -y \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone --depth 1 https://github.com/rhasspy/piper && \
    cd piper/src/python && \
    pip install -e ".[train]" && \
    cd piper_train/vits/monotonic_align && \
    mkdir -p monotonic_align && \
    cythonize -i core.pyx && \
    mv core*.so monotonic_align/

RUN pip uninstall -y torchtext && \
    pip install torchmetrics==0.11.4

# Patch cuFFT bug: torch.stft crashes on Ada Lovelace GPUs (RTX 40xx) with
# the PyTorch build in this base image. Run the STFT on CPU as a workaround.
RUN MEL=/opt/piper/src/python/piper_train/vits/mel_processing.py && \
    sed -i 's/torch\.stft(y,/torch.stft(y.cpu(),/' "$MEL" && \
    sed -i 's/window=hann_window\[wnsize_dtype_device\],/window=hann_window[wnsize_dtype_device].cpu(),/' "$MEL" && \
    sed -i '/return_complex=True,/{n;s/)/)).to(y.device)/}' "$MEL"

WORKDIR /workspace

ENV NUMBA_CACHE_DIR=/tmp/.numba_cache

CMD ["sleep", "infinity"]
