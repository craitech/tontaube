FROM vllm/vllm-openai:v0.16.0

ENV VOICE_PATH=/srv/voices
ENV TTS_HOST=0.0.0.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    PYTHONPATH=/srv \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    REQUIRE_API_KEY=0 \
    ENABLE_VERBALIZATION=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY pyproject.toml uv.lock requirements.txt requirements-pip.lock /srv/
RUN python3 -m pip install --no-cache-dir --no-deps --require-hashes \
        -r /srv/requirements-pip.lock

COPY README.md LICENSE NOTICE THIRD_PARTY_NOTICES /srv/
COPY app /srv/app
COPY VibeVoice /srv/VibeVoice
COPY voices /srv/voices
RUN pip install --no-cache-dir --no-deps --no-build-isolation /srv
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
