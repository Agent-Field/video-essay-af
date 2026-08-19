FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ffmpeg/ffprobe for narration measurement, clip normalisation and the final cut;
# Montserrat (captions) with DejaVu as the fallback face; curl/bash for the
# AgentField installer below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fonts-montserrat fonts-dejavu-core \
        curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

# AForge — AgentField's native coding harness — is what `app.harness()` runs
# for the finishing-editor slot (assemble_video). The official installer drops
# `af` and the pinned `aforge` binary into $AGENTFIELD_HOME/bin.
RUN curl -sSf https://agentfield.ai/get | AGENTFIELD_HOME=/opt/agentfield bash \
    && ln -sf /opt/agentfield/bin/aforge /usr/local/bin/aforge \
    && aforge version
ENV AFORGE_BIN=/opt/agentfield/bin/aforge \
    AFORGE_HOME=/opt/agentfield/aforge-home \
    PATH="/opt/agentfield/bin:${PATH}"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

EXPOSE 8001

CMD ["python", "main.py"]
