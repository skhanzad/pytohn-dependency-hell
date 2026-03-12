FROM python:3.11

ARG UNAME=appuser
ARG UID=1000
ARG GID=1000
ARG DOCKER_GID=999

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UNAME=$UNAME

# Match PLLM's Docker-aware container pattern.
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

RUN groupadd -g $GID -o $UNAME && \
    useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

RUN mkdir -p /EPLLM && chown $UID:$GID /EPLLM

COPY tools/EPLLM/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt

COPY tools/EPLLM/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY tools/EPLLM/ /EPLLM/
COPY hard-gists/ /EPLLM/hard-gists/
RUN chown -R $UID:$GID /EPLLM

WORKDIR /EPLLM

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
