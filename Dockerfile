FROM python:3.12-slim

# Unprivileged runtime user with a fixed UID/GID so bind-mount permissions are predictable.
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --system --gid "${APP_GID}" fieldguard \
    && useradd --system --uid "${APP_UID}" --gid "${APP_GID}" --create-home \
       --home-dir /home/fieldguard --shell /usr/sbin/nologin fieldguard

WORKDIR /app
RUN pip install --no-cache-dir uv==0.12.19
COPY pyproject.toml uv.lock README.md LICENSE NOTICE .python-version ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra all \
    && mkdir -p /work \
    && chown -R fieldguard:fieldguard /app /work

ENV HOME=/home/fieldguard \
    MPLCONFIGDIR=/home/fieldguard/.cache/matplotlib \
    XDG_CACHE_HOME=/home/fieldguard/.cache \
    PYTHONDONTWRITEBYTECODE=1

# Outputs (default fieldguard_output/...) are written relative to /work; mount a volume here to keep them.
WORKDIR /work
VOLUME ["/work"]
USER fieldguard:fieldguard
ENTRYPOINT ["/app/.venv/bin/fieldguard"]
CMD ["--help"]
