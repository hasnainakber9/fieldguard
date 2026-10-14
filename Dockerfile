FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv==0.12.19
COPY pyproject.toml uv.lock README.md LICENSE NOTICE .python-version ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra all
ENTRYPOINT ["./.venv/bin/fieldguard"]
CMD ["--help"]
