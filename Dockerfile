FROM python:3.12-slim

WORKDIR /app

COPY . .
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    . $HOME/.local/bin/env && \
    uv sync

CMD ["ls"]