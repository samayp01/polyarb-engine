FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY data/ ./data/

# Install dependencies
RUN uv sync --frozen

# Run application
CMD ["uv", "run", "python", "-m", "topic.run"]
