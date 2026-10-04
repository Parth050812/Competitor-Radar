FROM python:3.12-slim

# Install Chromium, ChromeDriver, and required system libraries
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-dev

# Copy application
COPY . .

# Tell Selenium where Docker's Chromium is
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Start FastAPI
CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${PORT}"]