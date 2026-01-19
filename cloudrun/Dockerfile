# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies for Playwright/Chromium
RUN apt-get update && apt-get install -y \
    git \
    curl \
    # Playwright Chromium dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI, Playwright, and Playwright MCP globally
# Note: @playwright/test is the test runner, playwright is the library
# @playwright/mcp enables MCP-based browser automation for Claude
RUN npm install -g @anthropic-ai/claude-code @playwright/test playwright @playwright/mcp
RUN npx playwright install chromium

# Create Claude MCP configuration for Playwright
# This enables Claude Code CLI to use Playwright MCP tools for E2E testing
RUN mkdir -p /root && echo '{\
  "mcpServers": {\
    "playwright": {\
      "type": "stdio",\
      "command": "npx",\
      "args": ["@playwright/mcp", "--headless", "--browser", "chromium", "--no-sandbox"]\
    }\
  }\
}' > /root/.claude.json

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: Python Playwright shares browsers with Node.js Playwright via PLAYWRIGHT_BROWSERS_PATH
# No need to install Chromium again - it's already installed by npx playwright install

# Copy application code
COPY . .

# Create temp directory for submissions
RUN mkdir -p /tmp/submissions /tmp/evaluations

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

# Cloud Run uses PORT environment variable
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
