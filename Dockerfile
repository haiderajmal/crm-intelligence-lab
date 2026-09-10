FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN mkdir -p /app/artifacts
EXPOSE 8000
CMD ["crm-lab", "serve", "--host", "0.0.0.0", "--port", "8000", "--database", "/app/artifacts/crm_lab.sqlite3"]
