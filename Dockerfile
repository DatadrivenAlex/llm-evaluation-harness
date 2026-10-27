FROM python:3.12-slim

# Local demo image. The process stays root so a mounted ./reports directory is writable.

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY datasets ./datasets

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

ENTRYPOINT ["python", "-m", "llm_eval"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
