# Deployment notes — FastAPI

This project now exposes a FastAPI application in `app.py` and includes a top-level `app` variable for Python ASGI deployments. This makes the repository compatible with Vercel's Python runtime and standard container-based deployment.

1) Local development with Uvicorn

Run the API locally from the project root:

```bash
uvicorn app:app --host 0.0.0.0 --port 8501
```

Then open http://localhost:8501 in your browser. The FastAPI interactive docs are available at http://localhost:8501/docs.

2) Docker (portable, reproducible)

Build the Docker image from the project root and run locally:

```bash
docker build -t ethiopia-electricity-analysis .
docker run -p 8501:8501 ethiopia-electricity-analysis
```

Then open http://localhost:8501 in your browser.

Notes:
- The included `Dockerfile` uses `python:3.11-slim` and installs `requirements.txt`.
- The `Dockerfile` exposes port `8501` and runs:
  `uvicorn app:app --host 0.0.0.0 --port 8501`.
- A `.dockerignore` is provided to avoid copying datasets, notebooks, and other large files into the image.

3) Vercel deployment

The FastAPI app is now compatible with Vercel's Python runtime because `app.py` exports a top-level `app` ASGI application object. A minimal `vercel.json` is included in the repository root to route all requests to `app.py`.

If you are deploying from the CLI, run:

```bash
vercel --prod
```

4) Container registry publishing

A GitHub Actions workflow is included that builds the Docker image and pushes it to GitHub Container Registry. It optionally publishes to Docker Hub when `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are configured.

If you'd like, I can also add a `vercel.json` file for explicit Python routing or a GitHub Actions workflow that deploys to Vercel automatically.
