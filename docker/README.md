# BioPipelines Docker Deployment Guide

This directory contains Docker configuration for deploying BioPipelines.

## Quick Start

### Development Environment

Start just PostgreSQL and Redis for local development:

```bash
cd docker
docker-compose -f docker-compose.dev.yml up -d
```

Then run the API locally:
```bash
export DATABASE_URL=postgresql://biopipelines:dev_password@localhost:5432/biopipelines_dev
export REDIS_URL=redis://localhost:6379/0
python -m uvicorn workflow_composer.api.app:create_app --reload --host 0.0.0.0 --port 8000
```

### Full Stack Deployment

1. Copy the environment template:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. Start all services:
   ```bash
   docker-compose up -d
   ```

3. Check status:
   ```bash
   docker-compose ps
   docker-compose logs -f api
   ```

4. (Optional) Start with monitoring:
   ```bash
   docker-compose --profile monitoring up -d
   # Access Flower at http://localhost:5555
   ```

## Services

| Service   | Port | Description                      |
|-----------|------|----------------------------------|
| api       | 8000 | FastAPI application              |
| worker    | -    | Celery worker (2 replicas)       |
| beat      | -    | Celery scheduler                 |
| postgres  | 5432 | PostgreSQL database              |
| redis     | 6379 | Cache and message broker         |
| flower    | 5555 | Celery monitoring (optional)     |

## Health Checks

- API: `curl http://localhost:8000/health`
- Readiness: `curl http://localhost:8000/ready`

## Scaling Workers

```bash
# Scale to 4 workers
docker-compose up -d --scale worker=4
```

## Volumes

- `postgres_data`: Database persistence
- `redis_data`: Redis persistence
- `workflow_outputs`: Generated workflow files
- `logs`: Application logs

## Troubleshooting

View logs:
```bash
docker-compose logs -f api
docker-compose logs -f worker
```

Restart a service:
```bash
docker-compose restart api
```

Reset everything:
```bash
docker-compose down -v
docker-compose up -d
```
