# deeprepo middleman: a small mutable blob/page store for reporam (or both tiers).
# stdlib only, no build deps. Deploy on any container host with a persistent volume at /data.
FROM python:3.12-slim
WORKDIR /app
COPY deepcore ./deepcore
COPY ram16.py ./ram16.py
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8080
CMD ["python", "-m", "deepcore.middle", "serve", "--root", "/data", "--port", "8080"]
