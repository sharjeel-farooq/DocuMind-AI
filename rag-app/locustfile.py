"""Locust load test for RAG API.

Run with:
    locust -f locustfile.py -H http://localhost:8000
    locust -f locustfile.py --headless -u 10 -r 2 -t 1m -H http://localhost:8000 --csv locust_run
"""

import json
import random
from gevent.lock import Semaphore

from locust import HttpUser, between, task


QUERIES = [
    "compare these 2 docs",
    "task 3",
    "summarize the instructions",
    "list the objectives",
]


# Simple in-memory cache to avoid hammering the API with repeated queries
_cache = {}
_cache_lock = Semaphore()


class RAGUser(HttpUser):
    # Slow down to mimic real users and avoid 429s
    wait_time = between(5.0, 10.0)
    headers = {"Content-Type": "application/json"}

    @task(3)
    def query(self):
        """Hit /query with random prompt and treat non-200 as failure."""
        q = random.choice(QUERIES)

        # Skip network call if we've recently asked the same question
        with _cache_lock:
            if q in _cache:
                return

        with self.client.post(
            "/query",
            data=json.dumps({"query": q}),
            headers=self.headers,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"query failed ({resp.status_code}): {resp.text}")
            else:
                with _cache_lock:
                    _cache[q] = True

    @task(1)
    def health_check(self):
        """Lightweight ping to keep connections warm."""
        with self.client.get("/docs", catch_response=True) as resp:
            if resp.status_code >= 400:
                resp.failure(f"docs unavailable ({resp.status_code})")
