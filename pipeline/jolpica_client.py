"""
A small, polite HTTP client for the Jolpica-F1 API.

Rules it follows (from the official rate limit page):
  - burst limit: 4 requests per second   -> we wait at least 0.5 s between requests
  - sustained limit: 500 requests / hour -> we stop at 400 per hour to keep a safety margin
Temporary failures (HTTP 429, HTTP 5xx, network errors) are retried with
increasing waits. Other errors (such as 404) stop immediately.
"""

import logging
import time
from collections import deque

import httpx

BASE_URL = "https://api.jolpi.ca/ergast/f1"
USER_AGENT = "f1-intelligence-portfolio/0.1 (github.com/Narasimhaav2006/f1-intelligence)"
PAGE_LIMIT = 100                      # the API's maximum page size
MIN_SECONDS_BETWEEN_REQUESTS = 0.5
HOURLY_BUDGET = 400
MAX_ATTEMPTS = 5

log = logging.getLogger("jolpica")


class JolpicaError(Exception):
    """Raised when the API cannot give us usable data."""


class JolpicaClient:
    def __init__(self):
        self._http = httpx.Client(
            base_url=BASE_URL,
            timeout=20.0,
            headers={"User-Agent": USER_AGENT},
        )
        self._last_request_at = 0.0
        self._recent = deque()  # times of requests made in the last hour

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._http.close()

    def _wait_for_turn(self):
        """Sleep as needed so we stay inside both rate limits."""
        now = time.monotonic()
        while self._recent and now - self._recent[0] > 3600:
            self._recent.popleft()

        if len(self._recent) >= HOURLY_BUDGET:
            pause = 3600 - (now - self._recent[0]) + 1
            log.warning("Hourly budget used up. Pausing %.0f seconds.", pause)
            time.sleep(pause)

        gap = time.monotonic() - self._last_request_at
        if gap < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - gap)

        self._last_request_at = time.monotonic()
        self._recent.append(self._last_request_at)

    def get_json(self, path, params=None):
        """GET a path like '2023/1/results.json' and return the parsed JSON."""
        problem = "unknown problem"

        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._wait_for_turn()
            delay = min(2 ** attempt, 60)

            try:
                response = self._http.get(path, params=params)
            except httpx.TransportError as error:
                problem = f"network error ({type(error).__name__})"
            else:
                code = response.status_code
                if code == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        raise JolpicaError(f"Response was not valid JSON: {response.url}")
                    if "MRData" not in data:
                        raise JolpicaError(f"Unexpected response shape: {response.url}")
                    return data
                if code == 429 or code >= 500:
                    problem = f"HTTP {code}"
                    retry_after = response.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        delay = min(int(retry_after), 300)
                else:
                    raise JolpicaError(f"HTTP {code} for {response.url}")

            log.warning(
                "Attempt %d of %d failed (%s). Waiting %d seconds.",
                attempt, MAX_ATTEMPTS, problem, delay,
            )
            time.sleep(delay)

        raise JolpicaError(f"Gave up after {MAX_ATTEMPTS} attempts ({problem}): {path}")