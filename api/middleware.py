"""Middleware que mede a latência de cada requisição."""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LatencyMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request, call_next):

        logger.info(f"[MIDDLEWARE] {request.method} {request.url.path}")

        start = time.time()

        response = await call_next(request)

        latency = round(time.time() - start, 4)

        logger.info(f"[LATENCY] {latency}s")

        return response
