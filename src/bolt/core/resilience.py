import asyncio
import logging
from enum import Enum
from typing import AsyncGenerator, Callable, Any, Awaitable

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation, API is healthy
    OPEN = "open"           # API is down, fail fast
    HALF_OPEN = "half_open" # Testing if the API is back online

class CircuitBreakerOpenException(Exception):
    """Raised when the circuit is open to prevent network calls."""
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.state = CircuitState.CLOSED
        self.failures = 0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Wraps an async function call with Circuit Breaker logic."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenException("Circuit is OPEN. Failing fast.")

        try:
            # Execute the actual API call
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
            
        except Exception as e:
            await self._record_failure()
            raise e

    async def stream_call(self, func: Callable[..., AsyncGenerator[Any, None]], *args, **kwargs) -> AsyncGenerator[Any, None]:
        """Wraps an async generator with Circuit Breaker logic."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenException("Circuit is OPEN. Failing fast to protect system.")
            
        try:
            # Iterate over the underlying stream
            async for chunk in func(*args, **kwargs):
                yield chunk
            
            # If the stream completes without raising an exception, it succeeded
            await self._record_success()
            
        except Exception as e:
            # If the stream drops halfway through or fails to connect, trip the circuit
            await self._record_failure()
            raise e

    async def _record_failure(self):
        async with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold and self.state == CircuitState.CLOSED:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit tripped OPEN after {self.failures} failures.")
                # Schedule a background task to transition to HALF_OPEN
                asyncio.create_task(self._attempt_reset())

    async def _record_success(self):
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info("Circuit recovered. Transitioning to CLOSED.")
            self.failures = 0
            self.state = CircuitState.CLOSED

    async def _attempt_reset(self):
        """Waits for the timeout, then allows 1 test request through."""
        await asyncio.sleep(self.recovery_timeout)
        async with self._lock:
            self.state = CircuitState.HALF_OPEN
            logger.info("Recovery timeout reached. Circuit transitioning to HALF_OPEN.")