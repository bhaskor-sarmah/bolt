import asyncio
import logging
import random
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
    def __init__(self, 
        failure_threshold: int = 3, 
        recovery_timeout: float = 30.0,
        max_retries: int = 3,         # How many times to retry before tripping
        base_delay: float = 1.0,      # Starting delay in seconds
        max_delay: float = 10.0       # Maximum allowed delay between retries
        ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.state = CircuitState.CLOSED
        self.failures = 0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Standard async call with Backoff, Jitter, and Circuit Breaking."""
        attempt = 0
        
        while attempt <= self.max_retries:
            async with self._lock:
                if self.state == CircuitState.OPEN:
                    raise CircuitBreakerOpenException("Circuit is OPEN. Failing fast.")

            try:
                result = await func(*args, **kwargs)
                await self._record_success()
                return result
                
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    await self._record_failure()
                    raise e
                    
                await self._sleep_with_jitter(attempt)

    async def stream_call(
        self, 
        func: Callable[..., AsyncGenerator[Any, None]], 
        *args, **kwargs
    ) -> AsyncGenerator[Any, None]:
        """Streaming async call with Backoff, Jitter, and Circuit Breaking."""
        attempt = 0
        
        while attempt <= self.max_retries:
            async with self._lock:
                if self.state == CircuitState.OPEN:
                    raise CircuitBreakerOpenException("Circuit is OPEN. Failing fast.")

            try:
                # In streaming, the API usually throws the 429 exactly here, 
                # when the connection is first established.
                async for chunk in func(*args, **kwargs):
                    yield chunk
                
                await self._record_success()
                return
                
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    await self._record_failure()
                    raise e
                    
                await self._sleep_with_jitter(attempt)

    async def _sleep_with_jitter(self, attempt: int):
        """Calculates Exponential Backoff with 'Full Jitter' and sleeps."""
        # Exponential backoff: 1s, 2s, 4s, 8s... capped by max_delay
        delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        
        # Full Jitter: pick a random float between 0 and the delay
        jittered_delay = random.uniform(0, delay)
        
        logger.warning(f"API failed. Retrying attempt {attempt}/{self.max_retries} in {jittered_delay:.2f}s...")
        await asyncio.sleep(jittered_delay)

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