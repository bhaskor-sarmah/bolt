import pytest
from bolt.core.resilience import CircuitBreaker

@pytest.fixture
def fast_circuit_breaker():
    """
    Provides a CircuitBreaker configured for immediate execution.
    Delays and timeouts are shrunk to milliseconds so tests run instantly.
    """
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=0.1,  
        max_retries=2,         
        base_delay=0.01,       
        max_delay=0.05
    )