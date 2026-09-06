import asyncio
import pytest
from unittest.mock import AsyncMock

from bolt.core.resilience import CircuitState, CircuitBreakerOpenException

async def test_success_path_closes_circuit(fast_circuit_breaker):
    """A standard successful call returns immediately without retries."""
    mock_func = AsyncMock(return_value="OK")
    
    result = await fast_circuit_breaker.call(mock_func)
    
    assert result == "OK"
    assert fast_circuit_breaker.state == CircuitState.CLOSED
    assert mock_func.call_count == 1
    assert fast_circuit_breaker.failures == 0

async def test_retry_loop_handles_transient_failures(fast_circuit_breaker):
    """Fails twice (e.g., 429 Rate Limit), but succeeds on the 3rd attempt."""
    mock_func = AsyncMock(side_effect=[ValueError("Fail 1"), ValueError("Fail 2"), "OK"])
    
    result = await fast_circuit_breaker.call(mock_func)
    
    assert result == "OK"
    assert mock_func.call_count == 3  # Initial + 2 retries
    assert fast_circuit_breaker.state == CircuitState.CLOSED
    assert fast_circuit_breaker.failures == 0  # No hard failure recorded

async def test_exhausted_retries_records_hard_failure(fast_circuit_breaker):
    """If all retries fail, it throws the exception and records 1 hard failure."""
    mock_func = AsyncMock(side_effect=ValueError("API Down"))
    
    with pytest.raises(ValueError, match="API Down"):
        await fast_circuit_breaker.call(mock_func)
        
    # max_retries = 2, so it called the API 3 times total
    assert mock_func.call_count == 3
    # BUT it only counts as 1 total failure against the Circuit Breaker threshold
    assert fast_circuit_breaker.failures == 1
    assert fast_circuit_breaker.state == CircuitState.CLOSED

async def test_circuit_trips_to_open_and_fails_fast(fast_circuit_breaker):
    """Exceeding the failure threshold trips the breaker and blocks future calls."""
    mock_func = AsyncMock(side_effect=ValueError("API Down"))
    
    # Trip the breaker by hitting the failure_threshold (3)
    for _ in range(3):
        with pytest.raises(ValueError):
            await fast_circuit_breaker.call(mock_func)
            
    assert fast_circuit_breaker.failures == 3
    assert fast_circuit_breaker.state == CircuitState.OPEN
    
    # NEXT CALL: Should raise CircuitBreakerOpenException immediately
    call_count_before = mock_func.call_count
    with pytest.raises(CircuitBreakerOpenException, match="Circuit is OPEN"):
        await fast_circuit_breaker.call(mock_func)
        
    # Verify the mock function was NEVER invoked on the blocked call
    assert mock_func.call_count == call_count_before

async def test_circuit_recovers_via_half_open_state(fast_circuit_breaker):
    """Tests the background task transitioning OPEN -> HALF_OPEN -> CLOSED."""
    mock_func = AsyncMock(side_effect=ValueError("API Down"))
    
    # 1. Trip the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            await fast_circuit_breaker.call(mock_func)
            
    assert fast_circuit_breaker.state == CircuitState.OPEN
    
    # 2. Wait for the background recovery task to fire (configured to 100ms)
    await asyncio.sleep(0.15)
    
    # 3. Assert it transitioned to testing mode
    assert fast_circuit_breaker.state == CircuitState.HALF_OPEN
    
    # 4. Issue a successful call
    mock_func.side_effect = ["Recovery OK"]
    result = await fast_circuit_breaker.call(mock_func)
    
    # 5. Assert the circuit healed itself
    assert result == "Recovery OK"
    assert fast_circuit_breaker.state == CircuitState.CLOSED
    assert fast_circuit_breaker.failures == 0

async def test_stream_call_generator_wrapping(fast_circuit_breaker):
    """Verifies that async generators can be safely wrapped and yielded."""
    async def mock_stream():
        yield "Chunk 1"
        yield "Chunk 2"
        
    results = []
    async for chunk in fast_circuit_breaker.stream_call(mock_stream):
        results.append(chunk)
        
    assert results == ["Chunk 1", "Chunk 2"]
    assert fast_circuit_breaker.state == CircuitState.CLOSED