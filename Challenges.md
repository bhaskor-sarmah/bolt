# LIST OF CHALLENGES ENCOUNTERED WHILE BUILDING IT AND HOW THOSE ARE OVERCOME

## Challenge: Working with Different Models

- Keeping an interface in the middle so that our app talks to the interface and we can switch models whenever required

## Challenge: Model Unreachable or HTTP Error

- Using Circuit Breaker pattern so that we don't need to wait for timeouts when things are unreachable

## Challenge: Model output format is unpredictable and sometimes it responds with its thought process along with the output and sometimes it only responds with the output. Also, the output token size varies a lot

- Using a higher token size 4096 in the dispatcher for now, but we might need to revise this approach later
- Providing the model with a Structured Outputs object and forcing it to produce output in that structure (this is obsolete nowadays). The problem is as follows:

  ````The "Thinking in JSON" Problem
  If you force a reasoning model to output its thoughts inside a strict JSON structure (e.g., {"thought_process": "...", "final_answer": "..."}), you actually break its ability to reason. It wastes compute trying to format JSON brackets instead of actually solving the problem, and its logic heavily degrades.```
  ````

- Nowadays the models return the reasoning and the actual answer in two separate streams, so we catch them separately and print them separately

## Network rate limits and transient 502 Bad Gateway errors are the most common failures in LLM integration

- Exponential Backoff. Instead of hammering the API immediately after a failure, we wait 1 second, then 2, then 4. More importantly, we add Jitter (randomness) so if 100 concurrent clients get rate-limited, they don't all retry at the exact same millisecond and accidentally DDoS the server. (the Thundering Herd problem)
