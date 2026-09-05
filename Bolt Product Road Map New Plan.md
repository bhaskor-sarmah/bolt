```
                              HEXAGONAL CLI ARCHITECTURE

               +------------------------------------------------------+
               |                  DRIVING ADAPTERS                    |
               |   [Interactive REPL]  [Headless/CI]  [Daemon / IPC]   |
               +--------------------------+---------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                CORE DOMAIN LOGIC                                  |
|                                                                                   |
|  +---------------------+   +---------------------+   +-------------------------+  |
|  |  ReAct Agent FSM    |-->| Execution Scheduler |-->| Safety & Policy Engine  |  |
|  | (Deterministic Loop)|   |  (DAG / Subagents)  |   |    (HITL Intercepts)    |  |
|  +---------------------+   +---------------------+   +-------------------------+  |
|             ^                         ^                            ^              |
|             |                         |                            |              |
|  +---------------------+   +---------------------+   +-------------------------+  |
|  | Context Budgeter &  |   | Tool Contracts      |   | Evaluation & Metrics    |  |
|  | Hierarchical Memory |   | (Standard Schemas)  |   | Telemetry Dispatcher    |  |
|  +---------------------+   +---------------------+   +-------------------------+  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
               +------------------------------------------------------+
               |                  DRIVEN ADAPTERS                     |
               |  [Model Providers]  [MCP & Sandboxes]  [SQLite Engine]|
               +------------------------------------------------------+
```

## Phase 1: The Pluggable Core & Provider Agnosticism

**Objective:** Decouple your system from vendor APIs and third-party frameworks. Enable switching from OpenAI to Anthropic, local vLLM, or future providers without touching a single line of business logic.

### 1.1 Model Provider Port (`ModelDriver` Interface)

- **Requirements:**
  - Define an abstract base protocol (`ModelDriver`) with two primary methods: `complete(prompt, tools, schema) -> ModelResult` and `stream_complete(...) -> AsyncIterator[ModelTokenChunk]`.
  - Standardize input/output data transfer objects (`UserMessage`, `AssistantMessage`, `ToolCallRequest`, `ToolCallResponse`) using Pydantic or standard dataclasses.
  - Implement concrete adapters for Anthropic, OpenAI, Google, and Ollama/vLLM.
  - _Anticipating Change:_ Future models might output multimodal streams, structured artifacts, or internal thought chains. The `ModelTokenChunk` must be an algebraic data type (Union) capable of delivering `TextChunk`, `ThinkingChunk`, or `ToolCallChunk`.
  - _Durable Skill Gained:_ High-throughput async stream processing; clean interface boundaries for non-deterministic APIs.

### 1.2 Resilient Dispatcher & Circuit Breaker

- **Requirements:**
  - Wrap model calls in an exponential backoff retry handler with jitter for transient HTTP errors (429, 500, 502, 503).
  - Implement an in-memory Circuit Breaker: if a model fails 3 consecutive times within 60 seconds, trip the circuit and fall back to a designated secondary model automatically without crashing the user session.
  - _Anticipating Change:_ Rate limits and model outages will never go away. Provider load-balancing and fallback failover are permanent distributed systems requirements.
  - _Durable Skill Gained:_ Fault-tolerant system design, circuit breakers, and network failure mitigation.

## Phase 2: The Agent Execution Loop as a Finite State Machine

**Objective:** Replace unstructured `while True` loops with an auditable, deterministic, and interruptible state machine.

### 2.1 ReAct Finite State Machine (FSM)

- **Requirements:**
  - Model the agent execution as explicit states: `IDLE`, `PLANNING`, `AWAITING_TOOL_EXECUTION`, `EXECUTING_TOOL`, `EVALUATING_OUTPUT`, `COMPACTING_CONTEXT`, and `TERMINATED`.
  - State transitions must be atomic and triggered by typed events (e.g., `ToolCallRequested`, `StepLimitExceeded`, `UserInterrupt`).
  - Enforce hard invariant bounds: Maximum execution steps (e.g., 25 steps per prompt) and loop-detection heuristics (detecting if the agent repeatedly issues the exact same tool call with the same arguments).
  - _Anticipating Change:_ Future models may use multi-agent debate or tree-of-thought exploration. Modeling this as an FSM ensures you can branch into complex topological execution without rewriting prompt wrappers.
  - _Durable Skill Gained:_ Designing robust control loops for autonomous software; loop prevention and state invariants.

### 2.2 Human-in-the-Loop (HITL) Policy Engine

- **Requirements:**
  - Build a deterministic policy engine that inspects every tool call before invocation.
  - Categorize actions into risk tiers:
    - **Read-Only (Tier 0):** Auto-approved (e.g., reading files, searching symbols, listing directories).
    - **Mutating (Tier 1):** Configurable confirmation (e.g., writing new files, running test suites).
    - **Destructive (Tier 2):** Mandatory confirmation (e.g., shell command execution, deleting files, `git push/reset`, modifying environment keys).
  - Allow dry-run visual diffs (e.g., generating Git-style unified diffs in Rich before patching a file).
  - _Anticipating Change:_ As agents become more capable, security, auditing, and blast-radius containment become the primary hurdles to real-world enterprise adoption.
  - _Durable Skill Gained:_ Security sandboxing, zero-trust architecture, command verification, and interactive terminal intercept patterns.

### KEEP IN MIND: The HITL Protocol Boundary

When you build your Human-in-the-Loop (HITL) intercepts in Phase 2, place the approval gate directly between your execution graph and the MCP client. Rather than wiring custom confirmation logic into individual subagents, maintaining a single global SENSITIVE_TOOLS set at the protocol boundary ensures that no matter how many agents you spawn, destructive tools cannot bypass user approval.

## Phase 3: Hierarchical Memory & Context Budgeting

**Objective:** Solve the core economic and technical challenge of LLM operations: attention degradation, token limits, and running costs.

### 3.1 Token Budget Engine

- **Requirements:**
  - Integrate a native local tokenizer (e.g., `tiktoken` for OpenAI, or accurate character/subword estimations for others).
  - Enforce a strict context allocation budget per turn:
    - **System Prompt & Invariant Rules:** 15% fixed allocation.
    - **Working Memory / Scratchpad (Recent turns + tool returns):** 50% dynamic allocation.
    - **Compacted Semantic History:** 20% allocation.
    - **Output Reservation:** 15% guaranteed headroom.
  - _Anticipating Change:_ Context windows may expand from 1M to 10M tokens, but "lost-in-the-middle" attention degradation, latency, and costs scale linearly with prompt size. Active context pruning will remain a necessity.
  - _Durable Skill Gained:_ Cache eviction algorithms, resource allocation, and token economics.

### 3.2 Tiered Memory Architecture

- **Requirements:**
  - **Tier 1 (Scratchpad Memory):** Raw conversational history kept in memory for the last $N$ turns.
  - **Tier 2 (Compaction Engine):** When token limits breach 80% capacity, invoke a fast background model to compress Tier 1 into structured memory artifacts: Key decisions made, modified file paths, and known constraints.
  - **Tier 3 (Episodic Store via SQLite):** Persist all sessions, state diffs, and compacted artifacts into `~/.bolt/state.db` using SQLite with Write-Ahead Logging (WAL) enabled.
  - Support session checkpointing: Users can resume any prior session via `bolt session resume <id>` or branch a past session into a new trajectory (`bolt session fork`).
  - _Anticipating Change:_ Vector databases will evolve, but local, structured relational storage (SQLite) is timeless. When ready, you can simply add a vector column or sqlite-vss extension without breaking the persistence engine.
  - _Durable Skill Gained:_ Relational database design for semi-structured data, WAL concurrency, and session snapshotting.

## Phase 4: Universal Tooling via Open Protocols (MCP)

**Objective:** Prevent tool lock-in by implementing the open standard Model Context Protocol (MCP) instead of writing proprietary tool formats.

### 4.1 MCP Client Implementation

- **Requirements:**
  - Implement an MCP client capable of discovering tools, resources, and prompts over standard JSON-RPC 2.0 (communicating via `stdio` or Server-Sent Events).
  - Parse external MCP servers configured in `~/.bolt/mcp_servers.json` (e.g., connecting instantly to existing Postgres, GitHub, Slack, or FileSystem MCP servers).
  - Map incoming tool definitions dynamically into the native `ModelDriver` function schemas.
  - _Anticipating Change:_ The industry is converging on Anthropic's Model Context Protocol as the standard interface between LLMs and external systems. Building on MCP means your CLI automatically gains access to thousands of pre-built tools without you writing custom code.
  - _Durable Skill Gained:_ Protocol integration, RPC serialization/deserialization, inter-process communication (IPC) via `stdin`/`stdout`.

### 4.2 Sandboxed Tool Execution Environment

- **Requirements:**
  - Isolate shell executions from your host system. Provide an adapter that runs execution commands either inside a local subprocess with constrained path access, an isolated subshell, or a temporary container/sandbox.
  - Capture streaming standard out and standard error with hard truncation limits (preventing an accidental `cat massive_file.log` from dumping 100,000 tokens directly into the LLM context).
  - _Anticipating Change:_ Running autonomous code locally will always be dangerous. Sandboxing paradigms will remain foundational to production agent tooling.
  - _Durable Skill Gained:_ Subprocess process management, streams truncation, IO multiplexing, and containment security.

### KEEP IN MIND :

**Stateless MCP Evolution**
When you reach Phase 4 (MCP Integration), ensure you implement the mid-2026 specification updates. The Model Context Protocol recently moved away from protocol-level sessions and Redis-backed state stores in favor of a completely stateless execution model using standard round-robin load balancing. Additionally, the older Server-Sent Events (SSE) transport was deprecated in favor of Streamable HTTP for remote servers. Building your client to support these stateless routing headers (Mcp-Method and Mcp-Name) from day one will make your CLI radically easier to scale.

**Transport Isolation**
Start with the stdio transport for your local CLI tools, as it keeps credentials on disk and operates with zero network surface area. Never write standard print statements to stdout inside an MCP server, as the protocol uses stdout for its JSON-RPC communication channel and stray prints will corrupt your message stream.

## Phase 5: Subagent Orchestration & DAG Task Engine

**Objective:** Enable the CLI to execute multi-step engineering tasks by isolating work into distinct worker processes rather than overloading a single model context.

### 5.1 Directed Acyclic Graph (DAG) Task Planner

- **Requirements:**
  - For tasks flagged as complex, introduce a `PlannerAgent` that decomposes a goal into a formal DAG of subtasks:
    JSON
    ```
    {
      "tasks": [
        {"id": "1", "action": "inspect_schema", "deps": []},
        {"id": "2", "action": "write_migration", "deps": ["1"]},
        {"id": "3", "action": "update_models", "deps": ["1"]},
        {"id": "4", "action": "run_test_suite", "deps": ["2", "3"]}
      ]
    }
    ```
  - Execute independent nodes in parallel (tasks 2 and 3 run concurrently).
  - _Anticipating Change:_ Monolithic prompting will always hit complexity limits. Graph-based execution mirrors standard workflow orchestrators (like Airflow, Temporal, or Celery) and remains the standard for deterministic task resolution.
  - _Durable Skill Gained:_ Graph algorithms, dependency resolution, concurrency, and task scheduling.

### 5.2 Context-Isolated Subagents

- **Requirements:**
  - Spawn short-lived child agents assigned solely to a single DAG node.
  - A child agent receives only the parent's task-specific instruction and relevant tool access, not the parent's entire chat history.
  - Upon completion, the subagent returns a typed, compressed result summary to the parent and terminates, discarding its intermediate thought loops and logs.
  - _Anticipating Change:_ Subagent orchestration prevents context poisoning (where an agent gets confused by its own previous errors). As inference costs decline, running multi-agent topologies will become standard practice.
  - _Durable Skill Gained:_ Actor model primitives, process isolation, and state synthesis.

## Phase 6: Observability, Evals & Production Hardening

**Objective:** Transform the tool from a personal hobby script into an auditable, enterprise-grade system that can run in headless CI environments.

### 6.1 OpenTelemetry (OTel) Distributed Tracing

- **Requirements:**
  - Instrument the entire system using the OpenTelemetry standard.
  - Every user prompt initiates a root trace. Each model invocation, tool call, compaction run, and subagent spawn creates an individual child span.
  - Record span attributes: Provider, model name, input/output token counts, latency, and tool return exit codes.
  - Support exporting traces to standard sinks (Jaeger, Honeycomb, or local JSON logs).
  - _Anticipating Change:_ Proprietary AI tracing platforms will come and go. OpenTelemetry is the open vendor-neutral enterprise standard for telemetry.
  - _Durable Skill Gained:_ Production observability, APM instrumentation, and distributed tracing.

### 6.2 Deterministic Eval Harness (Evaluation-Driven Development)

- **Requirements:**
  - Build a test suite that evaluates the CLI against a standardized test set (e.g., 20 real-world scenarios: fixing a broken Python function, generating a valid schema, searching a repo for an elusive bug).
  - Implement dual evaluation:
    - **Deterministic Assertion:** Did the agent exit with code 0? Did the resulting code pass `pytest`? Did it modify the expected files?
    - **LLM-as-a-Judge Assertion:** Use an independent model to evaluate readability and adherence to instructions.
  - Run this harness during CI to measure regression whenever you adjust system prompts, memory algorithms, or underlying models.
  - _Anticipating Change:_ The biggest difficulty in shipping AI systems is measuring whether a prompt or code change improved or degraded performance. Eval harnesses are the unit testing equivalent of the AI era.
  - _Durable Skill Gained:_ LLM evaluation frameworks, benchmark automation, and continuous integration for non-deterministic software.

### 6.3 Headless / Daemon Interface Separation

- **Requirements:**
  - Decouple the CLI view layer (Rich/Typer) entirely from the core execution engine.
  - Implement an entry point flag: `bolt run --headless "Refactor logging"` that reads from STDIN, suppresses interactive spinners, streams JSON Lines (NDJSON) to STDOUT, and exits with standard POSIX return codes.
  - Allow the CLI to be executed directly in GitHub Actions, pre-commit hooks, or Docker containers.
  - _Anticipating Change:_ A tool that lives only in an interactive terminal cannot be embedded in automated enterprise workflows. Building headless capability enables web wrappers, IDE extensions, or CI/CD pipelines.
  - _Durable Skill Gained:_ POSIX compliance, CLI design patterns, and decoupled frontend/backend system architecture.

## The Durable Engineering Core

| **Feature Area**       | **Ephemeral Knowledge (Avoid Over-Investing)** | **Durable Systems Knowledge (Invest Deeply)**                     |
| ---------------------- | ---------------------------------------------- | ----------------------------------------------------------------- |
| **Model Integration**  | Memorizing Pydantic-AI/LangChain APIs          | Designing Hexagonal Provider Adapters & Circuit Breakers          |
| **Prompt Engineering** | Hacks to stop a specific model hallucinating   | Structuring state as an explicit Finite State Machine             |
| **Memory**             | Shoving raw transcripts into 1M context        | Token budget management, cache eviction, and compaction           |
| **Tool Execution**     | Hardcoding vendor function-calling schemas     | Implementing Model Context Protocol (MCP) & Subprocess Sandboxing |
| **Orchestration**      | Complex single-prompt mega-chains              | Directed Acyclic Graphs (DAGs) and isolated subagents             |
| **Testing**            | Manually trying queries in the terminal        | Deterministic evaluation harnesses and OpenTelemetry tracing      |
