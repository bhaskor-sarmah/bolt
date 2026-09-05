# Bolt AI CLI ⚡️

An industry-grade, autonomous AI command-line assistant built with **Hexagonal Architecture**.

Unlike standard LLM wrappers, Bolt is designed as a resilient distributed system. It features decoupled model providers, token-aware memory compaction, Human-in-the-Loop (HITL) safety intercepts, and standard protocol (MCP) integrations.

## 🏗 Architecture

Bolt is built using **Ports and Adapters** (Hexagonal Architecture) to ensure the core business logic remains completely isolated from volatile third-party SDKs and vendor APIs.

- **Core Domain (`src/bolt/core/`):** Contains the ReAct Finite State Machine, token budgeter, and safety policies. Zero external AI framework dependencies.
- **Ports (`src/bolt/ports/`):** Abstract interfaces for LLM Drivers, Storage, and Tool execution.
- **Adapters (`src/bolt/adapters/`):** Concrete implementations (OpenAI, Anthropic, SQLite) that plug into the core.
- **UI (`src/bolt/cli/`):** The Typer/Rich terminal interface.

## 🚀 Quick Start (Development)

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management and packaging.

### 1. Prerequisites

Ensure you have `uv` and Python 3.11+ installed.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Installation

Clone the repository and install the project in editable mode (`-e`). This creates a symlink, meaning your code changes take effect instantly without needing to reinstall.

```bash
# Clone the repository
git clone https://github.com/yourusername/bolt.git
cd bolt

# Install in editable mode
uv pip install -e .
```

### 3. Verify Installation

Run the root command to ensure the CLI is routing correctly:

```bash
bolt
```

_(If your virtual environment isn't activated, you can always run `uv run bolt`)_

## 📂 Directory Structure (src layout)

```text
bolt/
├── pyproject.toml              # Build config and CLI entry point
├── tests/                      # Unit and integration tests
└── src/
    └── bolt/
        ├── cli/                # [Driving Adapter] Typer app & REPL
        ├── config/             # Environment & path management
        ├── core/               # [Domain] Schemas, FSM, memory, budgeting
        ├── ports/              # [Interfaces] ModelDriver, Storage
        └── adapters/           # [Driven Adapters] OpenAI, MCP, SQLite
```

## 🗺 Roadmap Milestones

- [ ] **Phase 1:** ModelDriver interface, multi-provider adapters, and circuit breakers.
- [ ] **Phase 2:** ReAct FSM state machine and HITL security intercepts.
- [ ] **Phase 3:** Token budgeting, compaction buffer, and SQLite WAL persistence.
- [ ] **Phase 4:** Model Context Protocol (MCP) client and sandboxed subprocesses.
- [ ] **Phase 5:** DAG task planner and context-isolated subagents.
- [ ] **Phase 6:** OpenTelemetry tracing and deterministic eval harness.

## 🛠 Development Workflow

Because the package is installed in editable mode, your workflow is simply:

1. Edit code in `src/`.
2. Run `bolt <command>` to test.

If you add new dependencies to `pyproject.toml`, sync your environment:

```bash
uv sync
```
