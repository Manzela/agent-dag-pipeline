# Contributing to Agent DAG Pipeline

## Quick Start

```bash
git clone https://github.com/Manzela/agent-dag-pipeline.git
cd agent-dag-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Development Workflow

1. Create a feature branch from `main`
2. Make changes following the Gate-Agent pattern
3. Add tests for new gates and agents
4. Run the full verification suite:
   ```bash
   pytest tests/ -v              # All tests pass
   ruff check agent_dag/ tests/  # Zero lint violations
   mypy agent_dag/               # Type coverage
   ```
5. Submit a pull request

## Code Standards

- **Type Safety**: All models use `frozen=True` Pydantic configs
- **Fail-Closed**: Every new node must handle errors via `record_failure()`
- **Gate-Agent Pattern**: Every node must implement a deterministic gate before any LLM call
- **NumPy Docstrings**: All public functions use NumPy-style documentation
- **No Proprietary Content**: Zero tolerance for company names, client data, or API keys
