# Contributing

Issues and pull requests are welcome. For larger changes, open an issue first to
agree on the approach.

## Development setup

Use Python 3.11 or newer from the project root:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

Tests should be deterministic and should not require an OpenAI API key. Keep
secrets in `.env` or your local environment; never commit credentials, runtime
indexes, or evaluation results.

## Pull requests

- Keep changes focused and explain the user-visible impact.
- Add or update tests and documentation when behavior changes.
- Run the offline test suite before opening a pull request.
- Include relevant evaluation results when changing retrieval or answer quality.
