# Repository Guidelines

## Project Structure & Module Organization
- `realestate_content_transformer/`: Core package, including `data/` for pipelines, models, and archiving plus `utils/` helpers.
- `scripts/`: CLI entry points and utilities (rewriter runner, log archiving, log parsing).
- `app/`: Experimental FastAPI service entry (`app/main.py`).
- `tests/`: `unittest`-based tests and fixtures (see `tests/test_config.yaml`).
- Root files: `setup.py` for packaging and `README.md`/`CLAUDE.md` for usage details.

## Build, Test, and Development Commands
- `pip install -e .`: Install the package in editable mode for local development.
- `python scripts/run_locallogic_content_rewriter.py --prov_code='BC' --lang='en'`: Run a full province rewrite.
- `python scripts/run_locallogic_content_rewriter.py --geog_id='g30_dxbcrsms' --lang='en'`: Run a single location.
- `python scripts/run_locallogic_content_rewriter.py --config scripts/config.yaml`: Run with YAML configuration.
- `python scripts/archive_logs.py`: Archive old log files into `.tar.gz` bundles.

## Coding Style & Naming Conventions
- Python style is lightweight and consistent with existing files: 2-space indents, `snake_case` for functions/vars, `PascalCase` for classes.
- Prefer clear, direct naming that matches existing modules (e.g., `rewrite_property_types`).
- No formatter or linter is enforced; align with adjacent code in the file you edit.

## Testing Guidelines
- Tests use `unittest` in `tests/` (e.g., `tests/test_LocallogicContentRewriter.py`).
- Run tests with:
  - `python -m unittest tests/test_LocallogicContentRewriter.py`
  - `python -m unittest tests/test_archiver.py`
- Test configuration lives in `tests/test_config.yaml`.
- Set Elasticsearch env vars before running tests:
  - `export ES_HOST=localhost`
  - `export ES_PORT=9201`

## Commit & Pull Request Guidelines
- Commit messages are short and descriptive; recent history shows lowercase imperatives and occasional prefixes like `bug:`.
- For PRs, include: a concise description, any relevant config changes, and links to related issues or runs. If behavior changes touch rewriting logic or archiving, call that out explicitly.

## Configuration & Safety Notes
- YAML config is under `scripts/config.yaml` (host, port, province, language, archive path, etc.).
- The rewriter uses external dependencies (`realestate_core`, `realestate_spam`) and expects Elasticsearch connectivity; validate before long runs.
