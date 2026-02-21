# Repository Guidelines

## Project Structure & Module Organization
- `agents/`: core pipelines and utilities.
- `agents/d2insight_agent_sys.py`: iterative domain/concept/analysis workflow.
- `agents/d2insight_gpt4o.py` and `agents/d2insight_gpt4o_domain.py`: baseline insight generators.
- `agents/insight2dashboard_tot.py`: converts insight JSON into chart code and figures.
- `agents/chart_utils.py` and `agents/utils.py`: plotting and serialization helpers.
- `dataset/`: source CSV files used in experiments.
- `notebooks/`: experiment notebooks (`exp01`-`exp03`) and evaluation traces.
- `exp_result/`: generated outputs (JSON, charts, analysis scripts, notebooks). Add new runs under `exp_result/expNN/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -r requirements.txt`: install dependencies.
- `python -m compileall agents`: quick syntax/smoke check for agent modules.
- `python agents/insight2dashboard_tot.py <data.csv> <insight.json>`: generate `analysis.py`, `analysis_thoughts.md`, and chart files.
- `python -c "from agents.d2insight_agent_sys import run_domain_detector; print(run_domain_detector('dataset/Finance_survey_data.csv')['analysis'])"`: run the main analysis loop on a sample dataset.

## Coding Style & Naming Conventions
- Use 4-space indentation and follow PEP 8.
- Prefer `snake_case` for functions, files, and variables; use `PascalCase` for classes (for example, `NumpyEncoder`).
- Keep prompt templates and other constants uppercase (for example, `EVAL_PROMPT`).
- Add type hints on public functions and keep docstrings concise and actionable.

## Testing Guidelines
- There is no formal `tests/` suite yet; treat `compileall` plus targeted script runs as minimum validation.
- For behavior changes, run at least one dataset through the modified pipeline and store artifacts in `exp_result/expNN/`.
- For new automated tests, use `pytest` and name files `tests/test_<module>.py`.

## Commit & Pull Request Guidelines
- Recent history favors short, imperative commit subjects (for example, `Create README.md`, `reorganized`).
- Prefer `<area>: <change>` for clarity (example: `agents: improve eval loop termination`).
- PRs should include purpose, key file paths changed, commands used for validation, and output artifact paths or screenshots for chart changes.

## Security & Configuration Tips
- Keep secrets in `.env` (`OPENAI_API_KEY`) and never commit credentials.
- Avoid checking in ad-hoc local artifacts outside `exp_result/` and `notebooks/`.
