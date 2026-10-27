# P4 — Enterprise Customer Support Intent Router

## Overview
A customer-support intent-routing prototype that classifies tickets by
category and priority and generates a concise summary and draft response.

## Features
- Structured output validated with Pydantic.
- Zero-shot, few-shot, and reasoning-summary prompt variants for API mode.
- OpenAI API wrapper with retry handling for transient failures.
- Explicit mock mode for offline development and testing.
- CSV evaluation reports and benchmark artifacts.
- Unit tests for core classification and schema behavior.

## Categories
`billing`, `technical_api`, `feature_request`, `account_access`, `other`

## Priorities
`low`, `medium`, `high`, `urgent`

## Project files
- `support_router.py` — standalone mock/API router.
- `support_tickets.csv` — 12 labeled demonstration tickets.
- `output_schema.json` — output schema artifact.
- `evaluate_mock.py` — repeatable local evaluation script.
- `router_evaluation_mock.csv` — per-ticket mock predictions.
- `router_evaluation_mock_summary.json` — mock evaluation metrics.
- `prompt_benchmark_details.csv` — prompt benchmark pipeline output.
- `prompt_benchmark_matrix.csv` — aggregated prompt benchmark output.
- `mock_classification_results.csv` — initial mock classification output.
- `api_test_status.json` — records the API quota limitation.
- `tests/` — automated tests.

## Run locally
Install dependencies:

```bash
pip install pydantic pandas openai
```

Run the local mock router:

```bash
python support_router.py
```

Run the mock evaluation:

```bash
python evaluate_mock.py
```

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

## Real API mode
Configure `OPENAI_API_KEY` using an environment variable or a secret manager.
Do not commit API keys to source control.

The API mode uses `gpt-4.1-mini` by default. API availability, model access,
and billing/credits are controlled by the OpenAI API account. The current
development run encountered `insufficient_quota`, so real LLM benchmark
metrics have not been obtained.

## Evaluation limitations
The current 12-ticket dataset is a small demonstration dataset. The mock
classifier is deterministic keyword-based logic, not an LLM. Its rules were
adjusted after reviewing these same labeled examples, so the reported 100%
category and priority accuracy are not independent test results and should
not be interpreted as production performance.

The mock prompt benchmark validates the benchmark pipeline only. Because
the mock classifier does not consume prompt variants, it cannot establish
whether zero-shot, few-shot, or reasoning-summary prompting improves LLM
accuracy or latency.

## Security notes
- Never print, commit, or share API keys.
- Treat customer messages as untrusted input.
- Review generated draft responses before sending them to customers.
- Do not claim a payment, refund, or account change has occurred unless
  an authorized system confirms it.
