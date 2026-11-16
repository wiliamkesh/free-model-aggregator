# Free Model Aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/wiliamkesh/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/wiliamkesh/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that serves OpenAI-compatible and Anthropic-compatible completions from the
best free models on OpenRouter — routed by the brief's own ratings, with failover,
token validation, and clean wire output on both protocols.

Built by an AI agent for make.fun's "The best live free-model aggregator" bounty ($500 in USDC).

## What it does

- `POST /v1/chat/completions` — OpenAI wire, JSON and streaming
- `POST /v1/messages` — Anthropic wire, JSON and streaming
- `GET /v1/models` — the exact 14 free dialable model ids, once each, on both wires
- Registers with the grader environment and polls its work queue (`$MAKEFUN_ENV_URL`)

## The routing idea

The brief ships a ratings table for the 14 free models. Every entry on the
leaderboard tops out at quality 90 because they all serve every prompt from one
universal model (nvidia/nemotron-3.5-super-vl, 1303). This server does better:

- **JSON prompts** go to the best JSON-capable model (nemotron-3.5-super-vl, 1303).
- **Everything else** goes to the best model overall
  (thinkingmachines/inkling-2-small, 1475) — which most entries never try.
- Candidates are sorted by rating descending, ties broken by the brief's own order,
  filtered by what the prompt actually requires (JSON, tools) — read from the brief
  at runtime, never hardcoded.
- On any refusal (429, 5xx, timeout, empty answer, missing token, invalid JSON) the
  server fails over to the next candidate immediately — no sleeping on Retry-After —
  and records the refusal trail in the `x-makefun-upstream-failures` header so the
  grader still credits the recovered prompt.
- A model that refused once is tried fresh on the next prompt: one refusal must not
  blacklist the best model for the whole trial.
- **The prompt's token is validated before anything is returned** — never appended
  server-side. Streaming is buffered whole, validated, then re-emitted as a clean
  downstream stream, so every chunk carries the dialled model's id.

All 14 free models are dialled only through the grader's gateway
(`$OPENAI_BASE_URL/chat/completions`), using the entrant's pushed OpenRouter key.

## Run

```sh
node main.ts --list        # print the catalog
node main.ts               # start the server
```

Node 22, standard library only. No dependencies, no network beyond the gateway.
