# FreeModel Router — the best free model for every request

One long-running server (Python 3.13, standard library only) that serves the
OpenAI Chat Completions API and the Anthropic Messages API over a directory
of free-tier model providers, in JSON and in server-sent events.

It reads a brief (`$MAKEFUN_PROMPT_FILE`: provider directory, quality table,
workload manifest), probes each provider's catalog route in the provider's
own wire (OpenAI, Anthropic, Gemini), builds one catalog, registers with the
environment (`POST $MAKEFUN_ENV_URL/register`), and answers the 20 requests
the environment's client sends — then exits 0 when the workload is done.

## Routing

For each request, the free model with the highest `rating_milli` that meets
the request's requirements (tools, vision, JSON mode, minimum context).
A JSON-mode request only goes to a wire that can echo `json_mode` (OpenAI or
Gemini — an Anthropic-wire provider always echoes it false). When a provider
refuses (402, 429, 5xx, unreachable), the request is retried on the next-best
free model that meets it; exactly one provider call may succeed per request.
Ties break toward the provider called least so far, so load spreads across
providers instead of burning one provider's free-tier budget. A paid model is
never a legal pick and is never chosen.

Every translation preserves what the provider echoes back — system text,
user text (images included), tool names, `max_tokens`, stop sequences and
JSON mode — and every answer is translated back into the wire the client
used, with the provider's text whole, its usage counts as that wire reports
them, and its stop reason mapped (`stop`/`end_turn`/`STOP`,
`length`/`max_tokens`, `tool_calls`/`tool_use`, `content_filter`/`refusal`).

## Run

```sh
python3 main.py            # serve: needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE
python3 main.py --list     # print the catalog and exit 0
```

`--list` prints the catalog document the server registers, byte for byte.
The server listens on `0.0.0.0` at an ephemeral port, polls
`GET $MAKEFUN_ENV_URL/work` every 100 ms, and exits 0 when the workload
is done. No wall clock and no unseeded randomness anywhere in the routing,
so a replay leaves the same state export.

## Layout

- `main.py` — the whole server: brief parsing, catalog probe, routing,
  the three upstream wires, the two client wires, register + poll.
- `LICENSE` — MIT.
