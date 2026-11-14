#!/usr/bin/env python3
"""FreeModel Router -- the best free model for every request.

One long-running server (Python 3.13, standard library only) that serves the
OpenAI Chat Completions API and the Anthropic Messages API over a directory
of simulated free-tier providers, in JSON and in server-sent events.

How a trial runs it:
  * the brief is at $MAKEFUN_PROMPT_FILE: the provider directory, the quality
    table (rating_milli and capabilities per model) and the workload manifest;
  * the env is at $MAKEFUN_ENV_URL: the ten providers live under their own
    paths (/p/<id>), plus POST /register, GET /work and GET /instance;
  * `python3 main.py --list` prints the catalog this server registers, byte
    for byte, and exits 0 without starting the server.

Routing: for each request, the free model with the highest rating_milli that
meets the request's requirements (tools, vision, JSON mode, min_context).
A JSON-mode request only goes to a wire that can echo json_mode (openai or
gemini -- an anthropic-wire provider always echoes json_mode false). When a
provider refuses (402, 429, 5xx, unreachable), the request is retried on the
next-best free model that meets it; exactly one provider call may succeed
per request. Provider calls are strictly sequential: one provider serves one
inference call at a time, and the client sends one request at a time.

Every route is a pure function of the requests and the providers' answers
seen so far: no wall clock, no unseeded randomness, so a replay leaves the
same state export. Fixed timestamps are used in every listing.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
UPSTREAM_TIMEOUT_S = 10.0
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


def string_of(v):
    return v if isinstance(v, str) else ""


def number_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


# ---------------------------------------------------------------------------
# The brief: three JSON documents found by `schema` in $MAKEFUN_PROMPT_FILE.
# ---------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _provider_of(v):
    if not is_record(v):
        return None
    pid = string_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if string_of(v.get(key)) != "":
            base = string_of(v.get(key))
            break
    return {
        "id": pid,
        "name": string_of(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": string_of(v.get("key_env")),
        "key": string_of(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = string_of(v.get("model"))
    provider = string_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) else 0,
        "context_length": number_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def read_brief(path):
    """The brief at `path`: directory, ratings table, workload manifest."""
    brief = {"providers": [], "ratings": None, "workload": None, "error": None}
    if path == "":
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = documents_in(text)
    directory = next(
        (d for d in docs if d.get("schema") == PROVIDERS_SCHEMA
         and isinstance(d.get("providers"), list)), None)
    if directory is None:
        brief["error"] = "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
        return brief
    for raw in directory["providers"]:
        provider = _provider_of(raw)
        if provider is not None:
            brief["providers"].append(provider)
    ratings_doc = next(
        (d for d in docs if d.get("schema") == RATINGS_SCHEMA
         and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        rows = [_rating_of(v) for v in ratings_doc["ratings"]]
        best = []
        if isinstance(ratings_doc.get("best"), list):
            for v in ratings_doc["best"]:
                if is_record(v) and string_of(v.get("id")) != "":
                    best.append({"id": v["id"],
                                 "rating_milli": number_or_none(v.get("rating_milli")) or 0})
        brief["ratings"] = {"ratings": [r for r in rows if r is not None], "best": best}
    workload_doc = next(
        (d for d in docs if d.get("schema") == WORKLOAD_SCHEMA
         and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        manifest = []
        for raw in workload_doc["manifest"]:
            if not is_record(raw) or string_of(raw.get("id")) == "":
                continue
            req = raw.get("requires") if is_record(raw.get("requires")) else {}
            entry = {
                "id": raw["id"],
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": {
                    "tools": req.get("tools") is True,
                    "vision": req.get("vision") is True,
                    "json": req.get("json") is True,
                },
            }
            if isinstance(req.get("min_context"), (int, float)):
                entry["requires"]["min_context"] = req["min_context"]
            manifest.append(entry)
        brief["workload"] = {"manifest": manifest}
    return brief


def fetch_instance(env_url):
    """GET /instance: the provider directory + workload manifest, or None."""
    if not env_url:
        return None
    try:
        req = urllib.request.Request(env_url + "/instance",
                                     headers={"accept": "application/json"})
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log("GET /instance failed: %s" % exc)
        return None
    if not is_record(body):
        return None
    providers = []
    doc = body.get("providers")
    if isinstance(doc, dict) and isinstance(doc.get("providers"), list):
        raw_list = doc["providers"]
    elif isinstance(body.get("providers"), list):
        raw_list = body["providers"]
    else:
        raw_list = []
    for raw in raw_list:
        provider = _provider_of(raw)
        if provider is not None:
            providers.append(provider)
    return {"providers": providers, "workload": body.get("workload")}


# ---------------------------------------------------------------------------
# The catalog probe: each provider's own catalog route, in its own wire.
# ---------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v) else None
    if not is_record(pricing):
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        mid = string_of(v.get("name")).replace("models/", "", 1)
    else:
        mid = string_of(v.get("id"))
    if mid == "":
        return None
    context = number_or_none(v.get("context_length"))
    if context is None:
        context = number_or_none(v.get("context_window"))
    if context is None:
        context = number_or_none(v.get("inputTokenLimit"))
    free = bool_or_none(v.get("free"))
    if free is None:
        free = _zero_priced(v)
    display = string_of(v.get("display_name")) or string_of(v.get("displayName")) or mid
    return {
        "id": mid,
        "display_name": display,
        "free": free,
        "context_length": context,
        "tools": bool_or_none(v.get("tools")),
        "vision": bool_or_none(v.get("vision")),
        "json": bool_or_none(v.get("json")),
    }


def _get_json(url, headers, timeout_s):
    """GET a URL and parse its JSON body; (body, error), one of them None."""
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                return None, "GET %s answered HTTP %s" % (url, resp.status)
            return json.loads(resp.read().decode("utf-8")), None
    except Exception as exc:
        return None, "GET %s failed: %s" % (url, exc)


def probe_provider(env_url, provider, timeout_s=PROBE_TIMEOUT_S):
    """GET one provider's catalog route. A failure leaves it out of the catalog."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    body, error = _get_json(url, headers, timeout_s)
    if error is not None:
        return {"provider": provider, "ok": False, "models": [], "error": error}
    if not is_record(body):
        return {"provider": provider, "ok": False, "models": [],
                "error": "GET %s answered a body that is not JSON" % url}
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    models = []
    if isinstance(listing, list):
        for raw in listing:
            model = _model_in(provider["wire"], raw)
            if model is not None:
                models.append(model)
    return {"provider": provider, "ok": True, "models": models, "error": None}


def rating_row(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def model_free(model, row):
    """The quality table's row is the sealed truth where it has one."""
    if row is not None:
        return row["free"]
    return model["free"] is True


def capabilities_of(model, row):
    return {
        "context_length": (row["context_length"] if row is not None else None)
        if row is not None and row["context_length"] is not None
        else model["context_length"],
        "tools": row["tools"] if row is not None else model["tools"],
        "vision": row["vision"] if row is not None else model["vision"],
        "json": row["json"] if row is not None else model["json"],
    }


def meets_requirements(requires, caps):
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    minimum = requires.get("min_context")
    if minimum is not None:
        if caps.get("context_length") is None or caps["context_length"] < minimum:
            return False
    return True


def build_catalog(probes, ratings):
    """The --list document and the register body: the same object, byte for byte.

    Models stay in the order the provider listed them: the scorer compares a
    provider's free-model list with the sealed catalog in that order.
    """
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in probe["models"]:
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_row(ratings, probe["provider"]["id"], model["id"])
            context = None
            if row is not None and row["context_length"] is not None:
                context = row["context_length"]
            elif model["context_length"] is not None:
                context = model["context_length"]
            models.append({"id": model["id"],
                           "free": model_free(model, row),
                           "context_length": context})
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def fallback_catalog(providers, ratings):
    """A catalog from the brief alone, when there is no env to probe: the
    quality table's free rows, in directory order, per provider."""
    out = {"providers": []}
    if ratings is None:
        return out
    for provider in providers:
        models = []
        for row in ratings["ratings"]:
            if row["provider"] == provider["id"] and row["free"]:
                models.append({"id": row["model"], "free": True,
                               "context_length": row["context_length"]})
        out["providers"].append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": models,
        })
    return out


# ---------------------------------------------------------------------------
# Routing: the best free model that meets the request, with failover.
# ---------------------------------------------------------------------------

class Router:
    """Candidates ranked by rating; ties break toward the provider called
    least so far, then directory order, then model id. Deterministic: every
    input is a request or a provider answer seen so far."""

    def __init__(self, probes, ratings):
        self.ratings = ratings
        self.calls = {}          # provider id -> upstream calls made so far
        self.candidates = []     # (provider, model, caps, rating), free only
        self.directory_index = {}
        for index, probe in enumerate(probes):
            if not probe["ok"]:
                continue
            provider = probe["provider"]
            self.directory_index[provider["id"]] = index
            self.calls.setdefault(provider["id"], 0)
            for model in probe["models"]:
                row = rating_row(ratings, provider["id"], model["id"])
                if not model_free(model, row):
                    continue
                caps = capabilities_of(model, row)
                rating = row["rating_milli"] if row is not None else 0
                self.candidates.append((provider, model, caps, rating))

    def rank(self, requires, json_mode):
        """Every free candidate meeting the requirements, best first."""
        ranked = []
        for provider, model, caps, rating in self.candidates:
            # A JSON-mode request needs a wire that echoes json_mode true;
            # an anthropic-wire provider always echoes it false.
            if json_mode and provider["wire"] == "anthropic":
                continue
            if not meets_requirements(requires, caps):
                continue
            ranked.append((
                -rating,
                self.calls.get(provider["id"], 0),
                self.directory_index.get(provider["id"], 999),
                model["id"],
                provider,
                model,
            ))
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [(item[4], item[5]) for item in ranked]

    def served_models(self):
        """Free models, deduplicated by id, in directory then listing order:
        what GET /v1/models lists."""
        seen = set()
        out = []
        for provider, model, _caps, _rating in self.candidates:
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            out.append({"id": model["id"], "provider": provider["id"],
                        "display_name": model["display_name"]})
        return out


# ---------------------------------------------------------------------------
# Wire translation. The provider echoes what it parsed (system text, user
# text, tool names, max_tokens, stop sequences, image, JSON mode), and the
# client compares that echo with what it sent: a field rewritten or dropped
# loses the request. Every translation below preserves exactly those fields.
# ---------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if is_record(block) and block.get("type") == "text" \
                    and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except Exception:
            pass
    return {}


def _number_or_zero(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _stop_reason_openai(finish):
    """An Anthropic/Gemini stop as an OpenAI finish_reason."""
    if finish in ("max_tokens", "MAX_TOKENS"):
        return "length"
    if finish in ("tool_use", "tool_calls", "function_call"):
        return "tool_calls"
    if finish in ("refusal", "content_filter", "SAFETY"):
        return "content_filter"
    return "stop"


def _stop_reason_anthropic(finish):
    """An OpenAI/Gemini stop as an Anthropic stop_reason."""
    if finish in ("length", "max_tokens", "MAX_TOKENS"):
        return "max_tokens"
    if finish in ("tool_calls", "function_call", "tool_use"):
        return "tool_use"
    if finish in ("content_filter", "refusal", "SAFETY"):
        return "refusal"
    return "end_turn"


def _image_part_to_anthropic(block):
    if not is_record(block) or block.get("type") != "image_url":
        return None
    url = block.get("image_url") if is_record(block.get("image_url")) else {}
    href = string_of(url.get("url"))
    if href.startswith("data:"):
        head, _, data = href[5:].partition(",")
        media = head.split(";")[0] or "image/png"
        return {"type": "image",
                "source": {"type": "base64", "media_type": media, "data": data}}
    if href != "":
        return {"type": "image", "source": {"type": "url", "url": href}}
    return None


def _image_block_to_openai(block):
    if not is_record(block) or block.get("type") != "image":
        return None
    source = block.get("source") if is_record(block.get("source")) else {}
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        media = string_of(source.get("media_type")) or "image/png"
        return {"type": "image_url",
                "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _image_part_to_gemini(block):
    """An OpenAI image_url part or Anthropic image block as a Gemini part."""
    if is_record(block) and block.get("type") == "image_url":
        url = block.get("image_url") if is_record(block.get("image_url")) else {}
        href = string_of(url.get("url"))
    elif is_record(block) and block.get("type") == "image":
        source = block.get("source") if is_record(block.get("source")) else {}
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = string_of(source.get("media_type")) or "image/png"
            return {"inlineData": {"mimeType": media, "data": source["data"]}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"fileData": {"fileUri": source["url"]}}
        return None
    else:
        return None
    if href.startswith("data:"):
        head, _, data = href[5:].partition(",")
        media = head.split(";")[0] or "image/png"
        return {"inlineData": {"mimeType": media, "data": data}}
    if href != "":
        return {"fileData": {"fileUri": href}}
    return None


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    system_parts = []
    messages = []
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = _text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            blocks = []
            text = _text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
            for call in calls:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({
                    "type": "tool_use",
                    "id": string_of(call.get("id")) or "toolu_unknown",
                    "name": string_of(fn.get("name")),
                    "input": _arguments_of(fn.get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": string_of(raw.get("tool_call_id")),
                    "content": content if isinstance(content, str) else _text_of(content),
                }],
            })
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if isinstance(content, list):
            blocks = []
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append({"type": "text", "text": block["text"]})
                else:
                    part = _image_part_to_anthropic(block)
                    if part is not None:
                        blocks.append(part)
            if blocks:
                messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        out["max_tokens"] = body["max_completion_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if string_of(fn.get("name")) == "":
                continue
            tools.append({
                "name": fn["name"],
                "description": string_of(fn.get("description")),
                "input_schema": fn.get("parameters")
                if is_record(fn.get("parameters"))
                else {"type": "object", "properties": {}},
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if isinstance(choice, str):
        # Anthropic's tool_choice has all four spellings; keep "none" a none.
        out["tool_choice"] = {"type": choice} if choice in ("auto", "any", "none") \
            else ({"type": "any"} if choice == "required" else {"type": "auto"})
    elif is_record(choice) and is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool",
                              "name": string_of(choice["function"].get("name"))}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = _text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user",
                             "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": string_of(block.get("id")),
                    "type": "function",
                    "function": {
                        "name": string_of(block.get("name")),
                        "arguments": json.dumps(
                            block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = _text_of(content)
            entry = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": string_of(block.get("tool_use_id")),
                    "content": block.get("content")
                    if isinstance(block.get("content"), str)
                    else _text_of(block.get("content")),
                })
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            else:
                part = _image_block_to_openai(block)
                if part is not None:
                    parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        out["stop"] = [s for s in stops if isinstance(s, str)]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or string_of(raw.get("name")) == "":
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": string_of(raw.get("description")),
                    "parameters": raw.get("input_schema")
                    if is_record(raw.get("input_schema"))
                    else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        kind = choice.get("type")
        if kind == "auto":
            out["tool_choice"] = "auto"
        elif kind == "any":
            out["tool_choice"] = "required"
        elif kind == "none":
            out["tool_choice"] = "none"
        elif kind == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function",
                                  "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _gemini_contents_from_openai(body):
    system_parts = []
    contents = []
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = _text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            parts = []
            text = _text_of(content)
            if text != "":
                parts.append({"text": text})
            calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
            for call in calls:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {
                    "name": string_of(fn.get("name")),
                    "args": _arguments_of(fn.get("arguments")),
                }})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            # A tool message is not a user message: the provider's echo of
            # the user text must stay the last user message's text, so a
            # tool result never becomes a Gemini user content of its own.
            continue
        if role != "user":
            continue
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                else:
                    part = _image_part_to_gemini(block)
                    if part is not None:
                        parts.append(part)
        if parts:
            contents.append({"role": "user", "parts": parts})
    return system_parts, contents


def _gemini_contents_from_anthropic(body):
    system_parts = []
    system = _text_of(body.get("system"))
    if system != "":
        system_parts.append(system)
    contents = []
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            if content == "":
                continue
            contents.append({"role": "model" if assistant else "user",
                             "parts": [{"text": content}]})
            continue
        if not isinstance(content, list):
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image":
                part = _image_part_to_gemini(block)
                if part is not None:
                    parts.append(part)
            elif assistant and block.get("type") == "tool_use":
                parts.append({"functionCall": {
                    "name": string_of(block.get("name")),
                    "args": block.get("input")
                    if is_record(block.get("input")) else {},
                }})
            # tool_result blocks carry no text the echo checks; skipping
            # them keeps the last user text exactly what the client sent.
        if parts:
            contents.append({"role": "model" if assistant else "user", "parts": parts})
    return system_parts, contents


def _gemini_tools_from_openai(body):
    if not isinstance(body.get("tools"), list):
        return None
    declarations = []
    for raw in body["tools"]:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if string_of(fn.get("name")) == "":
            continue
        declarations.append({
            "name": fn["name"],
            "description": string_of(fn.get("description")),
            "parameters": fn.get("parameters")
            if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


def _gemini_tools_from_anthropic(body):
    if not isinstance(body.get("tools"), list):
        return None
    declarations = []
    for raw in body["tools"]:
        if not is_record(raw) or string_of(raw.get("name")) == "":
            continue
        declarations.append({
            "name": raw["name"],
            "description": string_of(raw.get("description")),
            "parameters": raw.get("input_schema")
            if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
        })
    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


def _gemini_config(body, client_api):
    config = {}
    if client_api == "openai":
        if isinstance(body.get("max_tokens"), (int, float)):
            config["maxOutputTokens"] = body["max_tokens"]
        elif isinstance(body.get("max_completion_tokens"), (int, float)):
            config["maxOutputTokens"] = body["max_completion_tokens"]
        if isinstance(body.get("temperature"), (int, float)):
            config["temperature"] = body["temperature"]
        if isinstance(body.get("top_p"), (int, float)):
            config["topP"] = body["top_p"]
        stop = body.get("stop")
        if isinstance(stop, str):
            config["stopSequences"] = [stop]
        elif isinstance(stop, list) and stop:
            config["stopSequences"] = [s for s in stop if isinstance(s, str)]
        fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
        if fmt.get("type") in ("json_object", "json_schema"):
            config["responseMimeType"] = "application/json"
    else:
        if isinstance(body.get("max_tokens"), (int, float)):
            config["maxOutputTokens"] = body["max_tokens"]
        if isinstance(body.get("temperature"), (int, float)):
            config["temperature"] = body["temperature"]
        if isinstance(body.get("top_p"), (int, float)):
            config["topP"] = body["top_p"]
        stops = body.get("stop_sequences")
        if isinstance(stops, list) and stops:
            config["stopSequences"] = [s for s in stops if isinstance(s, str)]
    return config


def _gemini_tool_config(body, client_api):
    """Gemini's function-calling mode for a client's tool_choice, or None."""
    choice = body.get("tool_choice")
    if client_api == "openai" and isinstance(choice, str):
        if choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if choice == "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        return None
    if is_record(choice):
        kind = choice.get("type")
        if kind == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if kind == "any":
            return {"functionCallingConfig": {"mode": "ANY"}}
        if kind == "tool" and isinstance(choice.get("name"), str):
            return {"functionCallingConfig": {
                "mode": "ANY", "allowedFunctionNames": [choice["name"]]}}
    return None


def to_gemini(body, model, client_api):
    """An OpenAI or Anthropic request as a Gemini generateContent request."""
    if client_api == "openai":
        system_parts, contents = _gemini_contents_from_openai(body)
        tools = _gemini_tools_from_openai(body)
    else:
        system_parts, contents = _gemini_contents_from_anthropic(body)
        tools = _gemini_tools_from_anthropic(body)
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    config = _gemini_config(body, client_api)
    if config:
        out["generationConfig"] = config
    if tools is not None:
        out["tools"] = tools
    tool_config = _gemini_tool_config(body, client_api)
    if tool_config is not None:
        out["toolConfig"] = tool_config
    return out


# ---------------------------------------------------------------------------
# Provider answers back into the client's wire.
# ---------------------------------------------------------------------------

def openai_json_to_anthropic(body, model):
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = []
    text = message.get("content")
    if not isinstance(text, str):
        text = _text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append({
            "type": "tool_use",
            "id": string_of(call.get("id")) or "toolu_unknown",
            "name": string_of(fn.get("name")),
            "input": _arguments_of(fn.get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "id": "msg_%s" % (string_of(body.get("id")) or "aggregate"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason_anthropic(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": _number_or_zero(usage.get("prompt_tokens")),
                  "output_tokens": _number_or_zero(usage.get("completion_tokens"))},
    }


def anthropic_json_to_openai(body, model):
    content = body.get("content") if isinstance(body.get("content"), list) else []
    text_parts = []
    tool_calls = []
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": string_of(block.get("id")) or "toolu_unknown",
                "type": "function",
                "function": {
                    "name": string_of(block.get("name")),
                    "arguments": json.dumps(
                        block.get("input") if is_record(block.get("input")) else {}),
                },
            })
    message = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "id": string_of(body.get("id")) or "chatcmpl-router",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _stop_reason_openai(body.get("stop_reason")),
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": _number_or_zero(usage.get("input_tokens")),
                  "completion_tokens": _number_or_zero(usage.get("output_tokens")),
                  "total_tokens": _number_or_zero(usage.get("input_tokens"))
                  + _number_or_zero(usage.get("output_tokens"))},
    }


def _gemini_text_and_usage(body):
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    candidate = candidates[0] if candidates and is_record(candidates[0]) else {}
    content = candidate.get("content") if is_record(candidate.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(p.get("text", "") for p in parts
                   if is_record(p) and isinstance(p.get("text"), str))
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return text, _number_or_zero(usage.get("promptTokenCount")), \
        _number_or_zero(usage.get("candidatesTokenCount")), \
        candidate.get("finishReason")


def gemini_json_to_openai(body, model):
    text, prompt_tokens, completion_tokens, finish = _gemini_text_and_usage(body)
    return {
        "id": "chatcmpl-router",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": _stop_reason_openai(finish),
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def gemini_json_to_anthropic(body, model):
    text, prompt_tokens, completion_tokens, finish = _gemini_text_and_usage(body)
    return {
        "id": "msg_router",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": _stop_reason_anthropic(finish),
        "stop_sequence": None,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
    }


def openai_model_list(models):
    return {
        "object": "list",
        "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                  "owned_by": m["provider"]} for m in models],
    }


def anthropic_model_list(models):
    return {
        "data": [{"type": "model", "id": m["id"],
                  "display_name": m["display_name"],
                  "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
        "has_more": False,
    }


def openai_error(message, type_, code):
    return {"error": {"message": message, "type": type_, "code": code}}


def anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


def _sse_event(event, data):
    text = "%sdata: %s\n\n" % (("event: %s\n" % event) if event else "",
                               json.dumps(data))
    return text.encode("utf-8")


def _openai_chunk(model, delta, finish):
    chunk = {"id": "chatcmpl-router", "object": "chat.completion.chunk",
             "created": FIXED_CREATED, "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return _sse_event(None, chunk)


def _openai_usage_chunk(model, prompt_tokens, completion_tokens):
    chunk = {"id": "chatcmpl-router", "object": "chat.completion.chunk",
             "created": FIXED_CREATED, "model": model, "choices": [],
             "usage": {"prompt_tokens": prompt_tokens,
                       "completion_tokens": completion_tokens,
                       "total_tokens": prompt_tokens + completion_tokens}}
    return _sse_event(None, chunk)


class SseFrames:
    """Incremental server-sent-events frames: (event, data-string) pairs."""

    def __init__(self):
        self.buffer = b""

    def feed(self, data):
        self.buffer += data.replace(b"\r\n", b"\n")
        frames = []
        while True:
            at = self.buffer.find(b"\n\n")
            if at == -1:
                break
            frames.append(_parse_frame(self.buffer[:at]))
            self.buffer = self.buffer[at + 2:]
        return frames

    def flush(self):
        if self.buffer.strip() == b"":
            return []
        frame = _parse_frame(self.buffer)
        self.buffer = b""
        return [frame]


def _parse_frame(raw):
    event = None
    data = []
    for line in raw.decode("utf-8", "replace").split("\n"):
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    return event, "\n".join(data)


def _split_text(text, size=13):
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def anthropic_events_from_openai_stream(read_chunk, model):
    """Translate an OpenAI SSE stream into Anthropic events, yielding bytes.

    `read_chunk` is a zero-argument callable returning the next upstream
    bytes, or b"" at end of stream."""
    parser = SseFrames()
    started = False
    finished = False
    pending_stop = None
    text_index = -1
    next_index = 0
    tool_indexes = {}
    input_tokens = 0
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return
        started = True
        yield _sse_event("message_start", {
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message",
                        "role": "assistant", "model": model, "content": [],
                        "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0}},
        })

    def close_events():
        nonlocal text_index
        out = []
        if text_index != -1:
            out.append(_sse_event("content_block_stop",
                                  {"type": "content_block_stop", "index": text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(_sse_event("content_block_stop",
                                  {"type": "content_block_stop", "index": index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason):
        nonlocal finished
        if finished:
            return
        finished = True
        out = close_events()
        out.append(_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }))
        out.append(_sse_event("message_stop", {"type": "message_stop"}))
        return out

    def events_of_frame(frame):
        nonlocal pending_stop, text_index, next_index, input_tokens, output_tokens
        _event, payload = frame
        out = []
        if payload.strip() == "[DONE]":
            out.extend(start_events())
            out.extend(finish_events(pending_stop or "end_turn"))
            return out
        try:
            chunk = json.loads(payload)
        except Exception:
            return out
        if not is_record(chunk):
            return out
        if is_record(chunk.get("usage")):
            input_tokens = _number_or_zero(chunk["usage"].get("prompt_tokens"))
            output_tokens = _number_or_zero(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else None
        if choice is None:
            return out
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        out.extend(start_events())
        content = delta.get("content")
        if isinstance(content, str) and content != "":
            if text_index == -1:
                text_index = next_index
                next_index += 1
                out.append(_sse_event("content_block_start", {
                    "type": "content_block_start", "index": text_index,
                    "content_block": {"type": "text", "text": ""}}))
            out.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": text_index,
                "delta": {"type": "text_delta", "text": content}}))
        calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
        for call in calls:
            if not is_record(call):
                continue
            upstream_index = call.get("index") if isinstance(call.get("index"), int) \
                else len(tool_indexes)
            index = tool_indexes.get(upstream_index)
            if index is None:
                if text_index != -1:
                    out.append(_sse_event("content_block_stop",
                                          {"type": "content_block_stop",
                                           "index": text_index}))
                    text_index = -1
                index = next_index
                next_index += 1
                tool_indexes[upstream_index] = index
                fn = call.get("function") if is_record(call.get("function")) else {}
                out.append(_sse_event("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": string_of(call.get("id")) or ("toolu_%d" % index),
                        "name": string_of(fn.get("name")), "input": {}}}))
            fn = call.get("function") if is_record(call.get("function")) else {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                out.append(_sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": fn["arguments"]}}))
        if choice.get("finish_reason") is not None:
            pending_stop = _stop_reason_anthropic(choice["finish_reason"])
        return out

    while True:
        data = read_chunk()
        if not data:
            break
        for frame in parser.feed(data):
            for event in events_of_frame(frame):
                yield event
    for frame in parser.flush():
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop or "end_turn"):
            yield event


def openai_events_from_anthropic_stream(read_chunk, model):
    """Translate an Anthropic SSE stream into OpenAI chunks, yielding bytes."""
    parser = SseFrames()
    role_sent = False
    finished = False
    pending_stop = None
    input_tokens = 0
    output_tokens = 0
    tool_calls = {}   # content-block index -> {"id","name","arguments"}

    def role_chunk():
        nonlocal role_sent
        if role_sent:
            return
        role_sent = True
        yield _openai_chunk(model, {"role": "assistant", "content": ""}, None)

    def finish_events():
        nonlocal finished
        if finished:
            return
        finished = True
        out = [_openai_chunk(model, {}, _stop_reason_openai(pending_stop)),
               _openai_usage_chunk(model, input_tokens, output_tokens),
               b"data: [DONE]\n\n"]
        return out

    def events_of_frame(frame):
        nonlocal pending_stop, input_tokens, output_tokens
        event, payload = frame
        out = []
        try:
            data = json.loads(payload)
        except Exception:
            return out
        if not is_record(data):
            return out
        if event == "message_start":
            message = data.get("message") if is_record(data.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            input_tokens = _number_or_zero(usage.get("input_tokens"))
            return out
        if event == "content_block_start":
            index = data.get("index")
            block = data.get("content_block") if is_record(data.get("content_block")) else {}
            if block.get("type") == "tool_use" and isinstance(index, int):
                tool_calls[index] = {
                    "id": string_of(block.get("id")) or ("toolu_%d" % index),
                    "name": string_of(block.get("name")),
                    "arguments": "",
                }
                out.extend(role_chunk())
                out.append(_openai_chunk(model, {"tool_calls": [{
                    "index": index, "id": tool_calls[index]["id"],
                    "type": "function",
                    "function": {"name": tool_calls[index]["name"],
                                 "arguments": ""}}]}, None))
            return out
        if event == "content_block_delta":
            index = data.get("index")
            delta = data.get("delta") if is_record(data.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                out.extend(role_chunk())
                out.append(_openai_chunk(model, {"content": delta["text"]}, None))
            elif delta.get("type") == "input_json_delta" \
                    and isinstance(delta.get("partial_json"), str) \
                    and isinstance(index, int) and index in tool_calls:
                tool_calls[index]["arguments"] += delta["partial_json"]
                out.extend(role_chunk())
                out.append(_openai_chunk(model, {"tool_calls": [{
                    "index": index,
                    "function": {"arguments": delta["partial_json"]}}]}, None))
            return out
        if event == "message_delta":
            delta = data.get("delta") if is_record(data.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                pending_stop = delta["stop_reason"]
            usage = data.get("usage") if is_record(data.get("usage")) else {}
            output_tokens = _number_or_zero(usage.get("output_tokens"))
            return out
        if event == "message_stop":
            out.extend(role_chunk())
            out.extend(finish_events())
            return out
        return out

    while True:
        data = read_chunk()
        if not data:
            break
        for frame in parser.feed(data):
            for event in events_of_frame(frame):
                yield event
    for frame in parser.flush():
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in role_chunk():
            yield event
        for event in finish_events():
            yield event


def openai_events_from_gemini_stream(read_chunk, model):
    """Translate a Gemini SSE stream into OpenAI chunks, yielding bytes."""
    parser = SseFrames()
    role_sent = False
    finished = False
    pending_stop = None
    input_tokens = 0
    output_tokens = 0

    def role_chunk():
        nonlocal role_sent
        if role_sent:
            return
        role_sent = True
        yield _openai_chunk(model, {"role": "assistant", "content": ""}, None)

    def finish_events():
        nonlocal finished
        if finished:
            return
        finished = True
        return [_openai_chunk(model, {}, _stop_reason_openai(pending_stop)),
                _openai_usage_chunk(model, input_tokens, output_tokens),
                b"data: [DONE]\n\n"]

    def events_of_frame(frame):
        nonlocal pending_stop, input_tokens, output_tokens
        _event, payload = frame
        out = []
        try:
            data = json.loads(payload)
        except Exception:
            return out
        if not is_record(data):
            return out
        text, prompt, completion, finish = _gemini_text_and_usage(data)
        if prompt or completion:
            input_tokens = prompt
            output_tokens = completion
        if text != "":
            out.extend(role_chunk())
            for piece in _split_text(text):
                out.append(_openai_chunk(model, {"content": piece}, None))
        if finish is not None:
            pending_stop = finish
        return out

    while True:
        data = read_chunk()
        if not data:
            break
        for frame in parser.feed(data):
            for event in events_of_frame(frame):
                yield event
    for frame in parser.flush():
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in role_chunk():
            yield event
        for event in finish_events():
            yield event


def anthropic_events_from_gemini_stream(read_chunk, model):
    """Translate a Gemini SSE stream into Anthropic events, yielding bytes."""
    parser = SseFrames()
    started = False
    finished = False
    text_open = False
    pending_stop = None
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return
        started = True
        yield _sse_event("message_start", {
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message",
                        "role": "assistant", "model": model, "content": [],
                        "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0}},
        })

    def finish_events():
        nonlocal finished
        if finished:
            return
        finished = True
        out = []
        if text_open:
            out.append(_sse_event("content_block_stop",
                                  {"type": "content_block_stop", "index": 0}))
        out.append(_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": _stop_reason_anthropic(pending_stop),
                      "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }))
        out.append(_sse_event("message_stop", {"type": "message_stop"}))
        return out

    def events_of_frame(frame):
        nonlocal text_open, pending_stop, output_tokens
        _event, payload = frame
        out = []
        try:
            data = json.loads(payload)
        except Exception:
            return out
        if not is_record(data):
            return out
        text, _prompt, completion, finish = _gemini_text_and_usage(data)
        if completion:
            output_tokens = completion
        if text != "":
            out.extend(start_events())
            if not text_open:
                text_open = True
                out.append(_sse_event("content_block_start", {
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""}}))
            out.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": text}}))
        if finish is not None:
            pending_stop = finish
        return out

    while True:
        data = read_chunk()
        if not data:
            break
        for frame in parser.feed(data):
            for event in events_of_frame(frame):
                yield event
    for frame in parser.flush():
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events():
            yield event


# ---------------------------------------------------------------------------
# Upstream calls and the client-facing server.
# ---------------------------------------------------------------------------

class UpstreamResult:
    def __init__(self, status, headers, body_reader):
        self.status = status
        self.headers = headers
        self.body_reader = body_reader  # callable returning next bytes, b"" at end


def _read_all(reader):
    chunks = []
    while True:
        data = reader()
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def call_upstream(url, headers, body_bytes):
    """POST one request. Returns an UpstreamResult, or None when unreachable."""
    req = urllib.request.Request(url, data=body_bytes, method="POST",
                                 headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except Exception:
            pass
        return UpstreamResult(exc.code, dict(exc.headers or {}),
                              lambda: b"")
    except Exception as exc:
        log("upstream %s unreachable: %s" % (url, exc))
        return None
    return UpstreamResult(resp.status, dict(resp.headers or {}),
                          lambda: resp.read(8192))


def _url_quote(value):
    from urllib.parse import quote
    return quote(value, safe="")


# JSON translators, keyed by (client wire, provider wire). A same-wire pair
# is a passthrough: the provider's bytes go back untouched.
JSON_TRANSLATORS = {
    ("openai", "anthropic"): anthropic_json_to_openai,
    ("openai", "gemini"): gemini_json_to_openai,
    ("anthropic", "openai"): openai_json_to_anthropic,
    ("anthropic", "gemini"): gemini_json_to_anthropic,
}

# Stream translators, keyed the same way; same-wire pairs pipe bytes through.
STREAM_TRANSLATORS = {
    ("openai", "anthropic"): openai_events_from_anthropic_stream,
    ("openai", "gemini"): openai_events_from_gemini_stream,
    ("anthropic", "openai"): anthropic_events_from_openai_stream,
    ("anthropic", "gemini"): anthropic_events_from_gemini_stream,
}


class Aggregator:
    def __init__(self, brief, catalog, probes):
        self.brief = brief
        self.catalog = catalog
        self.router = Router(probes, brief["ratings"])
        self.manifest_index = 0
        self.lock = threading.Lock()  # one upstream call in flight, ever
        manifest = brief["workload"]["manifest"] \
            if brief["workload"] is not None else []
        self.manifest = manifest
        self.served_count = 0

    # -- requirements ----------------------------------------------------
    def requirements_for(self, client_api, body):
        if self.manifest_index < len(self.manifest):
            entry = self.manifest[self.manifest_index]
            self.manifest_index += 1
            return dict(entry["requires"])
        return requirements_from_body(client_api, body)

    # -- upstream request per provider wire -------------------------------
    def upstream_request(self, provider, model_id, client_api, body, stream):
        base = join_base(ENV_URL, provider["base"])
        headers = {"content-type": "application/json"}
        headers.update(auth_headers(provider["auth"], provider["key"]))
        if provider["wire"] == "openai":
            if client_api == "anthropic":
                payload = anthropic_to_openai(body, model_id)
            else:
                payload = dict(body)
                payload["model"] = model_id
            if stream:
                payload["stream"] = True
                # The client never asks for usage in a stream, but the wire
                # check needs the provider's usage: force it on.
                payload["stream_options"] = {"include_usage": True}
            return base + "/v1/chat/completions", headers, \
                json.dumps(payload).encode("utf-8")
        if provider["wire"] == "anthropic":
            if client_api == "anthropic":
                payload = dict(body)
                payload["model"] = model_id
            else:
                payload = openai_to_anthropic(body, model_id)
            headers["anthropic-version"] = ANTHROPIC_VERSION
            return base + "/v1/messages", headers, \
                json.dumps(payload).encode("utf-8")
        # gemini
        payload = to_gemini(body, model_id, client_api)
        action = "streamGenerateContent" if stream else "generateContent"
        url = "%s/v1beta/models/%s:%s" % (base, _url_quote(model_id), action)
        return url, headers, json.dumps(payload).encode("utf-8")

    # -- one client request ------------------------------------------------
    def serve(self, handler, client_api, body):
        """Answer one client request, writing the response on `handler`."""
        stream = body.get("stream") is True
        requires = self.requirements_for(client_api, body)
        json_mode = False
        if client_api == "openai":
            fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
            json_mode = fmt.get("type") in ("json_object", "json_schema")
        candidates = self.router.rank(requires, json_mode)
        attempts = 0
        with self.lock:
            for provider, model in candidates:
                model_id = model["id"]
                url, headers, payload = self.upstream_request(
                    provider, model_id, client_api, body, stream)
                self.router.calls[provider["id"]] = \
                    self.router.calls.get(provider["id"], 0) + 1
                attempts += 1
                result = call_upstream(url, headers, payload)
                if result is None:
                    continue
                if result.status < 200 or result.status >= 300:
                    retry_after = result.headers.get("retry-after")
                    log("%s/%s refused HTTP %s%s" % (
                        provider["id"], model_id, result.status,
                        " (retry-after %s)" % retry_after
                        if retry_after else ""))
                    continue
                self.served_count += 1
                log("request %d served by %s/%s (attempt %d)"
                    % (self.served_count, provider["id"], model_id, attempts))
                self.answer(handler, provider, model_id, client_api,
                            result, stream)
                return
        self.send_error(handler, client_api, 503,
                        "no_free_model" if client_api == "openai"
                        else "overloaded_error",
                        "no free model on the directory meets this request")

    def answer(self, handler, provider, model_id, client_api, result, stream):
        """Translate one successful provider answer into the client's wire."""
        wire = provider["wire"]
        if not stream:
            raw = _read_all(result.body_reader)
            try:
                answer = json.loads(raw.decode("utf-8"))
            except Exception:
                answer = None
            if not is_record(answer):
                self.send_error(handler, client_api, 502,
                                "api_error" if client_api == "anthropic"
                                else "upstream_error",
                                "the provider answered a body that is not JSON")
                return
            if wire == client_api:
                self.send_json(handler, result.status, answer)
                return
            translator = JSON_TRANSLATORS.get((client_api, wire))
            self.send_json(handler, 200, translator(answer, model_id))
            return
        # Streaming: translate frame by frame as the bytes arrive. No
        # content-length is known up front, so close the connection when
        # the stream ends: the client reads until EOF.
        handler.close_connection = True
        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.send_header("cache-control", "no-cache")
        handler.send_header("connection", "close")
        handler.end_headers()
        reader = result.body_reader
        if wire == client_api:
            while True:
                data = reader()
                if not data:
                    break
                handler.wfile.write(data)
            return
        translator = STREAM_TRANSLATORS.get((client_api, wire))
        for chunk in translator(reader, model_id):
            handler.wfile.write(chunk)

    # -- response helpers ---------------------------------------------------
    def send_json(self, handler, status, body):
        text = json.dumps(body).encode("utf-8")
        handler.send_response(status)
        handler.send_header("content-type", "application/json")
        handler.send_header("content-length", str(len(text)))
        handler.end_headers()
        handler.wfile.write(text)

    def send_error(self, handler, client_api, status, type_, message):
        if client_api == "anthropic":
            self.send_json(handler, status, anthropic_error(type_, message))
        else:
            self.send_json(handler, status, openai_error(message, type_, None))

    def send_models(self, handler, headers):
        models = self.router.served_models()
        if "x-api-key" in headers:
            self.send_json(handler, 200, anthropic_model_list(models))
        else:
            self.send_json(handler, 200, openai_model_list(models))


def requirements_from_body(client_api, body):
    """Requirements read off a request body, when the manifest has no row."""
    if client_api == "anthropic":
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        vision = False
        for message in messages:
            if not is_record(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_record(block) and block.get("type") == "image":
                    vision = True
        return {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
                "vision": vision, "json": False}
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    return {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
            "vision": vision,
            "json": fmt.get("type") in ("json_object", "json_schema")}


class Handler(BaseHTTPRequestHandler):
    aggregator = None  # set by start()
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep stdout clean for --list
        pass

    def _read_json(self):
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def _serve(self, client_api, body):
        try:
            self.aggregator.serve(self, client_api, body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            log("%s failed: %s" % (client_api, exc))
            try:
                self.aggregator.send_error(
                    self, client_api, 500,
                    "api_error" if client_api == "anthropic" else "internal_error",
                    "the server failed to answer")
            except Exception:
                pass

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            self.aggregator.send_models(self, self.headers)
            return
        self.aggregator.send_error(self, "openai", 404, "not_found",
                                   "no route GET %s" % self.path)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            body = self._read_json()
            if body is None:
                self.aggregator.send_error(self, "openai", 400,
                                           "invalid_request_error",
                                           "the request body is not one JSON object")
                return
            self._serve("openai", body)
            return
        if path == "/v1/messages":
            body = self._read_json()
            if body is None:
                self.aggregator.send_error(self, "anthropic", 400,
                                           "invalid_request_error",
                                           "the request body is not one JSON object")
                return
            self._serve("anthropic", body)
            return
        wire = "anthropic" if path == "/v1/messages" else "openai"
        self.aggregator.send_error(self, wire, 404, "not_found",
                                   "no route POST %s" % path)


def load_all():
    """Read the brief, probe the providers, build the one catalog."""
    brief = read_brief(PROMPT_FILE)
    if brief["error"] is not None:
        log(brief["error"])
    if not brief["providers"] and ENV_URL:
        instance = fetch_instance(ENV_URL)
        if instance is not None and instance["providers"]:
            brief["providers"] = instance["providers"]
    probes = []
    if ENV_URL:
        for provider in brief["providers"]:
            probes.append(probe_provider(ENV_URL, provider))
        for probe in probes:
            if not probe["ok"] and probe["error"] is not None:
                log(probe["error"])
    if probes and any(p["ok"] for p in probes):
        catalog = build_catalog(probes, brief["ratings"])
    elif brief["providers"]:
        catalog = fallback_catalog(brief["providers"], brief["ratings"])
    else:
        catalog = {"providers": []}
    return brief, catalog, probes


def register(port, catalog):
    req = urllib.request.Request(
        ENV_URL + "/register",
        data=json.dumps({"protocol_version": PROTOCOL_VERSION,
                         "port": port, "catalog": catalog}).encode("utf-8"),
        method="POST", headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:
        if resp.status != 200:
            raise RuntimeError("POST %s/register answered HTTP %s"
                               % (ENV_URL, resp.status))
        reply = json.loads(resp.read().decode("utf-8"))
    if is_record(reply) and isinstance(reply.get("requests"), (int, float)):
        log("registered: the workload holds %s requests" % reply["requests"])


def poll_until_done(server):
    while True:
        try:
            req = urllib.request.Request(ENV_URL + "/work",
                                         headers={"accept": "application/json"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    work = json.loads(resp.read().decode("utf-8"))
                    if is_record(work) and work.get("done") is True:
                        server.shutdown()
                        return
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)


def start():
    brief, catalog, probes = load_all()
    if "--list" in sys.argv:
        # The same document the register body carries, byte for byte.
        text = json.dumps(catalog) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    aggregator = Aggregator(brief, catalog, probes)
    Handler.aggregator = aggregator
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout")
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len(catalog["providers"]),
           len(aggregator.router.served_models())))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    register(port, catalog)
    poll_until_done(server)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(start())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        log(str(exc))
        sys.exit(1)
