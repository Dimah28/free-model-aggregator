#!/bin/bash
# Run one practice trial of free-model-aggregator-01 against this repo's main.py
# usage: run_trial.sh <trial-number> [trial-number...]
set -u
KIT="$HOME/workspace/makefun/free-model-aggregator-01-kit/free-model-aggregator-01-kit"
BOT="$HOME/workspace/makefun/free-model-aggregator-01"
for T in "$@"; do
  TRIAL="$KIT/practice/trial-$T"
  STATE="/tmp/fma-trial-$T"
  rm -rf "$STATE"; mkdir -p "$STATE"
  echo "=== trial $T ==="
  MAKEFUN_ENV_DATA_DIR="$TRIAL/input/env" MAKEFUN_STATE_DIR="$STATE" \
    MAKEFUN_ENV_PORT=0 node "$KIT/program/env/server.mjs" > "$STATE/env.log" 2>&1 &
  ENVPID=$!
  for i in $(seq 1 50); do
    [ -f "$STATE/ready" ] && break
    sleep 0.1
  done
  if [ ! -f "$STATE/ready" ]; then echo "env did not start"; cat "$STATE/env.log"; kill $ENVPID 2>/dev/null; continue; fi
  PORT=$(python3 -c "import json;print(json.load(open('$STATE/ready'))['port'])")
  echo "env on 127.0.0.1:$PORT"
  (cd "$BOT" && MAKEFUN_ENV_URL="http://127.0.0.1:$PORT" \
    MAKEFUN_PROMPT_FILE="$TRIAL/input/prompt.txt" \
    timeout 170 python3 main.py > "$STATE/bot.log" 2>&1)
  echo "bot exit: $?"
  kill -TERM $ENVPID 2>/dev/null
  wait $ENVPID 2>/dev/null
  if [ -f "$STATE/export/state.json" ]; then
    MAKEFUN_OUTPUT_DIR="$STATE/export" MAKEFUN_REFERENCE_DIR="$TRIAL/reference" \
      MAKEFUN_SCORES_DIR="$STATE/scores" node "$KIT/program/scorer/score.mjs"
    echo "--- per-request errors ---"
    python3 - "$STATE/export/state.json" <<'EOF'
import json, sys
st = json.load(open(sys.argv[1]))
for r in st["requests"]:
    if not r["valid"]:
        print(r["id"], "provider:", r["provider"], "model:", r["model"], "err:", r["error"])
        for c in r["checks"]:
            if not c["pass"]:
                print("   FAIL", c["name"], "-", c["detail"])
print("catalog_route:", st["catalog_route"])
EOF
  else
    echo "NO state.json; env log:"; cat "$STATE/env.log"; echo "bot log:"; cat "$STATE/bot.log"
  fi
done
