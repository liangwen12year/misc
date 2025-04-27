#!/usr/bin/env bash
set -euo pipefail

# Configuration
CORPUS_DIR="$HOME/.convokit/saved-corpora/movie-corpus"
UTTERANCE_FILE="$CORPUS_DIR/utterances.jsonl"
SCRIPT="./gen_sample_11_lab.py"
API_KEY="sk_7625b241b028c94cbdc22332e98547d609576805d9f386d2"
OUTPUT_DIR="./samples"
NUM_SAMPLES=10

# Requires: jq (for JSON parsing)
if ! command -v jq &>/dev/null; then
  echo "⚠️  Please install jq (e.g. 'sudo apt install jq') and rerun." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Read first 20 IDs into an array
mapfile -t ids < <(
  jq -r '.id' "$UTTERANCE_FILE" | head -n $((NUM_SAMPLES + 1))
)

# Sanity check
if [ "${#ids[@]}" -lt $((NUM_SAMPLES + 1)) ]; then
  echo "⚠️  Not enough utterances in $UTTERANCE_FILE" >&2
  exit 1
fi

# Generate samples
for i in $(seq 0 $((NUM_SAMPLES - 1))); do
  prev_id="${ids[$i]}"
  next_id="${ids[$((i + 1))]}"
  out="$OUTPUT_DIR/sample_${i}.mp3"

  echo "▶ Generating $out from $prev_id → $next_id"
  python "$SCRIPT" \
    -PRI "$prev_id" \
    -NRI "$next_id" \
    -C movie-corpus \
    -k "$API_KEY" \
    -o "$out"
done

echo "✅ All $NUM_SAMPLES samples written to $OUTPUT_DIR/"

