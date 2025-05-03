#!/usr/bin/env bash
set -uo pipefail  # No -e to keep processing on error

# Configuration
CORPUS_DIR="$HOME/.convokit/saved-corpora/movie-corpus"
UTTERANCES_FILE="$CORPUS_DIR/utterances.jsonl"
SCRIPT="./gen_sample_11_lab.py"
API_KEY="sk_7625b241b028c94cbdc22332e98547d609576805d9f386d2"
OUTPUT_DIR="./samples"
NUM_SAMPLES=5000  # Limit to 5000 samples

# Requires: jq (for JSON parsing)
if ! command -v jq &>/dev/null; then
  echo "⚠️  Please install jq (e.g. 'sudo apt install jq') and rerun." >&2
  exit 1
fi

# Check utterances.jsonl exists
if [ ! -f "$UTTERANCES_FILE" ]; then
  echo "❌ Error: $UTTERANCES_FILE not found. Ensure the corpus is downloaded." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Extract all unique conversation IDs from utterances.jsonl
mapfile -t conv_ids < <(
  jq -r 'select(.conversation_id != null) | .conversation_id' "$UTTERANCES_FILE" | sort -u
)

total_convs=${#conv_ids[@]}
echo "ℹ️  Found $total_convs conversations. Generating audio for the first $NUM_SAMPLES samples..."

count=0
for conv_id in "${conv_ids[@]}"; do
  if [ "$count" -ge "$NUM_SAMPLES" ]; then
    break
  fi

  echo "▶ [$((count+1))/$NUM_SAMPLES] Generating audio for conversation: $conv_id"
  if ! python "$SCRIPT" \
    -CONV "$conv_id" \
    -C movie-corpus \
    -k "$API_KEY" \
    -o "$OUTPUT_DIR"; then
      echo "⚠️  Skipped conversation $conv_id due to error."
  fi

  ((count++))
done

echo "✅ $NUM_SAMPLES conversations processed. Audios saved to $OUTPUT_DIR/ (reversed composed conversations with 1-sec pauses)"

