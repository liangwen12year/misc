#!/usr/bin/env bash
set -uo pipefail

### CONFIGURATION & HELP ###

NUM_SAMPLES=100
JOBS=4
CORPUS_NAME="movie-corpus"
OUTPUT_DIR="./samples"
FINAL_DIR="$OUTPUT_DIR/final"
LOG_DIR="./logs"
API_KEY="sk_7625b241b028c94cbdc22332e98547d609576805d9f386d2"
SCRIPT="./gen_sample_11_lab.py"

usage() {
  cat <<EOF
Usage: $0 [-n NUM] [-j JOBS] [-o OUTPUT_DIR] [-k API_KEY]
Generate TTS for up to NUM conversations (default $NUM_SAMPLES), in parallel (default $JOBS jobs).

Options:
  -n NUM         Maximum number of conversations to process
  -j JOBS        Number of parallel jobs
  -o OUTPUT_DIR  Base output directory (default $OUTPUT_DIR)
  -k API_KEY     ElevenLabs API key (overrides env ELEVEN_LABS_API_KEY)
  -h             Show this help message
EOF
  exit
}

while getopts "n:j:o:k:h" opt; do
  case $opt in
    n) NUM_SAMPLES=$OPTARG ;;
    j) JOBS=$OPTARG       ;;
    o) OUTPUT_DIR=$OPTARG ;;
    k) API_KEY=$OPTARG    ;;
    h) usage              ;;
    *) usage              ;;
  esac
done

# Ensure we have everything we need
if [[ -z "${API_KEY:-}" ]]; then
  echo "❌ ERROR: No API key supplied (-k) or in \$ELEVEN_LABS_API_KEY" >&2
  exit 1
fi
if ! command -v jq &>/dev/null; then
  echo "❌ ERROR: jq is required. Install it and retry." >&2
  exit 1
fi
if [[ ! -f "$HOME/.convokit/saved-corpora/$CORPUS_NAME/utterances.jsonl" ]]; then
  echo "❌ ERROR: Utterances file not found in ~/.convokit/saved-corpora/$CORPUS_NAME/" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$FINAL_DIR" "$LOG_DIR"

### BUILD LIST OF CONVERSATIONS TO PROCESS ###

# all conversation ids
mapfile -t ALL_CONV < <(
  jq -r 'select(.conversation_id) | .conversation_id' \
    ~/.convokit/saved-corpora/$CORPUS_NAME/utterances.jsonl \
    | sort -u
)

# which are already done? if any file for that conv_id exists in FINAL_DIR
declare -A DONE
while IFS= read -r path; do
  fname=$(basename "$path")
  # extract conv_id: speaker_conv_turn.mp3
  conv=${fname#*_}       # removes speaker_
  conv=${conv%%_*}       # takes up to next _
  DONE["$conv"]=1
done < <(ls -1 "$FINAL_DIR"/*.mp3 2>/dev/null || :)

# build to-process list
TO_PROCESS=()
for c in "${ALL_CONV[@]}"; do
  if [[ ${DONE[$c]:-0} -eq 1 ]]; then
    continue
  fi
  TO_PROCESS+=("$c")
done

echo "ℹ️  Total conversations: ${#ALL_CONV[@]}"
echo "✅ Already done: ${#DONE[@]}"
echo "▶ To process: ${#TO_PROCESS[@]} (capped at $NUM_SAMPLES)"

### WORKER ###

worker() {
  conv_id=$1
  logfile="$LOG_DIR/$conv_id.log"
  {
    echo "--- Processing $conv_id ---"
    if python "$SCRIPT" -CONV "$conv_id" -C "$CORPUS_NAME" -k "$API_KEY" -o "$OUTPUT_DIR"; then
      echo "SUCCESS: $conv_id"
    else
      echo "FAIL:    $conv_id"
    fi
  } &>>"$logfile"
}

export -f worker
export SCRIPT CORPUS_NAME API_KEY OUTPUT_DIR LOG_DIR

### RUN IN PARALLEL ###

printf "%s\n" "${TO_PROCESS[@]:0:$NUM_SAMPLES}" \
  | xargs -P "$JOBS" -n1 -I{} bash -c 'worker "$@"' _ {}

### SUMMARY ###

succ=$(grep -rl "SUCCESS" "$LOG_DIR" | wc -l)
fail=$(grep -rl "FAIL"    "$LOG_DIR" | wc -l)
skip=${#DONE[@]}
done=$((succ + fail + skip))

echo
echo "===== SUMMARY ====="
echo "Requested:   $NUM_SAMPLES"
echo "Processed:   $done"
echo "  ✔ Success: $succ"
echo "  ✖ Fail:    $fail"
echo "  ⏭ Skipped: $skip"
echo "Logs under $LOG_DIR/*.log"
echo "==================="

