#!/usr/bin/env bash
# Generate static TTS preview audio files for each actor × emotion variant.
#
# Prerequisites:
#   - Worker must be running (bun run dev:worker)
#   - CustomVoice TTS model must be loaded in Triton
#
# Usage:
#   bash scripts/generate-previews.sh [WORKER_URL]
#
# Output:
#   packages/web/public/audio/previews/{actor_id}/{variant_id}.wav

set -euo pipefail

WORKER_URL="${1:-http://localhost:8080}"
OUT_DIR="packages/web/public/audio/previews"

echo "Fetching actor catalog from ${WORKER_URL}/api/tts/actors ..."
CATALOG=$(curl -sf "${WORKER_URL}/api/tts/actors")

if [ -z "$CATALOG" ]; then
  echo "ERROR: Failed to fetch actor catalog. Is the worker running?" >&2
  exit 1
fi

SUPPORTS=$(echo "$CATALOG" | jq -r '.supports_preset_actors')
if [ "$SUPPORTS" != "true" ]; then
  echo "ERROR: Preset actors are not available. Is the CustomVoice model loaded?" >&2
  echo "       Message: $(echo "$CATALOG" | jq -r '.preset_actor_message // "none"')" >&2
  exit 1
fi

ACTOR_COUNT=$(echo "$CATALOG" | jq '.actors | length')
echo "Found ${ACTOR_COUNT} actors."
echo ""

GENERATED=0
FAILED=0

for ACTOR_IDX in $(seq 0 $((ACTOR_COUNT - 1))); do
  ACTOR_ID=$(echo "$CATALOG" | jq -r ".actors[$ACTOR_IDX].actor_id")
  ACTOR_LABEL=$(echo "$CATALOG" | jq -r ".actors[$ACTOR_IDX].label")
  ACTOR_LANG=$(echo "$CATALOG" | jq -r ".actors[$ACTOR_IDX].language")
  VARIANT_COUNT=$(echo "$CATALOG" | jq ".actors[$ACTOR_IDX].preview_variants | length")

  echo "=== ${ACTOR_LABEL} (${ACTOR_ID}, ${ACTOR_LANG}) - ${VARIANT_COUNT} variants ==="
  mkdir -p "${OUT_DIR}/${ACTOR_ID}"

  for VAR_IDX in $(seq 0 $((VARIANT_COUNT - 1))); do
    PREVIEW_ID=$(echo "$CATALOG" | jq -r ".actors[$ACTOR_IDX].preview_variants[$VAR_IDX].preview_id")
    PREVIEW_TEXT=$(echo "$CATALOG" | jq -r ".actors[$ACTOR_IDX].preview_variants[$VAR_IDX].text")
    PREVIEW_PROMPT=$(echo "$CATALOG" | jq -r ".actors[$ACTOR_IDX].preview_variants[$VAR_IDX].prompt")
    PREVIEW_LABEL=$(echo "$CATALOG" | jq -r ".actors[$ACTOR_IDX].preview_variants[$VAR_IDX].label")

    OUT_FILE="${OUT_DIR}/${ACTOR_ID}/${PREVIEW_ID}.wav"
    echo -n "  ${PREVIEW_LABEL} (${PREVIEW_ID})... "

    RESPONSE=$(curl -sf -X POST "${WORKER_URL}/api/tts" \
      -F "text=${PREVIEW_TEXT}" \
      -F "language=${ACTOR_LANG}" \
      -F "actor=${ACTOR_ID}" \
      -F "model=qwen3_tts_0_6b" \
      -F "prompt=${PREVIEW_PROMPT}" 2>&1) || {
        echo "FAILED"
        FAILED=$((FAILED + 1))
        continue
      }

    AUDIO_B64=$(echo "$RESPONSE" | jq -r '.audio_base64 // empty')
    if [ -z "$AUDIO_B64" ]; then
      echo "FAILED (empty audio)"
      FAILED=$((FAILED + 1))
      continue
    fi

    echo "$AUDIO_B64" | base64 -d > "$OUT_FILE"
    SIZE=$(wc -c < "$OUT_FILE")
    echo "OK ($(numfmt --to=iec "$SIZE" 2>/dev/null || echo "${SIZE}B"))"
    GENERATED=$((GENERATED + 1))
  done
  echo ""
done

echo "Done. Generated: ${GENERATED}, Failed: ${FAILED}"
echo "Files are in: ${OUT_DIR}/"
