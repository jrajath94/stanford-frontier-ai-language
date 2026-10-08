#!/bin/bash
# retry-loop push-dir; logs to per-stage files; no pipes masking exit codes
REPO="stanford-frontier-ai-language"
STAGE="$1"
LOG="$2"
rm -f "$LOG"
for i in $(seq 1 15); do
  echo "attempt $i $(date -u +%T)" >> "$LOG"
  cd ~/workspace/skills/github && python3 bin/gh_publish.py push-dir "$REPO" "$STAGE" --message "cs329h companions: u01-u10 artifacts + course index" >> "$LOG" 2>&1
  rc=$?
  echo "exit=$rc" >> "$LOG"
  if [ $rc -eq 0 ]; then echo "STAGE $STAGE succeeded on attempt $i"; exit 0; fi
  sleep 8
done
echo "STAGE $STAGE FAILED after 15 attempts"
exit 1
