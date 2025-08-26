#!/usr/bin/env bash

set -Eeuo pipefail

CID="09d"
PY="$(command -v python)"   # 현재 쉘의 python 절대경로로 고정 (conda/env 안전)
SCRIPT="run_sac.py"
BATCHES=(256 512 1024)
WAIT_SECS=10                # 컨테이너 시작 후 대기 시간(초)

# --- sudo 인증을 한 번만 받고 유지 ---
if ! docker ps >/dev/null 2>&1; then
  echo "sudo 권한 확인"
  sudo -v
  # keep-alive
  while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
  SUDO="sudo"
else
  SUDO=""
fi

start_container() {
  echo "==> Starting $CID"
  $SUDO docker start "$CID" >/dev/null
  sleep "$WAIT_SECS"
}

stop_container() {
  echo "==> Stopping $CID"
  $SUDO docker stop -t 30 "$CID" >/dev/null || true
}

run_with_batch() {
  local bs="$1"
  start_container
  echo "==> Running: $PY $SCRIPT $bs"

  "$PY" "$SCRIPT" "$bs"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!! Python exited with code $rc (batch_size=$bs)"
    stop_container
    exit "$rc"
  fi

  stop_container
}

for bs in "${BATCHES[@]}"; do
  run_with_batch "$bs"
done

echo "ALL RUNS COMPLETED"
