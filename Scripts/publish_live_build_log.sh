#!/usr/bin/env bash
# Publish a redacted-by-default snapshot of xcodebuild.log to a run-specific,
# force-updated orphan branch so logs can be read before the Actions job ends.
set -euo pipefail

BUILD_PID="${1:?Usage: publish_live_build_log.sh BUILD_PID}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"
TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [[ -z "$TOKEN" ]]; then
  echo "Live log publisher: no GitHub token; skipping live snapshots."
  exit 0
fi

LOG_FILE="${GITHUB_WORKSPACE:-$PWD}/xcodebuild.log"
BRANCH="actions-live-logs/run-${GITHUB_RUN_ID}"
TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

git -C "$TMP_DIR" init -q
git -C "$TMP_DIR" config user.name "github-actions[bot]"
git -C "$TMP_DIR" config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git -C "$TMP_DIR" checkout --orphan snapshot >/dev/null 2>&1
git -C "$TMP_DIR" remote add origin "https://github.com/${GITHUB_REPOSITORY}.git"
AUTH="$(printf 'x-access-token:%s' "$TOKEN" | base64 | tr -d '\n')"
git -C "$TMP_DIR" config http.https://github.com/.extraheader "AUTHORIZATION: basic $AUTH"
unset AUTH TOKEN

publish_snapshot() {
  [[ -f "$LOG_FILE" ]] || : > "$LOG_FILE"
  tail -c 1048576 "$LOG_FILE" > "$TMP_DIR/build.log"
  {
    echo
    echo "--- Live snapshot metadata ---"
    echo "Workflow run: ${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
    echo "Snapshot time (UTC): $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Note: this is a point-in-time copy; the next snapshot replaces this file."
  } >> "$TMP_DIR/build.log"
  git -C "$TMP_DIR" add build.log
  if git -C "$TMP_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$TMP_DIR" commit --amend --no-edit -q
  else
    git -C "$TMP_DIR" commit -m "Live build log snapshot for run ${GITHUB_RUN_ID}" -q
  fi
  if ! git -C "$TMP_DIR" push --quiet --force origin "HEAD:refs/heads/$BRANCH"; then
    echo "Live log snapshot push failed; will retry at the next interval." >&2
    return 1
  fi
  echo "Live log snapshot published: https://github.com/${GITHUB_REPOSITORY}/blob/$BRANCH/build.log"
}

echo "Live log publisher started; updating the run-specific snapshot every 30 seconds."
while kill -0 "$BUILD_PID" 2>/dev/null; do
  publish_snapshot || true
  sleep 30
done
# Publish the final contents even if the build exits between intervals.
publish_snapshot || true
echo "Live log publisher finished."
