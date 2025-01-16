#!/bin/bash
# Description: This script triggers a GitLab mirror update and waits for completion
# Usage: ./update_gitlab_mirror.sh <gitlab_access_token> <gitlab_mirror_url>

GITLAB_ACCESS_TOKEN=$1
GITLAB_MIRROR_URL=$2

curl --fail-with-body --request POST --header "PRIVATE-TOKEN: ${GITLAB_ACCESS_TOKEN}" "${GITLAB_MIRROR_URL}"

echo "Mirror update started. Waiting for completion..."

# Poll for completion with 5-minute timeout
start_time=$(date +%s)
timeout=300  # 5 minutes in seconds

while true; do
  current_time=$(date +%s)
  elapsed=$((current_time - start_time))

  if [ $elapsed -ge $timeout ]; then
    echo "Timeout reached. Mirror update did not complete within 5 minutes."
    exit 1
  fi

  STATUS=$(curl --silent --header "PRIVATE-TOKEN: ${GITLAB_ACCESS_TOKEN}" "${GITLAB_MIRROR_URL}" | jq -r .update_status)

  if [ "$STATUS" = "finished" ]; then
    echo "Mirror update completed successfully"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "Mirror update failed"
    exit 1
  else
    echo "Mirror update still in progress. Elapsed time: $elapsed seconds"
    sleep 30
  fi
done
