#!/usr/bin/env bash
# Runs ON the EC2 instance. Copied there by the deploy job, then executed with
# the registry token on stdin.
#
# Kept in the repository rather than inlined in the workflow so it can be read,
# reviewed and run by hand when a deploy needs debugging:
#
#   cd ~/mini-local-analytics && printf '%s\n' "$TOKEN" | REGISTRY_USER=you bash deploy.sh
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/mini-local-analytics}"
COMPOSE_FILE="docker-compose.prod.yml"
CONTAINER="mini-local-analytics"
REGISTRY="${REGISTRY:-ghcr.io}"

cd "$APP_DIR"

# The token arrives on stdin, so it never appears in a command line where any
# other user on the box could read it out of `ps`.
docker login "$REGISTRY" -u "${REGISTRY_USER:?REGISTRY_USER is required}" --password-stdin

cleanup() { docker logout "$REGISTRY" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# What is running right now, so a failed rollout has somewhere to go back to.
PREVIOUS=$(docker inspect --format '{{.Image}}' "$CONTAINER" 2>/dev/null || true)
echo "Currently running: ${PREVIOUS:-nothing}"

echo "Pulling the new image..."
docker compose -f "$COMPOSE_FILE" pull

# Stop before start, never both at once. DuckDB is embedded and single-writer:
# two processes holding one database file is precisely what it forbids. That
# costs a few seconds of downtime, which is the right trade for this engine.
echo "Restarting..."
docker compose -f "$COMPOSE_FILE" down --remove-orphans || true
docker compose -f "$COMPOSE_FILE" up -d

# Wait on the container's own healthcheck rather than a fixed sleep, so a slow
# start is tolerated and a broken one is caught quickly.
echo "Waiting for health..."
STATUS=starting
for _ in $(seq 1 40); do
  STATUS=$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo starting)
  [ "$STATUS" = "healthy" ] && break
  sleep 3
done

if [ "$STATUS" != "healthy" ]; then
  echo "::error::Container never became healthy (last status: $STATUS)."
  docker compose -f "$COMPOSE_FILE" logs --tail 80 || true
  if [ -n "$PREVIOUS" ]; then
    echo "Rolling back to $PREVIOUS"
    APP_IMAGE="$PREVIOUS" docker compose -f "$COMPOSE_FILE" up -d || true
  fi
  exit 1
fi

# Independent of the healthcheck, which runs inside the container: this proves
# the published port works too, so a broken port mapping fails the deploy
# instead of being discovered by a user.
curl -fsS --max-time 10 http://127.0.0.1:8000/api/status > /dev/null
echo "Deployed and healthy."

# The root volume is 8 GB and every deploy leaves a superseded image behind.
docker image prune -af --filter "until=168h" >/dev/null 2>&1 || true
