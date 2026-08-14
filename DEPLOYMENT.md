# Deployment

CI/CD for the Mini Local Analytics Assistant: GitHub Actions builds and tests,
GitHub Container Registry stores the image, and one EC2 instance runs it.

```
push to main
    ↓
  test      862 offline tests, no API key, no network
    ↓
  build     image built on the runner, pushed to ghcr.io, pinned by digest
    ↓
  deploy    ssh to EC2 → pull → down → up → wait for healthy → rollback on failure
```

**The image is built on the GitHub runner, never on EC2.** A t2.micro has one
vCPU and 1 GB of RAM; installing pandas, matplotlib and duckdb there is slow
enough to risk an OOM kill. The server only pulls.

---

## 1. Repository secrets

`Settings → Secrets and variables → Actions → New repository secret`

| Secret | Required | Value for your setup | What it is |
|---|---|---|---|
| `EC2_HOST` | yes | `3.91.232.233` | Public IPv4 of the instance. Use an Elastic IP if you can — a plain public IP changes every time the instance stops, and the deploy then fails. |
| `EC2_USER` | yes | `ec2-user` | The login user. `ec2-user` on Amazon Linux; it would be `ubuntu` on Ubuntu. |
| `EC2_SSH_KEY` | yes | the **entire contents** of your `.pem` | Not the filename, not the path — open the file and paste everything, including the `-----BEGIN …-----` and `-----END …-----` lines. |
| `GROQ_API_KEY` | yes | your Groq key | Written to `~/mini-local-analytics/.env` on the server with `umask 077`. Never enters an image layer. |
| `GHCR_PAT` | only if private | a PAT with `read:packages` | Needed only while the GHCR package is private. Make the package public and you can skip it. |
| `EC2_KNOWN_HOSTS` | optional | output of `ssh-keyscan -H 3.91.232.233` | Pins the server's host key. Without it the workflow trusts the key on first sight, which cannot tell a swapped host from the real one. |

Nothing else is needed. Pushing to `ghcr.io` uses the built-in `GITHUB_TOKEN`,
which Actions provides automatically — there is no registry secret to create.

### Pasting the SSH key correctly

This is where most first deploys fail. On Windows:

```powershell
Get-Content C:\path\to\your-key.pem -Raw | Set-Clipboard
```

Then paste into the secret. The workflow validates the key with `ssh-keygen -y`
before using it, so a truncated or CRLF-mangled paste fails immediately with a
clear message instead of a confusing SSH error.

---

## 2. Prepare the EC2 instance (once)

Connect via **EC2 Instance Connect** in the console, or:

```bash
ssh -i your-key.pem ec2-user@3.91.232.233
```

Then run the bootstrap, which installs Docker, the Compose v2 plugin, and 2 GB
of swap:

```bash
curl -fsSL https://raw.githubusercontent.com/diyapachori789/Mini-local-analytics-assistment/main/scripts/ec2-bootstrap.sh | bash
```

Log out and back in afterwards — Linux applies new group membership only to new
sessions, so `docker` without `sudo` will not work until you do.

Verify:

```bash
docker --version && docker compose version && free -h
```

### Why swap

t2.micro has 1 GB of RAM and no swap by default. When memory runs out the
kernel's OOM killer picks a process to kill, and it sometimes picks `sshd` —
locking you out of your own instance with no way back in short of a reboot.

---

## 3. Security group

`EC2 → Instances → your instance → Security → Security groups → Edit inbound rules`

| Type | Port | Source | Why |
|---|---|---|---|
| SSH | 22 | `0.0.0.0/0` | The deploy runs from a **GitHub-hosted runner in Azure**, not from your machine. |
| Custom TCP | 8000 | **My IP** | Only you need to open the app. |

**Port 22 cannot be scoped to "My IP".** That was the cause of
`Host key verification failed` on the first deploy attempt: `ssh-keyscan` on the
runner could not reach the instance, so the host key was never learned. GitHub's
runners use dynamic Azure addresses, and while GitHub publishes its ranges at
`https://api.github.com/meta`, there are thousands of them and a security group
allows 60 rules by default — so pinning them is not practical.

Opening 22 to the world is less alarming than it sounds *provided* password
authentication stays off, which is the default on Amazon Linux 2023: only your
`.pem` gets in. Confirm it with:

```bash
sudo sshd -T | grep -E 'passwordauthentication|permitrootlogin'
# want: passwordauthentication no
```

If you would rather not expose 22 at all, the clean answer is **AWS SSM Session
Manager**: attach an instance profile with `AmazonSSMManagedInstanceCore`, give
the workflow AWS credentials via OIDC, and replace the ssh steps with
`aws ssm send-command`. No inbound rule at all. That is the right production
posture and a reasonable next step once this pipeline is working.

> **Port 8000 is different — keep it scoped to your IP.** This application has
> no login of its own, so anyone who can reach it can query your data and spend
> your Groq quota. "My IP" is your current address; on a home connection it
> changes, and the app stops being reachable until you update the rule.

---

## 4. Deploy

```bash
git add .github scripts docker-compose.prod.yml DEPLOYMENT.md
git commit -m "Add CI/CD pipeline"
git push -u origin main
```

Watch it under the **Actions** tab. On success the app is at
`http://3.91.232.233:8000`.

Pull requests run the tests only — they never build or deploy.

### Requiring approval before a deploy

`Settings → Environments → New environment → production`, then add yourself as a
required reviewer. The workflow already declares `environment: production`, so
the deploy job will wait for your approval.

---

## 5. How the deploy behaves

**Images are pinned by digest, not by tag.** The deploy writes
`APP_IMAGE=ghcr.io/…@sha256:…` rather than `:latest`, so a container restarting
three weeks later runs exactly the code that was tested. A tag can be
overwritten; a digest cannot.

**The container is stopped before the new one starts.** DuckDB is embedded and
single-writer — exactly one process may hold a database file. A rolling restart
would briefly run two, which is precisely what the engine forbids. That costs a
few seconds of downtime per deploy, which is the correct trade for this engine
rather than an oversight.

**A failed rollout rolls back.** The script records the running image before it
starts. If the new container never reports healthy, it dumps the last 80 log
lines and restarts the previous image.

**Health is checked twice.** Once via the container's own healthcheck (which
calls `/api/status`, touching no database and making no model call, so it costs
no quota), and again with `curl` from the host — that second check is what
catches a broken port mapping.

---

## 6. Why production uses a separate compose file

`docker-compose.prod.yml` is standalone rather than an override of
`docker-compose.yml`. Two reasons it cannot simply extend it:

**Bind mounts break under a non-root user.** The image runs as uid 10001. A
host directory on EC2 is owned by `ec2-user` (uid 1000), so the container gets a
read-only home and cannot write its database. This works on Docker Desktop only
because of its filesystem shim, which does not exist on Linux. Production uses
**named volumes**, which Docker seeds from the image and which therefore inherit
the right ownership. Verified: the container writes `analytics.duckdb`,
`analytics.duckdb.wal` and `history.duckdb` as `appuser`, and they survive a
full `down` / `up`.

**`build:` would come with it.** An override merges with the base, so the merged
config would still want to build from source — which means shipping the whole
repository to the server. Standalone means only two files ever reach EC2:
`docker-compose.prod.yml` and `.env`.

Other differences:

| | local (`docker-compose.yml`) | server (`docker-compose.prod.yml`) |
|---|---|---|
| image | built from source | pulled, digest-pinned |
| port | `127.0.0.1:8000` — loopback only | `8000` on all interfaces |
| storage | bind mounts, visible in the project | named volumes |
| `data/` | mounted from the host | baked into the image |
| memory | unbounded | `768m` cap, so a runaway container cannot take down sshd |
| logs | unbounded | 10 MB × 3 files, so they cannot fill the 8 GB root volume |

---

## 7. Operating it

```bash
cd ~/mini-local-analytics

docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f --tail 100
docker compose -f docker-compose.prod.yml restart

# Application logs (rotating, inside the named volume)
docker compose -f docker-compose.prod.yml exec web tail -f /app/logs/web_app.log
```

**Back up the database** before anything destructive:

```bash
docker run --rm -v mini-local-analytics_duckdb-data:/data -v "$PWD":/backup \
  alpine tar czf /backup/duckdb-$(date +%F).tar.gz -C /data .
```

`docker compose -f docker-compose.prod.yml down -v` deletes the volumes and
every stored query along with them. `down` without `-v` is safe.

---

## 8. Validating the workflow before you push

A workflow file with a bad expression does not fail a job — it fails to load at
all, so nothing runs and you find out only after pushing. Check it locally in
one command:

```bash
docker run --rm -v "${PWD}:/repo" -w /repo rhysd/actionlint:latest
```

On PowerShell the `${PWD}` form works as written. Exit code 0 means clean. The
image also bundles shellcheck, so it lints the bash inside every `run:` block
and `scripts/deploy.sh` at the same time.

This is worth knowing because GitHub evaluates some fields before the `secrets`
context exists — `environment.url`, `if`, `runs-on`, `name`, `concurrency` and
`timeout-minutes` among them. A `${{ secrets.X }}` in any of those is rejected
with `Unrecognized named-value: 'secrets'`, which reads like a typo but is a
scoping rule. Use `vars.X` (a repository *variable*) in those positions, or move
the value into `env:` at step level as this workflow does.

---

## 9. Troubleshooting

| Symptom | Cause |
|---|---|
| `Host key verification failed` | Almost always a stale `EC2_KNOWN_HOSTS`: it holds a key for an IP the instance no longer has. **Delete the secret** — the workflow then trusts the host on first connection. Repin later with `ssh-keyscan -H <ip>`. The workflow now checks for this and fails with a named error instead. |
| `Cannot reach <ip>:22` | Port 22 is not open to the runner. The security group's SSH rule must be `0.0.0.0/0`, not "My IP" — see §3. Also check the instance is running and `EC2_HOST` matches its **current** public IP. |
| `Permission denied (publickey)` | `EC2_SSH_KEY` is truncated or has Windows line endings, or `EC2_USER` is wrong. |
| `Connection timed out` after 15s | Instance stopped, wrong IP, or no inbound rule on 22 at all. |
| `denied` when pulling from ghcr.io | Package is private and `GHCR_PAT` is missing or lacks `read:packages`. Or make the package public: `Packages → Package settings → Change visibility`. |
| Deploy succeeds, browser times out | Security group has no inbound rule for 8000 from your IP. |
| `required variable APP_IMAGE is missing` | `.env` was not written — the deploy step before it failed. |
| Container unhealthy, logs say `GROQ_API_KEY` | The secret is unset or empty. `/api/status` still answers without it; queries do not. |
| Host unreachable after a stop/start | The public IP changed. Attach an Elastic IP, or update `EC2_HOST`. |
| `docker: command not found` | Bootstrap not run, or you did not log out and back in after it. |
