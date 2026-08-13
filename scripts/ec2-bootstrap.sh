#!/usr/bin/env bash
# One-time setup for a fresh Amazon Linux 2023 instance.
#
# Run once, by hand, over EC2 Instance Connect or SSH:
#   curl -fsSL https://raw.githubusercontent.com/diyapachori789/Mini-local-analytics-assistment/main/scripts/ec2-bootstrap.sh | bash
# or paste it in. The CI pipeline assumes everything below is already in place.
set -euo pipefail

echo "==> Updating packages"
sudo dnf update -y

echo "==> Installing Docker"
sudo dnf install -y docker
sudo systemctl enable --now docker

# Lets ec2-user run docker without sudo, which is what the deploy script needs.
sudo usermod -aG docker ec2-user

echo "==> Installing the Docker Compose v2 plugin"
# Amazon Linux 2023 packages the engine but not the compose plugin, so it is
# installed as a CLI plugin. This is what makes `docker compose` (no hyphen)
# work; the old `docker-compose` binary is a different, unmaintained project.
COMPOSE_VERSION="v2.29.7"
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -fsSL \
  "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

echo "==> Adding swap"
# t2.micro has 1 GB of RAM and no swap. Pulling image layers and starting
# pandas/matplotlib can spike past that, and without swap the kernel's OOM
# killer picks a victim - sometimes sshd, which locks you out of your own box.
if [ ! -f /swapfile ]; then
  sudo dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  echo "    2 GB swap added"
else
  echo "    swap already present"
fi

mkdir -p ~/mini-local-analytics

echo
echo "==> Versions"
docker --version
sudo docker compose version
free -h | head -2

echo
echo "Done. Log out and back in before running docker without sudo"
echo "(group membership is only applied to new sessions)."
echo
echo "Still to do, in the AWS console:"
echo "  1. Security group: allow inbound TCP 8000 from YOUR IP only."
echo "     This app has no login of its own, so that rule is the only thing"
echo "     standing between the internet and your Groq quota."
echo "  2. Add the repository secrets listed in DEPLOYMENT.md."
