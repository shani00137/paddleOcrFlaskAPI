#!/usr/bin/env bash
# Fix broken Docker CLI symlinks (they point to an old /Volumes/Docker mount)
# Run: sh fix_docker_symlinks.sh   (it will ask for your Mac password)
set -e

BIN="/Applications/Docker.app/Contents/Resources/bin"
PLUGINS="/Applications/Docker.app/Contents/Resources/cli-plugins"

sudo ln -sf "$BIN/docker" /usr/local/bin/docker
sudo ln -sf "$PLUGINS/docker-compose" /usr/local/bin/docker-compose
sudo ln -sf "$BIN/docker-credential-desktop" /usr/local/bin/docker-credential-desktop
sudo ln -sf "$BIN/docker-credential-osxkeychain" /usr/local/bin/docker-credential-osxkeychain

echo "Symlinks fixed. Verifying..."
docker --version
docker compose version
echo "Done."
