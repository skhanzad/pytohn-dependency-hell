#!/bin/bash
set -e

if [ -S /var/run/docker.sock ]; then
    if stat --version &>/dev/null; then
        SOCK_GID=$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo "0")
    else
        SOCK_GID=$(stat -f '%g' /var/run/docker.sock 2>/dev/null || echo "0")
    fi

    USER_GROUPS=$(id -G "$UNAME")

    if ! echo "$USER_GROUPS" | grep -q "\b$SOCK_GID\b"; then
        echo "Adding user to Docker socket group (GID: $SOCK_GID)"
        GROUP_NAME=$(getent group "$SOCK_GID" | cut -d: -f1 || echo "dockerhost")
        if [ "$GROUP_NAME" = "dockerhost" ]; then
            groupadd -g "$SOCK_GID" dockerhost 2>/dev/null || true
        fi
        usermod -aG "$GROUP_NAME" "$UNAME" 2>/dev/null || true
    fi
fi

exec gosu "$UNAME" "$@"
