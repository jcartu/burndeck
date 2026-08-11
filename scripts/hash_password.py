#!/usr/bin/env python3
"""Hash a Burndeck dashboard password (sha256) ready to paste into config.toml.

Usage:
    python scripts/hash_password.py                 # prompts via getpass (no echo)
    BURNDECK_PASSWORD=secret python scripts/hash_password.py

The password is never read from argv -- it would land in shell history and
`ps`. Output is one line that pastes straight into the [auth] section:

    password_sha256 = "2e69ac8896687b4d069531b4ddc77788fbdc9498b78f6f0bd92513fe19c53b7f"
"""
from __future__ import annotations

import hashlib
import getpass
import os
import sys


def _read_password() -> str:
    env = os.environ.get("BURNDECK_PASSWORD")
    if env is not None:
        return env
    if not sys.stdin.isatty():
        sys.exit("hash_password: no tty and BURNDECK_PASSWORD unset; "
                 "set the env var or run interactively")
    first = getpass.getpass("Dashboard password: ")
    if first != getpass.getpass("Confirm password: "):
        sys.exit("hash_password: passwords did not match")
    return first


def main() -> int:
    pw = _read_password()
    if not pw:
        sys.exit("hash_password: empty password")
    digest = hashlib.sha256(pw.encode("utf-8")).hexdigest()
    print(f'password_sha256 = "{digest}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
