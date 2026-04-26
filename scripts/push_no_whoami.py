"""Run `openenv push` without calling Hugging Face `whoami()` (avoids 429 on /whoami-v2).

Use only when the Hub is rate-limiting the profile endpoint but uploads still work.
You must pass ``-r user/repo``; the user part must match the account that owns the token.

Usage (from project root)::

    python scripts/push_no_whoami.py -r Soundyy45/Qstorespace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="openenv push with whoami() stubbed (429 workaround)",
    )
    parser.add_argument(
        "-r",
        "--repo-id",
        required=True,
        help="Hugging Face Space id, e.g. username/env-name",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="OpenEnv project root (default: current directory)",
    )
    args = parser.parse_args()

    if "/" not in args.repo_id or args.repo_id.count("/") != 1:
        print("Error: -r must be 'username/repo-name'", file=sys.stderr)
        return 1

    user_from_repo, _ = args.repo_id.split("/", 1)

    # Patch before any push() logic that imports whoami
    import openenv.cli.commands.push as openenv_pushmod

    def _fake_whoami(*a, **k):
        return {"name": user_from_repo, "fullname": user_from_repo, "type": "user"}

    openenv_pushmod.whoami = _fake_whoami

    from openenv.cli.commands.push import push

    env_dir = Path(args.directory).resolve()
    if not (env_dir / "openenv.yaml").exists():
        print(
            f"Error: {env_dir!s} is not the OpenEnv root (missing openenv.yaml).",
            file=sys.stderr,
        )
        return 1
    if env_dir != Path.cwd().resolve():
        # push() takes a str path; change cwd so defaults match openenv's expectations
        import os

        os.chdir(env_dir)

    try:
        push(
            directory=None,
            repo_id=args.repo_id,
            base_image=None,
            interface=None,
            no_interface=False,
            registry=None,
            private=False,
            create_pr=False,
            exclude=None,
        )
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
