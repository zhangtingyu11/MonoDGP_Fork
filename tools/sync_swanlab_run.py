#!/usr/bin/env python3
"""Sync a SwanLab run with an explicit API host and login timeout."""

from __future__ import annotations

import argparse
from pathlib import Path

from swanlab.sdk.cmd.login import login_raw
from swanlab.sdk.cmd.sync import sync
from swanlab.sdk.internal.settings import Settings
from swanlab.sdk.internal.settings import settings as global_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--host", default="https://api.swanlab.cn")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    if global_settings.api_key is None:
        raise RuntimeError("no stored SwanLab API key is available")
    login_raw(
        api_key=global_settings.api_key,
        host=args.host,
        save=False,
        timeout=args.timeout,
        animation=False,
        print_welcome=False,
    )
    settings = Settings.model_validate(
        {
            "api_key": global_settings.api_key,
            "api_host": args.host,
            "web_host": "https://swanlab.cn",
            "run": {"id": args.id},
            "core": {"record_batch": args.batch_size},
        }
    )
    sync(args.run_dir, settings=settings)


if __name__ == "__main__":
    main()
