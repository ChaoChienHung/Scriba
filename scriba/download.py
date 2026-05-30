from __future__ import annotations

import argparse
from typing import Optional

from .model_store import ensure_pretrained_saved


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scriba.download")
    p.add_argument("--arch", required=True, choices=["trocr", "donut"])
    p.add_argument("--pretrained", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--set-latest", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    dst = ensure_pretrained_saved(
        arch=args.arch,
        pretrained_id=args.pretrained,
        name=args.name,
        set_latest=args.set_latest,
    )
    print(dst)


if __name__ == "__main__":
    main()

