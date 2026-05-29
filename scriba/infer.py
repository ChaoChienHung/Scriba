from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .inference import run_inference


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scriba.infer")
    p.add_argument("--arch", required=True, choices=["trocr", "donut"])
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--pretrained", default=None)

    p.add_argument("--image", default=None)
    p.add_argument("--images-dir", default=None)
    p.add_argument("--glob", default="*.jpg")
    p.add_argument("--output", default=None)

    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-beams", type=int, default=1)
    return p


def _iter_images(images_dir: Path, pattern: str) -> list[Path]:
    return sorted(images_dir.glob(pattern))


def _serialize(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.image is None and args.images_dir is None:
        raise SystemExit("Provide --image or --images-dir")
    if args.image is not None and args.images_dir is not None:
        raise SystemExit("Use only one of --image or --images-dir")

    out_fp = Path(args.output) if args.output else None
    handle = out_fp.open("w", encoding="utf-8") if out_fp else None

    try:
        if args.image is not None:
            img_path = Path(args.image)
            image = Image.open(img_path)
            res = run_inference(
                arch=args.arch,
                image=image,
                pretrained=args.pretrained,
                checkpoint_dir=args.checkpoint_dir,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            payload = {
                "image": str(img_path),
                "arch": res.arch,
                "latency_ms": res.latency_ms,
                "output": res.output,
            }
            line = _serialize(payload)
            if handle:
                handle.write(line + "\n")
            else:
                print(line)
            return

        images_dir = Path(args.images_dir)
        for img_path in _iter_images(images_dir, args.glob):
            image = Image.open(img_path)
            res = run_inference(
                arch=args.arch,
                image=image,
                pretrained=args.pretrained,
                checkpoint_dir=args.checkpoint_dir,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            payload = {
                "image": str(img_path),
                "arch": res.arch,
                "latency_ms": res.latency_ms,
                "output": res.output,
            }
            line = _serialize(payload)
            if handle:
                handle.write(line + "\n")
            else:
                print(line)
    finally:
        if handle:
            handle.close()


if __name__ == "__main__":
    main()

