"""CPU-only NVIDIA RULER preflight resolver."""

from __future__ import annotations

import json

from kitty_sim.cli.eval_ruler import build_parser, finalize_args
from kitty_sim.ruler.runner import resolve_ruler_preflight


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    if not args.model:
        args.model = "preflight-placeholder"
    payload = resolve_ruler_preflight(args)
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))


if __name__ == "__main__":
    main()
