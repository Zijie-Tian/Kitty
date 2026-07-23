"""CPU-only cache-aware PPL preflight resolver."""

from __future__ import annotations

import json

from kitty_sim.cli.eval_ppl import build_parser, finalize_args
from kitty_sim.ppl.runner import resolve_ppl_preflight


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    if not args.model:
        raise SystemExit("model is required")
    payload = resolve_ppl_preflight(args)
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))


if __name__ == "__main__":
    main()
