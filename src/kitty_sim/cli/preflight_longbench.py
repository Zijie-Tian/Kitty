"""No-GPU LongBench preflight: resolve variant / method slug / semantic hash."""

from __future__ import annotations

import json

from kitty_sim.cli.eval_longbench import build_parser, finalize_args
from kitty_sim.longbench.runner import resolve_longbench_preflight


def main() -> None:
    parser = build_parser()
    # Allow invoking without a model positional for slug-only resolution.
    args = finalize_args(parser.parse_args())
    if not args.model:
        args.model = "preflight-placeholder"
    payload = resolve_longbench_preflight(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
