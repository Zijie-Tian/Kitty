from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_LOCAL_PATHS = (
    "/mnt/" + "data/tzj",
    "/home/" + "zijie",
    "/home/" + "tzj",
)
SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".omx",
    "third_party",
    "logs",
    "longbench_out",
    "eval_logs_gsm8k_gpu1",
    "eval_results_gsm8k_gpu1",
    "eval_results_smoke_gpu1",
}


def _candidate_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=REPO_ROOT,
        text=True,
    )
    return [REPO_ROOT / line for line in output.splitlines() if line]


class NoHardcodedPathsTests(unittest.TestCase):
    def test_tracked_and_unignored_files_do_not_embed_host_local_paths(self) -> None:
        offenders: list[str] = []
        for path in _candidate_files():
            rel = path.relative_to(REPO_ROOT)
            if any(part in SKIP_PARTS for part in rel.parts):
                continue
            if path.name == ".env" or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for forbidden in FORBIDDEN_LOCAL_PATHS:
                if forbidden in text:
                    offenders.append(f"{rel}: contains {forbidden}")

        self.assertFalse(
            offenders,
            "Host-local paths must live in ignored .env only:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
