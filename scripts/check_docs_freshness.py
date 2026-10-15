"""Script to check that documentation is updated when
behavior-sensitive code changes.
This script is intended to be used in CI to ensure that when code related to
the CLI, engines, or SGF import/export is modified, the relevant documentation
is also updated to reflect those changes. It can be run with a list of changed
file paths or with a file containing those paths.
Usage:
    # Check with a list of changed paths
    python check_docs_freshness.py path/to/changed_file1.py \
        path/to/changed_file2.py
    # Check with a file containing changed paths
    python check_docs_freshness.py --from-file changed_files.txt
"""
from __future__ import annotations

import argparse
import fnmatch
import pathlib

CODE_PATTERNS = (
    "four_in_a_row/cli.py",
    "four_in_a_row/engines.py",
    "four_in_a_row/sgf.py",
    "four_in_a_row/ai_*.py",
)

DOC_PATHS = {
    "README.md",
    "doc/index.md",
    "doc/software_architecture.md",
    "doc/engines.md",
    "doc/engine_random.md",
    "doc/engine_minimax.md",
    "doc/engine_negamax.md",
    "doc/engine_alphabeta.md",
    "doc/engine_mcts.md",
    "doc/sgf.md",
    "doc/docs_freshness.md",
}


def _normalize(path: str) -> str:
    return path.strip().replace("\\", "/")


def _is_relevant_code_change(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in CODE_PATTERNS)


def _load_changed_files(from_file: str | None, paths: list[str]) -> list[str]:
    if from_file is not None:
        content = pathlib.Path(from_file).read_text(encoding="utf-8")
        return [_normalize(line) for line in content.splitlines()
                if line.strip()]
    return [_normalize(path) for path in paths if path.strip()]


def main() -> int:
    """Main function to run the docs freshness check."""
    parser = argparse.ArgumentParser(
        description="Check that docs are updated when "
        "behavior-sensitive code changes."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Changed file paths. Omit when using --from-file.",
    )
    parser.add_argument(
        "--from-file",
        dest="from_file",
        help="Path to a file containing one changed path per line.",
    )
    args = parser.parse_args()

    changed_files = _load_changed_files(args.from_file, args.paths)

    relevant_code_changes = sorted(
        path for path in changed_files if _is_relevant_code_change(path)
    )
    docs_updates = sorted(path for path in changed_files if path in DOC_PATHS)

    if not relevant_code_changes:
        print("Docs freshness check skipped: "
              "no CLI/engine/SGF code changes detected.")
        return 0

    if docs_updates:
        print("Docs freshness check passed.")
        print("Relevant code changes:")
        for path in relevant_code_changes:
            print(f"- {path}")
        print("Documentation updates:")
        for path in docs_updates:
            print(f"- {path}")
        return 0

    print("Docs freshness check failed.")
    print("The following behavior-sensitive files changed:")
    for path in relevant_code_changes:
        print(f"- {path}")
    print("Update at least one of:")
    for path in sorted(DOC_PATHS):
        print(f"- {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
