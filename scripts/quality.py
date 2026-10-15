"""Report coverage gates and bidirectional requirement traceability."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def percentage(covered: int, total: int) -> float:
    """Convert covered and total counts, treating an empty domain as covered."""
    return 100.0 * covered / total if total else 100.0


def coverage_metrics(data: dict[str, Any], root: Path) -> dict[str, Any]:
    """Measure statement, branch, and named-function entry coverage."""
    total_functions = covered_functions = 0
    missing: list[str] = []
    for filename, detail in data["files"].items():
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        executed = set(detail["executed_lines"])
        excluded = set(detail["excluded_lines"])
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = node.body
            if (
                isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if not body or body[0].lineno in excluded:
                continue
            total_functions += 1
            if body[0].lineno in executed:
                covered_functions += 1
            else:
                missing.append(f"{filename}:{node.name}:{body[0].lineno}")
    totals = data["totals"]
    return {
        "statements": percentage(
            totals["covered_lines"], totals["num_statements"]
        ),
        "branches": percentage(
            totals["covered_branches"], totals["num_branches"]
        ),
        "functions": percentage(covered_functions, total_functions),
        "function_count": total_functions,
        "missing_functions": missing,
    }


def requirement_metrics(
    document: Path, junit: Path, root: Path
) -> dict[str, Any]:
    """Reconcile requirement records, test headers, and passing JUnit cases."""
    records: dict[str, Any] = {}
    marker = "<!-- requirement: "
    for line in document.read_text(encoding="utf-8").splitlines():
        if line.startswith(marker) and line.endswith(" -->"):
            record_text = line.removeprefix(marker).removesuffix(" -->")
            record = json.loads(record_text)
            identifier = record["id"]
            status_is_valid = record["status"] in ("implemented", "planned")
            if identifier in records or not status_is_valid:
                raise ValueError("Duplicate requirement or invalid status")
            if record["status"] == "implemented" and not record["tests"]:
                raise ValueError("Implemented requirements must name tests")
            records[identifier] = record
    if not records:
        raise ValueError("No requirements registered")

    results: dict[str, bool] = {}
    for case in ET.parse(junit).iter("testcase"):
        name = (
            f"{case.get('classname', '')}.{case.get('name', '').split('[')[0]}"
        )
        passed = not any(
            child.tag in ("failure", "error", "skipped") for child in case
        )
        results[name] = results.get(name, True) and passed

    headers: dict[str, set[str]] = {}
    trace_errors: list[str] = []
    for file in (root / "tests").rglob("test_*.py"):
        module = ".".join(file.relative_to(root).with_suffix("").parts)
        lines = file.read_text(encoding="utf-8").splitlines()
        ids: set[str] = set()
        for line in lines:
            if not line.startswith("# Requirements: "):
                break
            ids.update(line.removeprefix("# Requirements: ").split(", "))
        headers[module] = ids
        for identifier in ids:
            if identifier not in records:
                trace_errors.append(
                    f"{module}: unknown requirement {identifier}"
                )
            elif not any(
                selector.rsplit(".", 1)[0] == module
                for selector in records[identifier]["tests"]
            ):
                trace_errors.append(
                    f"{module}: no forward link for {identifier}"
                )

    unverified: list[str] = []
    implemented = 0
    for identifier, record in records.items():
        if record["status"] == "planned":
            continue
        implemented += 1
        verified = True
        for selector in record["tests"]:
            matches = [
                passed
                for name, passed in results.items()
                if fnmatch.fnmatchcase(name, selector)
            ]
            module = selector.rsplit(".", 1)[0]
            if identifier not in headers.get(module, set()):
                trace_errors.append(
                    f"{module}: missing header ID {identifier}"
                )
                verified = False
            if not matches or not all(matches):
                verified = False
        if not verified:
            unverified.append(identifier)
    covered = implemented - len(unverified)
    return {
        "total": len(records),
        "implemented": implemented,
        "planned": len(records) - implemented,
        "verified": covered,
        "overall_percent": percentage(covered, len(records)),
        "implemented_percent": percentage(covered, implemented),
        "unverified": unverified,
        "trace_errors": trace_errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write a reproducible quality report and enforce delivery thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--coverage", default="coverage.json")
    parser.add_argument("--junit", default="test-results.xml")
    parser.add_argument("--requirements", default="doc/requirements.md")
    parser.add_argument("--output", default="quality-report.json")
    parser.add_argument("--threshold", type=float, default=95)
    arguments = parser.parse_args(argv)
    root = Path(arguments.root).resolve()
    try:
        coverage = coverage_metrics(
            json.loads(
                (root / arguments.coverage).read_text(encoding="utf-8"),
            ),
            root,
        )
        requirements = requirement_metrics(
            root / arguments.requirements, root / arguments.junit, root
        )
        passed = (
            all(
                coverage[key] > arguments.threshold
                for key in ("statements", "branches", "functions")
            )
            and requirements["implemented_percent"] == 100
            and not requirements["trace_errors"]
        )
        report = {
            "coverage": coverage,
            "requirements": requirements,
            "threshold_exclusive": arguments.threshold,
            "passed": passed,
        }
        text = json.dumps(report, indent=2)
        (root / arguments.output).write_text(text, encoding="utf-8")
        print(text)
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, ET.ParseError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
