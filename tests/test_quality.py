# Requirements: NFR-QA-001, NFR-TRACE-001
"""Verify separate code-coverage gates and executed requirement evidence."""

import json
import runpy
import sys

import pytest

from scripts.quality import coverage_metrics, main, requirement_metrics


def artifacts(tmp_path):
    """Create linked requirement, test-result, and coverage fixtures."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text(
        "# Requirements: FR-DEMO-001\ndef test_ok():\n    assert True\n",
    )
    source = tmp_path / "module.py"
    source.write_text('def reached():\n    """Doc."""\n    return 1\n')
    coverage = {
        "totals": {
            "num_statements": 2,
            "covered_lines": 2,
            "num_branches": 0,
            "covered_branches": 0,
        },
        "files": {
            "module.py": {"executed_lines": [1, 3], "excluded_lines": []}
        },
    }
    (tmp_path / "coverage.json").write_text(json.dumps(coverage))
    (tmp_path / "requirements.md").write_text(
        '<!-- requirement: {"id":"FR-DEMO-001", "status":"implemented", '
        '"tests":["tests.test_demo.test_ok"]} -->\n'
        '<!-- requirement: {"id":"FR-FUTURE-001", "status":"planned", '
        '"tests":[]} -->\n',
    )
    (tmp_path / "test-results.xml").write_text(
        '<testsuites><testsuite><testcase classname="tests.test_demo" '
        'name="test_ok" /></testsuite></testsuites>',
    )
    return coverage


def test_independent_coverage_metrics(tmp_path):
    """Keep statement, branch, and function-entry thresholds independent."""
    data = artifacts(tmp_path)
    result = coverage_metrics(data, tmp_path)
    assert (
        result["statements"]
        == result["branches"]
        == result["functions"]
        == 100
    )
    (tmp_path / "module.py").write_text(
        "def reached():\n    return 1\ndef missed():\n    return 2\n",
    )
    data["files"]["module.py"]["executed_lines"] = [1, 2, 3]
    data["totals"].update(num_branches=2, covered_branches=1)
    result = coverage_metrics(data, tmp_path)
    assert result["functions"] == result["branches"] == 50
    assert result["missing_functions"] == ["module.py:missed:4"]
    data["files"]["module.py"]["excluded_lines"] = [4]
    assert coverage_metrics(data, tmp_path)["functions"] == 100


def test_requirements_use_passed_tests_and_suite_headers(tmp_path):
    """Require every parametrization to pass before verifying a requirement."""
    artifacts(tmp_path)
    report = requirement_metrics(
        tmp_path / "requirements.md", tmp_path / "test-results.xml", tmp_path
    )
    assert report["overall_percent"] == 50
    assert report["implemented_percent"] == 100
    assert report["planned"] == 1
    (tmp_path / "test-results.xml").write_text(
        '<testsuite><testcase classname="tests.test_demo" name="test_ok[a]" />'
        '<testcase classname="tests.test_demo" name="test_ok[b]"><failure />'
        "</testcase></testsuite>",
    )
    report = requirement_metrics(
        tmp_path / "requirements.md", tmp_path / "test-results.xml", tmp_path
    )
    assert report["implemented_percent"] == 0
    assert report["unverified"] == ["FR-DEMO-001"]


def test_missing_links_and_unknown_reverse_ids(tmp_path):
    """Report stale forward links and unregistered suite-header IDs."""
    artifacts(tmp_path)
    (tmp_path / "requirements.md").write_text(
        '<!-- requirement: {"id":"FR-DEMO-001", "status":"implemented", '
        '"tests":["tests.test_demo.test_missing"]} -->\n',
    )
    (tmp_path / "tests" / "test_demo.py").write_text(
        "# Requirements: UNKNOWN-001\ndef test_ok():\n    pass\n",
    )
    report = requirement_metrics(
        tmp_path / "requirements.md", tmp_path / "test-results.xml", tmp_path
    )
    assert report["unverified"] == ["FR-DEMO-001"]
    assert len(report["trace_errors"]) == 2


def test_multiline_requirement_headers(tmp_path):
    """Recognize requirement IDs split across consecutive header comments."""
    artifacts(tmp_path)
    (tmp_path / "requirements.md").write_text(
        '<!-- requirement: {"id":"FR-DEMO-001", "status":"implemented", '
        '"tests":["tests.test_demo.test_ok"]} -->\n'
        '<!-- requirement: {"id":"FR-DEMO-002", "status":"implemented", '
        '"tests":["tests.test_demo.test_ok"]} -->\n',
    )
    (tmp_path / "tests" / "test_demo.py").write_text(
        "# Requirements: FR-DEMO-001\n# Requirements: FR-DEMO-002\n"
        "def test_ok():\n    assert True\n",
    )
    report = requirement_metrics(
        tmp_path / "requirements.md", tmp_path / "test-results.xml", tmp_path
    )
    assert report["implemented_percent"] == 100
    assert not report["trace_errors"]


@pytest.mark.parametrize(
    "records",
    [
        "",
        '<!-- requirement: {"id":"A", "status":"wrong", "tests":[]} -->',
        '<!-- requirement: {"id":"A", "status":"implemented", "tests":[]} -->',
        '<!-- requirement: {"id":"A", "status":"planned", "tests":[]} -->\n'
        * 2,
    ],
)
def test_reject_invalid_register(tmp_path, records):
    """Reject absent, duplicate, or invalid requirement records."""
    artifacts(tmp_path)
    (tmp_path / "requirements.md").write_text(records)
    with pytest.raises(ValueError):
        requirement_metrics(
            tmp_path / "requirements.md",
            tmp_path / "test-results.xml",
            tmp_path,
        )


def test_quality_cli_success_failure_and_missing_input(
    tmp_path, monkeypatch, capsys
):
    """Distinguish passing gates, failing gates, and invalid input."""
    data = artifacts(tmp_path)
    monkeypatch.chdir(tmp_path)
    arguments = ["--requirements", "requirements.md"]
    assert main(arguments) == 0
    assert json.loads((tmp_path / "quality-report.json").read_text())["passed"]
    data["totals"].update(num_statements=100, covered_lines=95)
    (tmp_path / "coverage.json").write_text(json.dumps(data))
    assert main(arguments) == 1
    assert main(["--coverage", "missing.json"]) == 2
    assert "error:" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["quality", *arguments])
    monkeypatch.delitem(sys.modules, "scripts.quality", raising=False)
    with pytest.raises(SystemExit) as result:
        runpy.run_module("scripts.quality", run_name="__main__")
    assert result.value.code == 1
