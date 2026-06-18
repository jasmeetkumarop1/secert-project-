"""Automatic one-second self-review loop for the X agent project.

The reviewer is intentionally conservative: it only inspects this project's own
Python files, runs local checks, writes a JSON report, and can apply a tiny set
of deterministic safe fixes such as removing trailing whitespace. It does not
edit external data, model files, credentials, or arbitrary user files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

OWNED_PATHS = (Path("x_agent"), Path("tests"))
DEFAULT_REPORT = Path("data/autoreview_report.json")


@dataclass(frozen=True)
class CheckResult:
    """Result from one automatic review check."""

    name: str
    command: list[str]
    passed: bool
    output: str


@dataclass(frozen=True)
class ReviewReport:
    """Serializable automatic review report."""

    checked_at: float
    changed_files: list[str]
    fixes_applied: list[str]
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def iter_owned_python_files(root: Path = Path(".")) -> list[Path]:
    """Return Python files owned by this project, excluding caches."""
    files: list[Path] = []
    for owned_path in OWNED_PATHS:
        base = root / owned_path
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" not in path.parts:
                files.append(path)
    return sorted(files)


def snapshot_files(paths: Iterable[Path]) -> dict[str, int]:
    """Create a cheap mtime snapshot for change detection."""
    return {str(path): path.stat().st_mtime_ns for path in paths if path.exists()}


def apply_safe_fixes(paths: Iterable[Path]) -> list[str]:
    """Apply deterministic fixes that cannot change program behavior."""
    fixes: list[str] = []
    for path in paths:
        original = path.read_text(encoding="utf-8")
        fixed = "\n".join(line.rstrip() for line in original.splitlines())
        if original.endswith("\n"):
            fixed += "\n"
        if fixed != original:
            path.write_text(fixed, encoding="utf-8")
            fixes.append(f"trimmed trailing whitespace in {path}")
    return fixes


def run_check(name: str, command: list[str]) -> CheckResult:
    """Run a subprocess check and capture a compact result."""
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    output = (completed.stdout + completed.stderr).strip()
    return CheckResult(name=name, command=command, passed=completed.returncode == 0, output=output[-4_000:])


def run_review(
    *,
    apply_fixes: bool = False,
    report_path: Path = DEFAULT_REPORT,
    include_unit: bool = True,
) -> ReviewReport:
    """Run one self-review cycle and optionally apply safe fixes."""
    files = iter_owned_python_files()
    before = snapshot_files(files)
    fixes = apply_safe_fixes(files) if apply_fixes else []
    after = snapshot_files(files)
    changed_files = sorted(path for path, mtime in after.items() if before.get(path) != mtime)

    checks = [run_check("compile", ["python", "-m", "compileall", "x_agent", "tests"])]
    if include_unit:
        checks.append(run_check("unit", ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]))
    report = ReviewReport(
        checked_at=time.time(),
        changed_files=changed_files,
        fixes_applied=fixes,
        checks=checks,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "checked_at": report.checked_at,
                "passed": report.passed,
                "changed_files": report.changed_files,
                "fixes_applied": report.fixes_applied,
                "checks": [asdict(check) for check in report.checks],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report


def watch_reviews(*, interval: float = 1.0, apply_fixes: bool = False, report_path: Path = DEFAULT_REPORT) -> None:
    """Run self-review forever on a one-second loop."""
    print(f"Auto-review running every {interval:g}s. Press Ctrl+C to stop.")
    while True:
        started = time.monotonic()
        report = run_review(apply_fixes=apply_fixes, report_path=report_path)
        status = "PASS" if report.passed else "FAIL"
        print(f"{status}: {len(report.fixes_applied)} fixes, report={report_path}")
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, interval - elapsed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-review and safely fix this X agent project")
    parser.add_argument("--watch", action="store_true", help="Run review every interval forever")
    parser.add_argument("--interval", type=float, default=1.0, help="Review interval in seconds")
    parser.add_argument("--fix", action="store_true", help="Apply deterministic safe fixes to owned files")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="JSON report path")
    args = parser.parse_args()

    report_path = Path(args.report)
    if args.watch:
        watch_reviews(interval=args.interval, apply_fixes=args.fix, report_path=report_path)
        return

    report = run_review(apply_fixes=args.fix, report_path=report_path)
    print(json.dumps({"passed": report.passed, "report": str(report_path)}, indent=2))
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
