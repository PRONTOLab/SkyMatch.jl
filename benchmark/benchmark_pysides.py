#!/usr/bin/env python3
"""Run one filters-off pySIDES baseline point and write canonical JSON.

The timed region starts immediately before ``gen_sfr_props`` and ends after
``gen_CI`` has returned a complete host-resident pandas catalog.  Importing
packages, loading parameters, reading the first N CSV rows, and JSON writing
are intentionally outside the timed region.  The upstream checkout is used
without modifying its scientific source.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_SEED = 1234


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark the released pySIDES CPU implementation once."
    )
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=repo_root / "data" / "SIDES_Bethermin2017_short.csv",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=repo_root / "sides-public-release",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "benchmark" / "results",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional wall-clock limit for the timed model call.",
    )
    args = parser.parse_args()
    if args.rows <= 0:
        parser.error("--rows must be positive")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(directory: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ram_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def exact_command() -> str:
    return " ".join(shlex.quote(arg) for arg in [sys.executable, *sys.argv])


def base_record(args: argparse.Namespace) -> dict[str, Any]:
    upstream_commit = git_output(args.upstream_dir, "rev-parse", "HEAD")
    tracked_changes = git_output(
        args.upstream_dir, "status", "--porcelain", "--untracked-files=no"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation": "pySIDES",
        "backend": "CPU",
        "hardware_label": "pySIDES CPU (default multiprocessing)",
        "source_count": args.rows,
        "status": "running",
        "runtime_seconds": None,
        "unit": "s",
        "internal_estimator": "single time.perf_counter wall-time measurement",
        "internal_executions_or_samples": 1,
        "seed": args.seed,
        "filters_enabled": False,
        "output_written": False,
        "dataset": str(args.dataset.resolve()),
        "upstream_repository": "https://gitlab.lam.fr/mbethermin/sides-public-release.git",
        "upstream_commit": upstream_commit,
        "upstream_tracked_dirty": bool(tracked_changes),
        "exact_command": exact_command(),
        "timing_scope": (
            "initialized host catalog generation from gen_sfr_props through "
            "gen_CI; excludes imports, parameter/CSV loading, filters, and "
            "output writing; includes complete pandas host materialization"
        ),
        "worker_policy": "upstream multiprocessing.Pool(cpu_count()) default",
        "worker_count": multiprocessing.cpu_count(),
        "python_version": platform.python_version(),
        "dependency_versions": {
            name: package_version(name)
            for name in ("numpy", "pandas", "scipy", "astropy", "ipython")
        },
        "hardware": {
            "cpu_model": cpu_model(),
            "logical_cpu_threads": os.cpu_count(),
            "ram_bytes": ram_bytes(),
            "os": platform.platform(),
        },
        "recorded_at_utc": utc_now(),
        "completed_at_utc": None,
        "output_columns": None,
        "all_numeric_outputs_finite": None,
        "error": None,
    }


def output_path(args: argparse.Namespace) -> Path:
    return args.results_dir / f"pysides_forward_{args.rows}_CPUbenchmarks.json"


def save_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class ModelTimeout(TimeoutError):
    pass


def alarm_handler(_signum: int, _frame: Any) -> None:
    raise ModelTimeout("timed model call exceeded --timeout-seconds")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    record = base_record(args)
    destination = output_path(args)
    save_record(destination, record)

    dataset = args.dataset.resolve()
    pysides_dir = (args.upstream_dir / "PYSIDES").resolve()
    parameter_file = pysides_dir / "PAR_FILES" / "SIDES_from_original.par"
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset}")
    if not parameter_file.is_file():
        raise FileNotFoundError(f"pySIDES parameter file not found: {parameter_file}")

    old_cwd = Path.cwd()
    sys.path.insert(0, str(pysides_dir))
    try:
        os.chdir(pysides_dir)
        import numpy as np
        from pysides.gen_fluxes import gen_fluxes
        from pysides.gen_lines import gen_CI, gen_CII, gen_CO
        from pysides.gen_magnification import gen_magnification
        from pysides.gen_sfr_props import gen_sfr_props
        from pysides.load_params import load_params
        from pysides.load_sides_csv import load_sides_csv

        params = load_params(str(parameter_file))
        catalog = load_sides_csv(str(dataset), nrows=args.rows)
        if len(catalog) != args.rows:
            raise ValueError(
                f"requested {args.rows} rows but dataset supplied {len(catalog)}"
            )
        np.random.seed(args.seed)

        previous_handler = None
        if args.timeout_seconds is not None:
            previous_handler = signal.signal(signal.SIGALRM, alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
        try:
            start = time.perf_counter()
            catalog = gen_sfr_props(catalog, params)
            catalog = gen_magnification(catalog, params)
            catalog = gen_fluxes(catalog, params)
            # Filters are deliberately disabled for parity with BayesMMfwd.
            catalog = gen_CO(catalog, params)
            catalog = gen_CII(catalog, params)
            catalog = gen_CI(catalog, params)
            runtime_seconds = time.perf_counter() - start
        finally:
            if args.timeout_seconds is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)

        numeric = catalog.select_dtypes(include=[np.number])
        finite = bool(np.isfinite(numeric.to_numpy()).all())
        if not finite:
            raise ValueError("pySIDES returned non-finite numeric catalog values")

        record.update(
            status="complete",
            runtime_seconds=runtime_seconds,
            output_columns=len(catalog.columns),
            all_numeric_outputs_finite=finite,
            completed_at_utc=utc_now(),
        )
    except ModelTimeout as exception:
        record.update(
            status="timeout",
            error=str(exception),
            completed_at_utc=utc_now(),
        )
    except MemoryError as exception:
        record.update(
            status="oom",
            error=str(exception) or "Python MemoryError",
            completed_at_utc=utc_now(),
        )
    except Exception as exception:
        message = f"{type(exception).__name__}: {exception}"
        lower = message.lower()
        status = "oom" if "out of memory" in lower else "failed"
        record.update(status=status, error=message, completed_at_utc=utc_now())
    finally:
        os.chdir(old_cwd)
        save_record(destination, record)

    return record, destination


def main() -> int:
    args = parse_args()
    record, destination = run(args)
    print(
        "PYSIDES_BENCHMARK_RESULT="
        + json.dumps(
            {
                "path": str(destination),
                "source_count": record["source_count"],
                "status": record["status"],
                "runtime_seconds": record["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] in {"complete", "timeout", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
