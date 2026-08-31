import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "weighted_4_2_1"
COUNTRIES = ("aus", "cn", "usa", "br")
TARGET_GAP = 1e-3


def process_status(pid):
    status = {"pid": pid}
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        status["rss_mb"] = int(
            next(line.split()[1] for line in text.splitlines() if line.startswith("VmRSS:"))
        ) / 1024
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        status["cpu_ticks"] = int(stat[13]) + int(stat[14])
    except (FileNotFoundError, StopIteration, IndexError, ValueError):
        pass
    return status


def exact_primary_active():
    for command in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if "primary_exact.py" in command.read_bytes().replace(b"\0", b" ").decode(errors="ignore"):
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return False


def certified(country):
    summary_path = RESULTS / f"{country}.json"
    validation_path = RESULTS / f"{country}.validation.json"
    if not summary_path.exists() or not validation_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    solve = summary.get("solve", {})
    return bool(
        solve.get("certified")
        and solve.get("solutions", 0) > 0
        and solve.get("gap", float("inf")) <= TARGET_GAP
        and validation.get("all_passed")
    )


def archive_interrupted(country):
    current = RESULTS / "current-run.json"
    if not current.exists():
        return
    state = json.loads(current.read_text(encoding="utf-8"))
    if state.get("country") != country:
        return
    pid = state.get("pid")
    if isinstance(pid, int) and Path(f"/proc/{pid}").exists():
        return
    attempts = RESULTS / "interrupted_attempts"
    attempts.mkdir(exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    metadata = {
        **state,
        "interrupted_at": datetime.now().astimezone().isoformat(),
        "reason": "runner_restarted_or_host_reboot",
    }
    (attempts / f"{country}_{stamp}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(country, arguments, stdout_name, stderr_name, environment):
    with (RESULTS / stdout_name).open("w", encoding="utf-8") as stdout, (
        RESULTS / stderr_name
    ).open("w", encoding="utf-8") as stderr:
        return subprocess.run(arguments, stdout=stdout, stderr=stderr, env=environment).returncode


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    batch_started = monotonic()
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "code"),
        "LIVESTOCK_DATA_ROOT": str(ROOT.parents[1] / "data"),
        "WEIGHTED_OUTPUT_DIR": str(RESULTS),
    }
    for country in COUNTRIES:
        if certified(country):
            continue
        archive_interrupted(country)

        preflight_code = run(
            country,
            [sys.executable, str(ROOT / "code" / "run_weighted_4_2_1.py"), country, "--preflight"],
            f"{country}.preflight.stdout.log",
            f"{country}.preflight.stderr.log",
            environment,
        )
        if preflight_code:
            continue

        country_started = monotonic()
        concurrent = exact_primary_active()
        solve_environment = {
            **environment,
            "WEIGHTED_CONCURRENT_EXACT_PRIMARY": str(concurrent).lower(),
        }
        with (RESULTS / f"{country}.stdout.log").open("w", encoding="utf-8") as stdout, (
            RESULTS / f"{country}.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "code" / "run_weighted_4_2_1.py"), country],
                stdout=stdout,
                stderr=stderr,
                env=solve_environment,
            )
            started_at = datetime.now().astimezone().isoformat()
            (RESULTS / "current-run.json").write_text(
                json.dumps(
                    {
                        "country": country,
                        "pid": process.pid,
                        "started_at": started_at,
                        "stop_rule": "relative_mip_gap",
                        "target_gap": TARGET_GAP,
                        "time_limit": None,
                        "external_timeout": None,
                        "current_log": f"{country}_weighted.log",
                        "concurrent_with_exact_primary_at_start": concurrent,
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            while process.poll() is None:
                (RESULTS / "current-status.json").write_text(
                    json.dumps(
                        {
                            "country": country,
                            "started_at": started_at,
                            "elapsed_seconds": monotonic() - country_started,
                            "current_log": f"{country}_weighted.log",
                            "stop_rule": "relative_mip_gap",
                            "target_gap": TARGET_GAP,
                            "time_limit": None,
                            "concurrent_with_exact_primary_now": exact_primary_active(),
                            **process_status(process.pid),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                sleep(60)
        if process.returncode:
            continue

        validation_code = run(
            country,
            [sys.executable, str(ROOT / "code" / "verify_weighted_4_2_1.py"), country],
            f"{country}.validation.stdout.log",
            f"{country}.validation.stderr.log",
            solve_environment,
        )
        if validation_code or not certified(country):
            continue

    (RESULTS / "batch-timing.json").write_text(
        json.dumps(
            {
                "countries": COUNTRIES,
                "target_gap": TARGET_GAP,
                "time_limit": None,
                "total_elapsed_seconds": monotonic() - batch_started,
                "completed_at": datetime.now().astimezone().isoformat(),
                "certified": {country: certified(country) for country in COUNTRIES},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
