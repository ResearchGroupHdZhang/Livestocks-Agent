import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "weighted_4_2_1_cn_restart"
SEED = RESULTS / "cn.seed.json"
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


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "code"),
        "LIVESTOCK_DATA_ROOT": str(ROOT.parents[1] / "data"),
        "WEIGHTED_OUTPUT_DIR": str(RESULTS),
        "WEIGHTED_WARM_START": str(SEED),
        "WEIGHTED_CONCURRENT_EXACT_PRIMARY": "true",
    }
    if not SEED.exists():
        seed_process = subprocess.run(
            [
                sys.executable,
                str(ROOT / "code" / "build_weighted_seed.py"),
                "cn",
                "--output",
                str(SEED),
            ],
            env=environment,
            stdout=(RESULTS / "cn.seed.stdout.log").open("w", encoding="utf-8"),
            stderr=(RESULTS / "cn.seed.stderr.log").open("w", encoding="utf-8"),
        )
        if seed_process.returncode:
            raise SystemExit(seed_process.returncode)
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    if seed.get("weighted_objective", 0) <= 0:
        raise RuntimeError(f"non-positive China seed: {seed}")

    started_at = datetime.now().astimezone().isoformat()
    started = monotonic()
    with (RESULTS / "cn.stdout.log").open("w", encoding="utf-8") as stdout, (
        RESULTS / "cn.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "code" / "run_weighted_4_2_1.py"), "cn"],
            stdout=stdout,
            stderr=stderr,
            env=environment,
        )
        (RESULTS / "current-run.json").write_text(
            json.dumps(
                {
                    "country": "cn",
                    "attempt": "nonzero_full_warm_start",
                    "pid": process.pid,
                    "started_at": started_at,
                    "stop_rule": "relative_mip_gap",
                    "target_gap": TARGET_GAP,
                    "time_limit": None,
                    "external_timeout": None,
                    "seed": str(SEED),
                    "seed_weighted_objective": seed["weighted_objective"],
                    "current_log": "cn_weighted.log",
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
                        "country": "cn",
                        "attempt": "nonzero_full_warm_start",
                        "started_at": started_at,
                        "elapsed_seconds": monotonic() - started,
                        "seed_weighted_objective": seed["weighted_objective"],
                        "current_log": "cn_weighted.log",
                        "target_gap": TARGET_GAP,
                        "time_limit": None,
                        **process_status(process.pid),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            sleep(60)
    (RESULTS / "terminal-status.json").write_text(
        json.dumps(
            {
                "country": "cn",
                "returncode": process.returncode,
                "started_at": started_at,
                "completed_at": datetime.now().astimezone().isoformat(),
                "total_elapsed_seconds": monotonic() - started,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
