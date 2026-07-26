#!/usr/bin/env python3

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORRECTNESS = ROOT / "artifacts" / "correctness"
BIN = CORRECTNESS / "bin"
GR_ROOT = Path("/home/zyl/data/ggr_data/singlegpu")
FORK_ROOT = Path("/home/zyl/Projects/ocgp/baselines/ForkGraph")
GLIGN_ROOT = Path("/home/zyl/Projects/ocgp/baselines/Glign")
DATASETS = ("cit-Patents", "soc-orkut", "soc-twitter", "soc-sinaweibo")
SIGNATURE = re.compile(
    r"validation_summary query=0 reached=(\d+) "
    r"value_sum=(\d+) weighted_sum=(\d+)"
)


def parse_signature(text: str) -> tuple[int, int, int]:
    matches = SIGNATURE.findall(text)
    if len(matches) != 1:
        raise RuntimeError(f"expected one validation summary, found {len(matches)}")
    return tuple(int(value) for value in matches[0])


def run(command: list[str], cwd: Path, log: Path) -> tuple[int, int, int]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "96"
    env["OMP_DYNAMIC"] = "FALSE"
    start = time.monotonic()
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=7200,
    )
    log.write_text(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f"exit code {process.returncode}: {' '.join(command)}")
    signature = parse_signature(process.stdout)
    print(
        f"validated {log.stem} wall_s={time.monotonic() - start:.3f} "
        f"signature={signature}",
        flush=True,
    )
    return signature


def main() -> None:
    output = CORRECTNESS / "weighted_cross_system"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []

    for dataset in DATASETS:
        source = (ROOT / "query_sets" / dataset / "sources.txt").read_text().splitlines()[0]
        query_file = output / f"{dataset}-q1.sources"
        query_file.write_text(source + "\n")

        puer = {}
        for algorithm in ("sssp", "sswp"):
            log = CORRECTNESS / "puercgp" / f"{dataset}-{algorithm}.log"
            puer[algorithm] = parse_signature(log.read_text())

        glign = {}
        for algorithm in ("sssp", "sswp"):
            executable = BIN / f"glign_{algorithm}_validate"
            log = output / f"{dataset}-glign-{algorithm}.log"
            command = [
                str(executable),
                "-option",
                "glign",
                "-mode",
                "3",
                "-delay",
                "-gr",
                "-batch",
                "1",
                "-max_combination",
                "1",
                "-qf",
                str(query_file),
                str(GR_ROOT / f"{dataset}.gr"),
            ]
            glign[algorithm] = run(command, GLIGN_ROOT / "apps", log)

        fork_log = output / f"{dataset}-forkgraph-sssp.log"
        data = FORK_ROOT / "inputs_gr_partitioned"
        fork = run(
            [
                str(BIN / "forkgraph_sssp_validate"),
                "-p",
                "8",
                "-gr",
                str(query_file),
                str(data / f"{dataset}.intra.gr"),
                str(data / f"{dataset}.inter.gr"),
                str(data / f"{dataset}.part"),
            ],
            FORK_ROOT,
            fork_log,
        )

        signatures = {
            "sssp": {"puercgp": puer["sssp"], "glign": glign["sssp"], "forkgraph": fork},
            "sswp": {"puercgp": puer["sswp"], "glign": glign["sswp"]},
        }
        for algorithm, systems in signatures.items():
            expected = next(iter(systems.values()))
            for system, signature in systems.items():
                correct = signature == expected
                rows.append(
                    {
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "system": system,
                        "source": source,
                        "reached": signature[0],
                        "value_sum": signature[1],
                        "weighted_sum": signature[2],
                        "correct": int(correct),
                    }
                )
                if not correct:
                    failures.append(f"{dataset}/{algorithm}/{system}")

    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    status = {
        "state": "complete" if not failures else "failed",
        "checks": len(rows),
        "failures": failures,
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
