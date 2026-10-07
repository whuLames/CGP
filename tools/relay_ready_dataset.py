#!/usr/bin/env python3
"""Relay one prepared bundle between isolated hosts and publish it to a GPU queue."""
import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

GIB = 1 << 30


def endpoint(host, port, key):
    return ["ssh", "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-p", str(port), f"root@{host}"]


def remote(base, command, capture=True):
    return subprocess.run(base + [command], check=True, text=True,
                          capture_output=capture)


def source_ready(source, root, dataset, prepare_on_worker=False):
    status = f"{root}/status/{dataset}.json"
    if prepare_on_worker:
        validator = (
            "import json,sys; from pathlib import Path; "
            "s=json.load(open(sys.argv[1])); d=Path(sys.argv[2]); "
            "paths=[d/n for n in ('csr_vlist.bin','csr_elist.bin','csr_weightlist.bin')]; "
            "raise SystemExit(0 if s.get('status')=='success' and "
            "all(p.is_file() for p in paths) else 1)")
        command = ("python3 -c " + shlex.quote(validator) + f" {shlex.quote(status)} "
                   f"{shlex.quote(root + '/datasets/' + dataset)}")
        return subprocess.run(source + [command], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    fragment = f"{root}/workloads/{dataset}/campaign_manifest_fragment.json"
    validator = (
        "import json,sys; from pathlib import Path; "
        "s=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); "
        "paths=[]; "
        "[(paths.extend([Path(d['graph'])/n for n in "
        "('csr_vlist.bin','csr_elist.bin','csr_weightlist.bin')]), "
        "paths.extend(Path(p) for p in d.get('workloads',{}).values())) "
        "for d in m.get('datasets',{}).values()]; "
        "raise SystemExit(0 if s.get('status')=='success' and paths and "
        "all(p.is_file() for p in paths) else 1)")
    command = ("python3 -c " + shlex.quote(validator) + f" {shlex.quote(status)} "
               f"{shlex.quote(fragment)}")
    return subprocess.run(source + [command], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--source-port", required=True, type=int)
    parser.add_argument("--destination-host", required=True)
    parser.add_argument("--destination-port", required=True, type=int)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--source-root", default="/data/graphweft-prep-v1")
    parser.add_argument("--destination-root", default="/data/graphweft-adaptive-v1")
    parser.add_argument("--destination-repo", default="/data/coding/GraphWeft")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--prepare-on-worker", action="store_true")
    parser.add_argument("--queue-prefix", default="")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.dataset):
        raise ValueError("invalid dataset name")
    if args.device < 0:
        raise ValueError("device must be nonnegative")
    source = endpoint(args.source_host, args.source_port, args.key.resolve())
    destination = endpoint(args.destination_host, args.destination_port, args.key.resolve())
    while not source_ready(source, args.source_root, args.dataset, args.prepare_on_worker):
        if not args.wait:
            raise RuntimeError("source bundle is not ready")
        time.sleep(args.poll_seconds)

    relative = [f"datasets/{args.dataset}", f"status/{args.dataset}.json"]
    if not args.prepare_on_worker:relative.insert(1,f"workloads/{args.dataset}")
    quoted = " ".join(shlex.quote(value) for value in relative)
    byte_count = int(remote(source, f"cd {shlex.quote(args.source_root)} && "
                                   f"du -sb {quoted} | awk '{{s += $1}} END {{print s}}'").stdout)
    free = int(remote(destination, f"df -B1 --output=avail {shlex.quote(args.destination_root)} | tail -1").stdout)
    if free - byte_count < 30 * GIB:
        raise RuntimeError(f"destination would fall below 30 GiB reserve: free={free}, bundle={byte_count}")

    root = shlex.quote(args.destination_root)
    dataset = shlex.quote(args.dataset)
    workload_check = "" if args.prepare_on_worker else f"test ! -e {root}/workloads/{dataset} && "
    preflight = (f"test ! -e {root}/datasets/{dataset} && {workload_check}"
                 f"mkdir -p {root}/transfer_staging {root}/datasets {root}/workloads {root}/status")
    remote(destination, preflight, capture=False)
    stamp = int(time.time())
    stage = f"{args.destination_root}/transfer_staging/{args.dataset}-{stamp}"
    remote(destination, f"mkdir {shlex.quote(stage)}", capture=False)
    producer = subprocess.Popen(source + [f"tar -C {shlex.quote(args.source_root)} -cf - -- {quoted}"],
                                stdout=subprocess.PIPE)
    consumer = subprocess.run(destination + [f"tar -C {shlex.quote(stage)} -xf -"],
                              stdin=producer.stdout)
    assert producer.stdout is not None
    producer.stdout.close()
    producer_return = producer.wait()
    if producer_return or consumer.returncode:
        raise RuntimeError(f"relay failed: producer={producer_return}, consumer={consumer.returncode}; staging={stage}")

    if not args.prepare_on_worker:
        fragment = f"{stage}/workloads/{args.dataset}/campaign_manifest_fragment.json"
        rewrite = (f"sed -i {shlex.quote('s#' + args.source_root + '#' + args.destination_root + '#g')} "
                   f"{shlex.quote(fragment)}")
        remote(destination, rewrite, capture=False)
    moves = [f"mv {shlex.quote(stage + '/datasets/' + args.dataset)} {root}/datasets/",
             f"mv -f {shlex.quote(stage + '/status/' + args.dataset + '.json')} {root}/status/"]
    cleanup = [shlex.quote(stage + '/datasets'),shlex.quote(stage + '/status')]
    if not args.prepare_on_worker:
        moves.insert(1,f"mv {shlex.quote(stage + '/workloads/' + args.dataset)} {root}/workloads/")
        cleanup.append(shlex.quote(stage + '/workloads'))
    promote = " && ".join(moves+["rmdir " + " ".join(cleanup+[shlex.quote(stage)])])
    remote(destination, promote, capture=False)
    publish = (f"cd {shlex.quote(args.destination_repo)} && python3 tools/publish_ready_dataset.py "
               f"--root {root} --dataset {dataset} --device {args.device}" +
               (" --prepare-on-worker" if args.prepare_on_worker else "") +
               (f" --queue-prefix {shlex.quote(args.queue_prefix)}" if args.queue_prefix else ""))
    result = remote(destination, publish)
    print(json.dumps({"dataset": args.dataset, "device": args.device,
                      "bytes": byte_count, "receipt": result.stdout.strip()}, indent=2))


if __name__ == "__main__":
    main()
