#!/usr/bin/env python3
"""Validate a transferred dataset bundle and atomically publish it to one GPU queue."""
import argparse
import hashlib
import json
import os
import re
from datetime import datetime,timezone
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--dataset",required=True);parser.add_argument("--device",type=int,required=True)
    parser.add_argument("--prepare-on-worker",action="store_true",
                        help="publish a converted graph before its workload bundle exists")
    parser.add_argument("--queue-prefix",default="",
                        help="optional filename prefix used to prioritize a READY receipt")
    args=parser.parse_args();root=args.root.resolve()
    if args.device<0:raise ValueError("device must be nonnegative")
    if args.queue_prefix and not re.fullmatch(r"[A-Za-z0-9_.-]+",args.queue_prefix):
        raise ValueError("invalid queue prefix")
    status_path=root/"status"/f"{args.dataset}.json"
    status=json.loads(status_path.read_text())
    if status.get("status")!="success":raise RuntimeError(f"dataset is not successful: {status_path}")
    if status.get("campaign_eligible") is False:raise RuntimeError(f"dataset is campaign-ineligible: {args.dataset}")
    fragment=root/"workloads"/args.dataset/"campaign_manifest_fragment.json"
    paths=[status_path]
    if fragment.is_file():
        manifest=json.loads(fragment.read_text())
        if args.dataset not in manifest.get("datasets",{}):raise RuntimeError("manifest lacks base dataset")
        paths.append(fragment)
        for dataset in manifest["datasets"].values():
            graph=Path(dataset["graph"])
            for name in ("csr_vlist.bin","csr_elist.bin","csr_weightlist.bin"):
                paths.append(graph/name)
            for name in ("conversion_manifest.json","augmentation_manifest.json"):
                if (graph/name).is_file():paths.append(graph/name)
            paths.extend(Path(path) for path in dataset.get("workloads",{}).values())
    elif args.prepare_on_worker:
        graph=root/"datasets"/args.dataset
        paths.extend(graph/name for name in ("csr_vlist.bin","csr_elist.bin","csr_weightlist.bin"))
        for name in ("conversion_manifest.json","external_vertex_ids.bin"):
            if (graph/name).is_file():paths.append(graph/name)
    else:
        raise RuntimeError(f"workload bundle is not ready: {fragment}")
    artifacts=[];seen=set()
    for path in paths:
        path=path.resolve()
        if path in seen:continue
        seen.add(path)
        try:path.relative_to(root)
        except ValueError:raise RuntimeError(f"artifact is outside campaign root: {path}")
        if not path.is_file():raise RuntimeError(f"missing artifact: {path}")
        artifacts.append({"path":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":digest(path)})
    receipt={"schema":1,"dataset":args.dataset,"device":args.device,
             "prepare_on_worker":not fragment.is_file(),
             "published_utc":datetime.now(timezone.utc).isoformat(),"artifacts":artifacts}
    ready=root/"inbox"/f"gpu{args.device}"/"ready";ready.mkdir(parents=True,exist_ok=True)
    stem=f"{args.queue_prefix}-{args.dataset}" if args.queue_prefix else args.dataset
    target=ready/f"{stem}.json";temporary=target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n");os.replace(temporary,target)
    print(target)


if __name__=="__main__":main()
