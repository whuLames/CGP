#!/usr/bin/env python3
"""Validate a transferred dataset bundle and atomically publish it to one GPU queue."""
import argparse
import hashlib
import json
import os
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
    args=parser.parse_args();root=args.root.resolve()
    if args.device<0:raise ValueError("device must be nonnegative")
    status_path=root/"status"/f"{args.dataset}.json"
    status=json.loads(status_path.read_text())
    if status.get("status")!="success":raise RuntimeError(f"dataset is not successful: {status_path}")
    if status.get("campaign_eligible") is False:raise RuntimeError(f"dataset is campaign-ineligible: {args.dataset}")
    fragment=root/"workloads"/args.dataset/"campaign_manifest_fragment.json"
    manifest=json.loads(fragment.read_text())
    if args.dataset not in manifest.get("datasets",{}):raise RuntimeError("manifest lacks base dataset")
    paths=[status_path,fragment]
    for name in ("csr_vlist.bin","csr_elist.bin","csr_weightlist.bin"):
        paths.append(root/"datasets"/args.dataset/name)
    for dataset in manifest["datasets"].values():
        paths.extend(Path(path) for path in dataset.get("workloads",{}).values())
    artifacts=[]
    for path in paths:
        path=path.resolve()
        try:path.relative_to(root)
        except ValueError:raise RuntimeError(f"artifact is outside campaign root: {path}")
        if not path.is_file():raise RuntimeError(f"missing artifact: {path}")
        artifacts.append({"path":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":digest(path)})
    receipt={"schema":1,"dataset":args.dataset,"device":args.device,
             "published_utc":datetime.now(timezone.utc).isoformat(),"artifacts":artifacts}
    ready=root/"inbox"/f"gpu{args.device}"/"ready";ready.mkdir(parents=True,exist_ok=True)
    target=ready/f"{args.dataset}.json";temporary=target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n");os.replace(temporary,target)
    print(target)


if __name__=="__main__":main()
