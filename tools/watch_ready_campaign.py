#!/usr/bin/env python3
"""Run datasets as soon as an atomic READY receipt appears for this GPU."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):value.update(block)
    return value.hexdigest()


def validate(root,receipt,device):
    if receipt.get("schema")!=1 or receipt.get("device")!=device or not receipt.get("dataset"):
        raise RuntimeError("invalid READY receipt identity")
    if not receipt.get("artifacts"):raise RuntimeError("READY receipt has no artifacts")
    for artifact in receipt["artifacts"]:
        path=(root/artifact["path"]).resolve()
        try:path.relative_to(root)
        except ValueError:raise RuntimeError(f"artifact escapes campaign root: {path}")
        if not path.is_file() or path.stat().st_size!=artifact["bytes"] or digest(path)!=artifact["sha256"]:
            raise RuntimeError(f"artifact validation failed: {path}")


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--device",type=int,required=True);parser.add_argument("--poll-seconds",type=float,default=15)
    parser.add_argument("--once",action="store_true");args=parser.parse_args();root=args.root.resolve()
    queue=root/"inbox"/f"gpu{args.device}"
    ready=queue/"ready";running=queue/"running";done=queue/"done";failed=queue/"failed"
    for path in (ready,running,done,failed):path.mkdir(parents=True,exist_ok=True)
    while True:
        candidates=sorted(ready.glob("*.json"))
        if not candidates:
            if args.once:return 0
            time.sleep(args.poll_seconds);continue
        source=candidates[0];claimed=running/source.name
        try:os.replace(source,claimed)
        except FileNotFoundError:continue
        try:
            receipt=json.loads(claimed.read_text());validate(root,receipt,args.device)
            command=[sys.executable,str(PROJECT/"tools/run_adaptive_remote_pipeline.py"),
                     "--root",str(root),"--device",str(args.device),"--dataset",receipt["dataset"]]
            result=subprocess.run(command,cwd=PROJECT)
            if result.returncode:raise RuntimeError(f"campaign pipeline exited {result.returncode}")
            receipt["completed_utc"]=datetime.now(timezone.utc).isoformat()
            atomic_json(claimed,receipt);os.replace(claimed,done/claimed.name)
        except Exception as error:
            record={"failed_utc":datetime.now(timezone.utc).isoformat(),"error":repr(error),"receipt":str(claimed)}
            atomic_json(failed/(claimed.stem+".error.json"),record);os.replace(claimed,failed/claimed.name)
        if args.once:return 0


if __name__=="__main__":raise SystemExit(main())
