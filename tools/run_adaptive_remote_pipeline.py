#!/usr/bin/env python3
"""Continue prepared datasets through workloads and the complete GPU campaign."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
FULL_ORACLE_DATASETS={"soc-orkut","uk-2002","graph500-scale23-ef16-adj","delaunay-n24"}


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def run(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a") as output:
        output.write("\nCOMMAND "+" ".join(command)+"\n");output.flush()
        result=subprocess.run(command,cwd=PROJECT,stdout=output,stderr=subprocess.STDOUT,text=True)
    return result.returncode


def wait_dataset(root,name,deadline,status_log):
    path=root/"status"/f"{name}.json"
    while time.time()<deadline:
        paused=root/"PAUSED_INSUFFICIENT_SPACE.json"
        if paused.exists():return "paused",json.loads(paused.read_text())
        if path.exists():
            try:record=json.loads(path.read_text())
            except json.JSONDecodeError:record={}
            if record.get("status") in {"success","failed","paused","paused_insufficient_space"}:
                return record.get("status"),record
        time.sleep(30)
    return "timeout",{"dataset":name,"deadline":deadline}


def jobs(fragment,name,device):
    fit=fragment["fits"];result=[f"{name}:ordinary_m128:128:none:{device}"]
    tail=name+"-longtail"
    result += [f"{tail}:long_tail_m128:128:{policy}:{device}" for policy in ("none","eager_global")]
    if fit["base_m256"]:result.append(f"{name}:ordinary_m256:256:none:{device}")
    if fit["derived_m256"]:result += [f"{tail}:long_tail_m256:256:{policy}:{device}"
                                       for policy in ("none","eager_global")]
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--device",type=int,required=True);parser.add_argument("--dataset",action="append",required=True)
    parser.add_argument("--wait-hours",type=float,default=168);args=parser.parse_args();root=args.root.resolve()
    pipeline=root/"pipeline";pipeline.mkdir(parents=True,exist_ok=True);state_path=pipeline/f"gpu{args.device}.json"
    lease=(pipeline/f"gpu{args.device}.lock").open("a+")
    fcntl.flock(lease,fcntl.LOCK_EX)
    state={"status":"running","device":args.device,"datasets":args.dataset,
           "started_utc":datetime.now(timezone.utc).isoformat(),"completed":[],"skipped":[]};atomic_json(state_path,state)
    deadline=time.time()+args.wait_hours*3600
    for name in args.dataset:
        status,record=wait_dataset(root,name,deadline,pipeline/f"gpu{args.device}.log")
        if status=="paused":state["status"]="paused_insufficient_space";state["pause"]=record;atomic_json(state_path,state);return 2
        if status!="success":state["skipped"].append({"dataset":name,"reason":status});atomic_json(state_path,state);continue
        if record.get("duplicate_of"):
            state["skipped"].append({"dataset":name,"reason":"canonical_duplicate","of":record["duplicate_of"]});atomic_json(state_path,state);continue
        if not record.get("fits_v100_32g_80pct",{}).get("128",False):
            state["skipped"].append({"dataset":name,"reason":"M128_memory_gate"});atomic_json(state_path,state);continue
        bundle=root/"workloads"/name;fragment=bundle/"campaign_manifest_fragment.json"
        if not fragment.exists():
            code=run([sys.executable,str(PROJECT/"tools/prepare_adaptive_workload_bundle.py"),f"--name={name}",
                f"--graph={root/'datasets'/name}",f"--output={bundle}",f"--device={args.device}"],pipeline/f"{name}.log")
            if code:
                state["skipped"].append({"dataset":name,"reason":"workload_preparation_failed","exit":code});atomic_json(state_path,state);continue
        manifest=json.loads(fragment.read_text());oracle_profile="core" if name in FULL_ORACLE_DATASETS else "paired"
        state["oracle_profile"]={"dataset":name,"mode":oracle_profile};atomic_json(state_path,state)
        for job in jobs(manifest,name,args.device):
            code=run([sys.executable,str(PROJECT/"tools/run_round_oracle_campaign.py"),f"--manifest={fragment}",
                      f"--output={root/'results'}",f"--job={job}",f"--oracle-profile={oracle_profile}"],pipeline/f"{name}.log")
            if code==2:state["status"]="paused_insufficient_space";state["paused_job"]=job;atomic_json(state_path,state);return 2
            if code:state["status"]="failed";state["failed_job"]=job;state["exit"]=code;atomic_json(state_path,state);return code
        state["completed"].append(name);atomic_json(state_path,state)
    state["status"]="success";state["completed_utc"]=datetime.now(timezone.utc).isoformat();atomic_json(state_path,state);return 0


if __name__=="__main__":raise SystemExit(main())
