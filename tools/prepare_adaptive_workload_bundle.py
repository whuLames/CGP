#!/usr/bin/env python3
"""Prepare ordinary and controlled-tail SSSP workloads for one CSR graph."""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1];GIB=1<<30


def run(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w") as output:
        result=subprocess.run(command,cwd=PROJECT,stdout=output,stderr=subprocess.STDOUT,text=True)
    if result.returncode:raise RuntimeError(f"failed command; see {log}: {command}")


def inspect(cli,graph,capacity,device,log):
    command=[str(cli),f"--graph={graph}","--legacy_int_weights","--n=0",f"--q={capacity}",
             "--layout=grouped","--group_width=32","--frontier_build=fused",f"--device={device}","--log_level=warn"]
    result=subprocess.run(command,cwd=PROJECT,text=True,capture_output=True)
    log.write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(f"graph inspection failed: {log}")
    identity=re.search(r"graph=([0-9a-f]+)",result.stdout);total=re.search(r"memory.total=(\d+)",result.stdout)
    budget=re.search(r"budget=(\d+)",result.stdout)
    if not identity or not total or not budget:raise RuntimeError("incomplete graph inspection output")
    return identity.group(1),int(total.group(1))<=int(budget.group(1))


def capacity_copy(source,target,capacity):
    target.parent.mkdir(parents=True,exist_ok=True)
    lines=source.read_text().splitlines();found=False
    for index,line in enumerate(lines):
        if line.startswith("# capacity="):lines[index]=f"# capacity={capacity}";found=True;break
    if not found:raise ValueError(f"missing capacity header: {source}")
    target.write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--name",required=True)
    parser.add_argument("--graph",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cli",type=Path,default=PROJECT/"build/graphweft_cli")
    parser.add_argument("--sampler",type=Path,default=PROJECT/"build/graphweft_sample_csr_workloads")
    parser.add_argument("--device",type=int,default=0);parser.add_argument("--seed",type=int,default=20261007)
    args=parser.parse_args();args.graph=args.graph.resolve();args.output=args.output.resolve();args.cli=args.cli.resolve();args.sampler=args.sampler.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output).free<30*GIB:
        state={"status":"paused_insufficient_space","free_bytes":shutil.disk_usage(args.output).free,
               "required_reserve_bytes":30*GIB,"dataset":args.name}
        (args.output/"PAUSED_INSUFFICIENT_SPACE.json").write_text(json.dumps(state,indent=2)+"\n")
        print("*** WORKLOAD PREPARATION PAUSED: less than 30 GiB free ***");raise SystemExit(2)
    logs=args.output/"logs";logs.mkdir(parents=True,exist_ok=True)
    identity,fit128=inspect(args.cli,args.graph,128,args.device,logs/"inspect_m128.log")
    if not fit128:raise RuntimeError("base graph does not pass the M=128 memory gate")
    _,fit256=inspect(args.cli,args.graph,256,args.device,logs/"inspect_m256.log")
    sampled=args.output/"sampled_m128"
    if not sampled.exists():run([str(args.sampler),str(args.graph),str(sampled),identity,"128",str(args.seed)],logs/"sample.log")
    completions=args.output/"candidate_completions.csv"
    if not completions.exists():
        run([str(args.cli),f"--graph={args.graph}","--legacy_int_weights",f"--queries={sampled/'sssp_candidates.csv'}",
             "--n=4096","--q=128","--layout=grouped","--group_width=32","--algorithm=sssp","--selector=threshold",
             "--push_mapping=iteration","--pull_kernel=auto","--frontier=unordered","--frontier_build=fused",
             "--frontier_mask64=true",f"--completion_output={completions}",f"--device={args.device}","--log_level=warn"],
            logs/"candidate_completion.log")
    derived=args.output/"derived_graph"
    if not derived.exists():
        run(["python3",str(PROJECT/"tools/prepare_strong_tail_workloads.py"),"augment",f"--graph={args.graph}",
             f"--output={derived}","--paths=256","--bands=256,512,1024,2048",f"--seed={args.seed}"],logs/"augment.log")
    derived_identity,derived_fit128=inspect(args.cli,derived,128,args.device,logs/"inspect_derived_m128.log")
    if not derived_fit128:raise RuntimeError("derived graph does not pass the M=128 memory gate")
    _,derived_fit256=inspect(args.cli,derived,256,args.device,logs/"inspect_derived_m256.log")
    long128=args.output/"long_tail_m128"
    if not long128.exists():
        run(["python3",str(PROJECT/"tools/prepare_adaptive_sssp_workload.py"),
             f"--candidates={sampled/'sssp_candidates.csv'}",f"--completions={completions}",
             f"--augmentation={derived/'augmentation_manifest.json'}",f"--identity={derived_identity}",
             f"--output={long128}","--capacity=128","--group-width=32",f"--seed={args.seed}"],logs/"long_tail.log")
    ordinary={"ordinary_m128":str(sampled/"sssp.csv")};long_tail={"long_tail_m128":str(long128/"sssp_long_tail.csv")}
    if fit256:
        path=args.output/"sampled_m256/sssp.csv";capacity_copy(sampled/"sssp.csv",path,256);ordinary["ordinary_m256"]=str(path)
    if derived_fit256:
        path=args.output/"long_tail_m256/sssp_long_tail.csv";capacity_copy(long128/"sssp_long_tail.csv",path,256)
        long_tail["long_tail_m256"]=str(path)
    fragment={"datasets":{
        args.name:{"graph":str(args.graph),"directed":False,"legacy_int_weights":True,"workloads":ordinary},
        args.name+"-longtail":{"graph":str(derived),"directed":False,"legacy_int_weights":True,"workloads":long_tail}},
        "identity":identity,"derived_identity":derived_identity,"fits":{"base_m128":fit128,"base_m256":fit256,
        "derived_m128":derived_fit128,"derived_m256":derived_fit256}}
    (args.output/"campaign_manifest_fragment.json").write_text(json.dumps(fragment,indent=2)+"\n")
    print(json.dumps(fragment,indent=2))


if __name__=="__main__":main()
