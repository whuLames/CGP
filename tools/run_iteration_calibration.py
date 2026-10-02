#!/usr/bin/env python3
"""Collect the disjoint seed-45 G=32 iteration-model calibration corpus."""
import argparse, concurrent.futures, csv, hashlib, json, random, struct, subprocess
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
BASE=PROJECT/"experiments/20260927_fixed_n1024_m64"
GRAPHS={
 "cit-Patents":Path("/home/zyl/data/csr_data/cit-Patents"),
 "soc-LiveJournal1":Path("/home/zyl/data/csr_data/soc-LiveJournal1"),
 "indochina":Path("/home/zyl/data/csr_data/indochina"),
 "soc-orkut":Path("/home/zyl/data/csr_data/soc-orkut"),
 "soc-twitter":Path("/home/zyl/data/csr_data/soc-twitter")}
IDENTITIES={"cit-Patents":"ca71a63d2bc4aa14","soc-LiveJournal1":"61876d811dd317b5",
 "indochina":"ab252d0d7a2c312e","soc-orkut":"c476f4df35515f1e","soc-twitter":"54434ea949424dbb"}
WORKLOADS={"bfs":0,"sssp":1,"sswp":2}
LANES=(1,2,4,8,16,32); GRAINS=range(5)

def complete_round_csv(path):
 if not path.exists() or not path.stat().st_size:return False
 try:
  with path.open(newline="") as f:
   rows=list(csv.DictReader(f))
  required=("batch","round","frontier_vertices","live_queries","vertex_pairs","edge_pairs","kernel_gpu_ms")
  return (bool(rows) and {int(r["batch"]) for r in rows}==set(range(16))
          and all(all(r.get(field) not in (None,"") for field in required) for r in rows))
 except (OSError,csv.Error):return False

def complete_fingerprint(path):
 if not path.exists() or not path.stat().st_size:return False
 try:
  with path.open(newline="") as f:return sum(1 for _ in csv.DictReader(f))==1024
 except (OSError,csv.Error):return False

def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

def read_sources(path):
 out=set()
 with path.open() as f:
  for line in f:
   if line.strip() and not line.startswith("#"):out.add(int(next(csv.reader([line]))[1]))
 return out

def make_inputs(root,name):
 graph=GRAPHS[name]; raw=(graph/"csr_vlist.bin").read_bytes()
 offsets=struct.unpack(f"<{len(raw)//4}i",raw)
 eligible=[v for v in range(len(offsets)-1) if offsets[v+1]>offsets[v]]
 excluded=set()
 for leaf in ("bfs_frozen.csv","sssp_frozen.csv","bfs_candidates.csv","sssp_candidates.csv"):
  excluded|=read_sources(BASE/name/"workloads"/leaf)
 population=[v for v in eligible if v not in excluded]
 sources=random.Random(45).sample(population,1024)
 target=root/"inputs"/name;target.mkdir(parents=True,exist_ok=True)
 result={}
 for workload,algorithm in WORKLOADS.items():
  path=target/f"{workload}.csv"
  with path.open("w",newline="") as f:
   f.write(f"# graph_identity={IDENTITIES[name]}\n# capacity=64\n# seed=45; disjoint from seed-42 formal and seed-43 candidate sources\n")
   f.write("# id,source,score,offset,feature_key,algorithm,reference_rounds\n")
   w=csv.writer(f)
   for i,source in enumerate(sources):w.writerow((600000+algorithm*100000+i,source,0,0,0,algorithm,0))
  result[workload]={"path":str(path),"sha256":sha256(path),"rows":1024}
 return result

def candidate(index):
 lanes=LANES[index//5];grain=index%5
 suffix=("w"+str((1,2,4)[grain])) if grain<3 else ("b"+str((2,4)[grain-3]))
 return f"q{lanes}_{suffix}",lanes,grain

def worker(args,name,device,workloads=None,candidate_start=0,candidate_end=30):
 raw=args.output/name/"raw";raw.mkdir(parents=True,exist_ok=True)
 selected=tuple(workloads or WORKLOADS)
 suffix="_"+"_".join(selected)+f"_{candidate_start}_{candidate_end}" if workloads else ""
 commands=args.output/name/f"commands{suffix}.jsonl"
 for workload in selected:
  common=[str(args.cli),f"--graph={GRAPHS[name]}","--directed","--legacy_int_weights",
   f"--queries={args.output/'inputs'/name/(workload+'.csv')}","--n=1024","--q=64",
   "--layout=grouped","--group_width=32",f"--algorithm={workload}","--selector=push",
   "--frontier=unordered","--frontier_build=direct","--frontier_mask64=true",
   "--same_algorithm_groups","--profile_kernel",f"--device={device}","--log_level=warn"]
  fingerprint=raw/f"{workload}_final_fingerprint.csv"
  if candidate_start==0 and not complete_fingerprint(fingerprint):
   fingerprint_cmd=common+["--push_mapping=static","--push_query_lanes=1","--push_grain=0",f"--result_fingerprints={fingerprint}"]
   subprocess.run(fingerprint_cmd,cwd=PROJECT,check=True,stdout=subprocess.DEVNULL)
  for index in range(candidate_start,candidate_end):
   label,lanes,grain=candidate(index);flags=["--push_mapping=static",f"--push_query_lanes={lanes}",f"--push_grain={grain}"]
   pending=[repetition for repetition in range(args.repetitions)
            if not complete_round_csv(raw/f"{workload}_{label}_r{repetition}.csv")]
   if not pending:continue
   subprocess.run(common+flags,cwd=PROJECT,check=True,stdout=subprocess.DEVNULL)
   for repetition in pending:
    rounds=raw/f"{workload}_{label}_r{repetition}.csv";cmd=common+flags+[f"--round_metrics={rounds}"]
    run=subprocess.run(cmd,cwd=PROJECT,text=True,capture_output=True)
    with commands.open("a") as f:f.write(json.dumps({"workload":workload,"candidate":label,"candidate_index":index,
      "repetition":repetition,"returncode":run.returncode,"round_metrics":str(rounds),"command":cmd})+"\n")
    if run.returncode:raise RuntimeError(f"{name}/{workload}/{label}/r{repetition}: {run.stderr}")
 return name

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
 p.add_argument("--cli",type=Path,default=PROJECT/"build/graphweft_cli")
 p.add_argument("--devices",nargs=5,type=int,default=[0,1,2,3,4]);p.add_argument("--repetitions",type=int,default=5)
 p.add_argument("--job",action="append",default=[],metavar="GRAPH:WORKLOAD:DEVICE[:START:END]",
                help="Run an independent resumable graph/workload candidate range; may be repeated")
 args=p.parse_args();args.output=args.output.resolve();args.cli=args.cli.resolve();args.output.mkdir(parents=True,exist_ok=True)
 manifest={name:make_inputs(args.output,name) for name in GRAPHS}
 config={"seed":45,"N":1024,"Q":64,"group_width":32,"selector":"all-push","frontier":"direct unordered mask64",
  "warmups":1,"formal_repetitions":args.repetitions,"graphs":{k:str(v) for k,v in GRAPHS.items()},"inputs":manifest,
  "binary_sha256":sha256(args.cli),"devices":dict(zip(GRAPHS,args.devices))}
 (args.output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
 requested=[]
 for spec in args.job:
  fields=spec.split(":")
  if len(fields) not in (3,5):raise ValueError(f"invalid --job: {spec}")
  name,workload,device=fields[:3]
  if name not in GRAPHS or workload not in WORKLOADS:raise ValueError(f"invalid --job: {spec}")
  start,end=(map(int,fields[3:]) if len(fields)==5 else (0,30))
  if start<0 or end>30 or start>=end:raise ValueError(f"invalid candidate range: {spec}")
  requested.append((name,workload,int(device),start,end))
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(requested) or 5) as pool:
  jobs=([pool.submit(worker,args,name,device,(workload,),start,end) for name,workload,device,start,end in requested]
        if requested else [pool.submit(worker,args,name,device) for name,device in zip(GRAPHS,args.devices)])
  for job in concurrent.futures.as_completed(jobs):print("complete",job.result(),flush=True)

if __name__=="__main__":main()
