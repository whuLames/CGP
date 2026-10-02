#!/usr/bin/env python3
"""Formal Iteration Push campaign: frozen Hybrid baselines and fresh All-Push."""
import argparse, concurrent.futures, csv, hashlib, json, math, re, statistics, subprocess, time
from datetime import datetime
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
MODEL_HEADER=PROJECT/"include/graphweft/iteration_model_coefficients.hpp"
INPUTS=PROJECT/"experiments/20260929-175832_adaptive_push_g32/inputs"
FROZEN_HYBRID=PROJECT/"experiments/20260929-175832_adaptive_push_g32/result.csv"
GRAPHS={"cit-Patents":Path("/home/zyl/data/csr_data/cit-Patents"),"soc-LiveJournal1":Path("/home/zyl/data/csr_data/soc-LiveJournal1"),
 "indochina":Path("/home/zyl/data/csr_data/indochina"),"soc-orkut":Path("/home/zyl/data/csr_data/soc-orkut"),"soc-twitter":Path("/home/zyl/data/csr_data/soc-twitter")}
STATIC={"cit-Patents":(16,1,"q16_w2"),"soc-LiveJournal1":(16,1,"q16_w2"),"indochina":(2,2,"q2_w4"),
 "soc-orkut":(16,2,"q16_w4"),"soc-twitter":(4,3,"q4_b2")}
WORKLOADS=("bfs","sssp","sswp");ALL_PUSH=("shared","static","iteration")
METRICS=("workload_ms","throughput_qps","kernel_ms","kernel_gpu_ms","feature_ms","frontier_ms","copy_ms","selector_ms","round_ms","rounds","push_rounds","pull_rounds")

def parse(text):
 out={}
 for k,v in re.findall(r"([A-Za-z_]+)=([^\s]+)",text):
  try:out[k]=float(v)
  except ValueError:out[k]=v
 return out
def fp(path):
 with path.open(newline="") as f:return list(csv.DictReader(f))
def write(path,rows):
 fields=sorted(set().union(*(r for r in rows))) if rows else []
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def geomean(x):return math.exp(statistics.fmean(math.log(v) for v in x))

class Runner:
 def __init__(self,args,name,device):self.a=args;self.name=name;self.device=device;self.raw=args.output/name/"raw";self.raw.mkdir(parents=True,exist_ok=True);self.rows=[]
 def command(self,w,m,mode,rounds=None,hashes=None):
  cmd=[str(self.a.cli),f"--graph={GRAPHS[self.name]}","--directed","--legacy_int_weights",f"--queries={INPUTS/self.name/(w+'.csv')}",
   "--n=1024","--q=64","--layout=grouped","--group_width=32",f"--algorithm={w}",f"--selector={'push' if mode=='all_push' else 'threshold'}",
   "--pull_threshold=0.20","--frontier=unordered","--frontier_build=direct","--frontier_mask64=true","--same_algorithm_groups",
   "--profile_kernel","--log_level=warn",f"--device={self.device}",f"--push_mapping={m}"]
  if m=="static":lanes,grain,_=STATIC[self.name];cmd += [f"--push_query_lanes={lanes}",f"--push_grain={grain}"]
  if rounds:cmd.append(f"--round_metrics={rounds}")
  if hashes:cmd.append(f"--result_fingerprints={hashes}")
  return cmd
 def execute(self,cmd,stem,meta):
  started=time.time();run=subprocess.run(cmd,cwd=PROJECT,text=True,capture_output=True)
  (self.raw/(stem+".stdout.log")).write_text(run.stdout);(self.raw/(stem+".stderr.log")).write_text(run.stderr)
  with (self.a.output/self.name/"commands.jsonl").open("a") as f:f.write(json.dumps({**meta,"returncode":run.returncode,"wall_s":time.time()-started,"command":cmd})+"\n")
  if run.returncode:raise RuntimeError(f"{self.name}/{stem}: {run.stderr}")
  return parse(run.stdout)
 def aggregate(self,path):
  with path.open(newline="") as f:rows=list(csv.DictReader(f))
  out={"round_metric_rows":len(rows),"predicted_cost_sum":sum(float(r["predicted_relative_cost"]) for r in rows if r["predicted_relative_cost"]!="nan")}
  for r in rows:
   key="selected_"+r["kernel_family"]
   if r["kernel_family"]=="partition_push":key+=f"_q{r['group_size']}_"+(f"w{r['warps_per_block']}" if r["blocks_per_vertex"]=="1" else f"b{r['blocks_per_vertex']}")
   out[key]=out.get(key,0)+1
  versions={r["iteration_model_version"] for r in rows if r["iteration_model_version"]};out["model_version"]=";".join(sorted(versions))
  return out
 def validate(self,w,mode,mappings):
  values={}
  for m in mappings:
   path=self.raw/f"validate_{mode}_{w}_{m}.csv";self.execute(self.command(w,m,mode,hashes=path),f"validate_{mode}_{w}_{m}",{"kind":"validation","mode":mode,"workload":w,"mapping":m});values[m]=fp(path)
  if any(v!=values[mappings[0]] for v in values.values()):raise RuntimeError(f"{self.name}/{mode}/{w}: fingerprint mismatch")
 def measured(self,w,m,mode):
  self.execute(self.command(w,m,mode),f"warmup_{mode}_{w}_{m}",{"kind":"warmup","mode":mode,"workload":w,"mapping":m})
  for rep in range(self.a.repetitions):
   stem=f"formal_{mode}_{w}_{m}_r{rep}";rounds=self.raw/(stem+"_rounds.csv")
   values=self.execute(self.command(w,m,mode,rounds=rounds),stem,{"kind":"formal","mode":mode,"workload":w,"mapping":m,"repetition":rep})
   self.rows.append({"dataset":self.name,"mode":mode,"workload":w,"mapping":m,"repetition":rep,"static_source":STATIC[self.name][2],
    **{k:values[k] for k in METRICS if k in values},**self.aggregate(rounds)});write(self.a.output/self.name/"raw_measurements.csv",self.rows)
 def run(self):
  for w in self.a.workloads:
   if "all_push" in self.a.modes:
    self.validate(w,"all_push",ALL_PUSH)
    for m in ALL_PUSH:self.measured(w,m,"all_push")
   if "hybrid" in self.a.modes:
    # Hybrid Shared/Static are frozen; only validate and measure the new mapping.
    self.validate(w,"hybrid",("shared","static","iteration"));self.measured(w,"iteration","hybrid")
  return self.rows

def summarize(rows,workloads,modes):
 frozen=list(csv.DictReader(FROZEN_HYBRID.open(newline="")));out=[]
 for name in GRAPHS:
  for w in workloads:
   for mode,mappings in (("all_push",ALL_PUSH),("hybrid",("shared","static","iteration"))):
    if mode not in modes:continue
    med={}
    for m in mappings:
     source=[r for r in (rows if m=="iteration" or mode=="all_push" else frozen) if r["dataset"]==name and r["workload"]==w and r["mapping"]==m and (mode=="hybrid" or r.get("mode")==mode)]
     value=statistics.median(float(r["workload_ms"]) for r in source);med[m]=value;out.append({"dataset":name,"workload":w,"mode":mode,"mapping":m,"workload_ms_median":value,"samples":len(source)})
    for row in out[-3:]:row["speedup_vs_shared"]=med["shared"]/row["workload_ms_median"];row["speedup_vs_static"]=med["static"]/row["workload_ms_median"]
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--cli",type=Path,default=PROJECT/"build/graphweft_cli");p.add_argument("--repetitions",type=int,default=5);p.add_argument("--devices",nargs=5,type=int,default=[0,1,2,3,4])
 p.add_argument("--workloads",nargs="+",choices=WORKLOADS,default=list(WORKLOADS))
 p.add_argument("--modes",nargs="+",choices=("all_push","hybrid"),default=["all_push","hybrid"])
 a=p.parse_args()
 model_match=re.search(r'char version\[\] = "([^"]+)"',MODEL_HEADER.read_text())
 if not model_match or "bootstrap" in model_match.group(1):raise SystemExit("refusing formal campaign: fit and rebuild the iteration model first")
 a.output=a.output.resolve();a.cli=a.cli.resolve();a.output.mkdir(parents=True,exist_ok=True)
 (a.output/"config.json").write_text(json.dumps({"started_at":datetime.now().astimezone().isoformat(),"formal_repetitions":a.repetitions,"frozen_hybrid":str(FROZEN_HYBRID),"all_push_mappings":ALL_PUSH,"hybrid_new_mapping":"iteration","model_version":model_match.group(1),"workloads":a.workloads,"modes":a.modes,"devices":dict(zip(GRAPHS,a.devices))},indent=2)+"\n")
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
  jobs=[pool.submit(Runner(a,n,d).run) for n,d in zip(GRAPHS,a.devices)]
  for job in concurrent.futures.as_completed(jobs):rows+=job.result()
 write(a.output/"result.csv",rows);summary=summarize(rows,a.workloads,a.modes);write(a.output/"summary.csv",summary)
 gates={mode:geomean([r["speedup_vs_static"] for r in summary if r["mode"]==mode and r["mapping"]=="iteration"]) for mode in a.modes}
 result={**{mode+"_geomean_speedup_vs_static":value for mode,value in gates.items()},
  "accepted":(gates["all_push"]>=1.05 and gates["hybrid"]>=1.0) if set(a.modes)=={"all_push","hybrid"} and set(a.workloads)==set(WORKLOADS) else None,
  "fingerprints_match":True}
 (a.output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
