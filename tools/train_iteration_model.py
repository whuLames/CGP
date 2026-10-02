#!/usr/bin/env python3
"""Fit/audit the frozen 30-output weighted quadratic ridge model."""
import argparse, csv, hashlib, json, math, re, statistics
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
LAMBDAS=(1e-6,1e-4,1e-2,1.0,100.0)
FEATURES=("log_frontier_vertices","log_live_queries","log_vertex_pairs_per_frontier",
 "log_edge_pairs_per_vertex_pair","log_edge_pairs_per_frontier","log_graph_edges_per_vertex","density")

def hash_files(paths):
 h=hashlib.sha256()
 for path in sorted(paths):
  h.update(str(path).encode());h.update(path.read_bytes())
 return h.hexdigest()

def parse_name(path):
 m=re.fullmatch(r"(bfs|sssp|sswp)_q(1|2|4|8|16|32)_(w1|w2|w4|b2|b4)_r(\d+)\.csv",path.name)
 if not m:raise ValueError(path)
 lanes=(1,2,4,8,16,32).index(int(m.group(2)));grain=("w1","w2","w4","b2","b4").index(m.group(3))
 return m.group(1),lanes*5+grain,int(m.group(4))

def base_features(r,graph_vertices,graph_edges):
 f=float(r["frontier_vertices"]);v=float(r["vertex_pairs"]);e=float(r["edge_pairs"]);live=float(r["live_queries"])
 return [math.log1p(f),math.log1p(float(r["live_queries"])),math.log1p(v/f if f else 0),
  math.log1p(e/v if v else 0),math.log1p(e/f if f else 0),
  math.log1p(graph_edges/graph_vertices if graph_vertices else 0),e/(graph_edges*live) if graph_edges and live else 0]

def expand(z):return np.asarray([1,*z,*[z[i]*z[j] for i in range(7) for j in range(i,7)]],dtype=float)

def load(root):
 config=json.loads((root/"config.json").read_text());paths=sorted(root.glob("*/raw/*_q*_r*.csv"));groups={}
 # Graph V/E are invariant and recovered from feature identities: density=Epair/(E*Qlive),
 # while V is read from the CSR offset count.
 for path in paths:
  graph=path.parents[1].name;workload,candidate,repetition=parse_name(path)
  graph_path=Path(config["graphs"][graph]);vertices=(graph_path/"csr_vlist.bin").stat().st_size//4-1
  edges=(graph_path/"csr_elist.bin").stat().st_size//4
  with path.open(newline="") as f:
   for r in csv.DictReader(f):
    required=("batch","round","frontier_vertices","live_queries","vertex_pairs","edge_pairs","kernel_gpu_ms")
    if any(r.get(field) in (None,"") for field in required):raise ValueError(f"malformed/incomplete calibration row in {path}")
    key=(graph,workload,int(r["batch"]),int(r["round"]),candidate)
    groups.setdefault(key,{"times":[],"features":base_features(r,vertices,edges)})["times"].append(float(r["kernel_gpu_ms"]))
 rows={}
 for (g,w,b,r,c),item in groups.items():
  if len(item["times"])!=config["formal_repetitions"]:raise ValueError(f"incomplete {(g,w,b,r,c)}")
  rows.setdefault((g,w,b,r),{"features":item["features"],"times":np.zeros(30)})["times"][c]=statistics.median(item["times"])
 for key,item in rows.items():
  if np.any(item["times"]<=0):raise ValueError(f"missing/nonpositive candidate {key}")
 return config,paths,rows

def fit(x,y,w,lam):
 penalty=np.eye(x.shape[1]);penalty[0,0]=0
 return np.linalg.solve(x.T@(w[:,None]*x)+lam*penalty,x.T@(w*y))

def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--audit",type=Path,required=True)
 p.add_argument("--split-manifest",type=Path)
 p.add_argument("--header",type=Path,default=PROJECT/"include/graphweft/iteration_model_coefficients.hpp");args=p.parse_args()
 split_path=args.split_manifest or args.input/"dataset"/"dataset_split_manifest.json"
 if not split_path.exists():raise ValueError(f"missing frozen split manifest: {split_path}")
 split=json.loads(split_path.read_text())
 if split.get("seed45",{}).get("train_rule")!="batch % 4 != 3" or split.get("seed45",{}).get("validation_rule")!="batch % 4 == 3":
  raise ValueError("unexpected train/validation split")
 config,paths,rows=load(args.input);keys=sorted(rows);raw=np.asarray([rows[k]["features"] for k in keys])
 minimum=raw.min(0);maximum=raw.max(0);clipped=np.clip(raw,minimum,maximum);mean=clipped.mean(0);scale=clipped.std(0);scale[scale==0]=1
 x=np.asarray([expand(z) for z in (clipped-mean)/scale]);times=np.asarray([rows[k]["times"] for k in keys])
 y=np.log(times)-np.log(times).mean(1,keepdims=True)
 # Equal graph/workload mass; inside each pair weight rounds by q1_w1 time.
 weights=np.zeros(len(keys))
 for pair in sorted(set(k[:2] for k in keys)):
  idx=np.asarray([i for i,k in enumerate(keys) if k[:2]==pair]);weights[idx]=times[idx,0]/times[idx,0].sum()/15
 train=np.asarray([k[2]%4!=3 for k in keys]);valid=~train;selection=[]
 for lam in LAMBDAS:
  coef=np.stack([fit(x[train],y[train,c],weights[train],lam) for c in range(30)])
  valid_idx=np.flatnonzero(valid);chosen=np.argmin(x[valid_idx]@coef.T,axis=1)
  actual=times[valid_idx,chosen];valid_keys=[keys[i] for i in valid_idx]
  score=sum(actual[[i for i,k in enumerate(valid_keys) if k[:2]==pair]].sum()
            for pair in sorted(set(k[:2] for k in valid_keys)))
  selection.append({"lambda":lam,"validation_selected_kernel_ms":float(score)})
 selected=min(selection,key=lambda z:(z["validation_selected_kernel_ms"],z["lambda"]))["lambda"]
 coef=np.stack([fit(x,y[:,c],weights,selected) for c in range(30)])
 data_hash=hash_files(paths);payload={"model":"weighted quadratic ridge","features":FEATURES,"expanded_features":36,
  "candidate_order":"q1..q32 x W1,W2,W4,B2,B4","split":"batch % 4 == 3 validation","lambdas":LAMBDAS,
  "selection":selection,"selected_lambda":selected,"rows":len(keys),"raw_csv_sha256":data_hash,
  "split_manifest":str(split_path),"split_manifest_sha256":hashlib.sha256(split_path.read_bytes()).hexdigest(),
  "minimum":minimum.tolist(),"maximum":maximum.tolist(),"mean":mean.tolist(),"scale":scale.tolist(),"coefficients":coef.tolist()}
 version="iteration-ridge-v1-"+hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16];payload["version"]=version
 args.audit.parent.mkdir(parents=True,exist_ok=True);args.audit.write_text(json.dumps(payload,indent=2)+"\n")
 def arr(values):return "{"+",".join(f"{v:.17g}" for v in values)+"}"
 lines=["#pragma once","// Generated by tools/train_iteration_model.py; do not hand edit.","#include <array>",
  "namespace graphweft::iteration_model_data {",f'inline constexpr char version[] = "{version}";',
  f"inline constexpr std::array<double,7> minimum = {arr(minimum)};",f"inline constexpr std::array<double,7> maximum = {arr(maximum)};",
  f"inline constexpr std::array<double,7> mean = {arr(mean)};",f"inline constexpr std::array<double,7> scale = {arr(scale)};",
  "inline constexpr std::array<std::array<double,36>,30> coefficients = {{"]
 lines += ["  std::array<double,36>"+arr(row)+"," for row in coef]
 lines += ["}};","}"];args.header.write_text("\n".join(lines)+"\n")
 print(version,selected,data_hash)

if __name__=="__main__":main()
