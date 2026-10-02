#!/usr/bin/env python3
"""Freeze and audit the Iteration model train/validation/test split."""
import argparse, csv, hashlib, json, shutil
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION=PROJECT/"experiments/20260929_iteration_calibration"
DEFAULT_TEST=PROJECT/"experiments/20260929-175832_adaptive_push_g32/inputs"
GRAPHS=("cit-Patents","soc-LiveJournal1","indochina","soc-orkut","soc-twitter")
WORKLOADS=("bfs","sssp","sswp")
TRAIN_BATCHES=(0,1,2,4,5,6,8,9,10,12,13,14)
VALIDATION_BATCHES=(3,7,11,15)

def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

def read(path):
 comments=[];rows=[]
 with path.open() as f:
  for line in f:
   if line.startswith("#"):comments.append(line.rstrip("\n"))
   elif line.strip():rows.append(next(csv.reader([line])))
 if len(rows)!=1024:raise ValueError(f"{path}: expected 1024 rows, found {len(rows)}")
 return comments,rows

def write_subset(path,comments,rows,batches,label):
 path.parent.mkdir(parents=True,exist_ok=True)
 selected=[row for i,row in enumerate(rows) if i//64 in batches]
 with path.open("w",newline="") as f:
  for line in comments:f.write(line+"\n")
  f.write(f"# iteration_split={label}; batches={','.join(map(str,batches))}\n")
  csv.writer(f).writerows(selected)
 return selected

def main():
 p=argparse.ArgumentParser();p.add_argument("--calibration-root",type=Path,default=DEFAULT_CALIBRATION)
 p.add_argument("--test-root",type=Path,default=DEFAULT_TEST);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 a.output=a.output.resolve();records=[]
 for graph in GRAPHS:
  for workload in WORKLOADS:
   calibration=a.calibration_root/"inputs"/graph/f"{workload}.csv";test=a.test_root/graph/f"{workload}.csv"
   comments,calibration_rows=read(calibration);_,test_rows=read(test)
   train_path=a.output/"train"/graph/f"{workload}.csv";validation_path=a.output/"validation"/graph/f"{workload}.csv"
   train=write_subset(train_path,comments,calibration_rows,TRAIN_BATCHES,"train")
   validation=write_subset(validation_path,comments,calibration_rows,VALIDATION_BATCHES,"validation")
   test_path=a.output/"test"/graph/f"{workload}.csv";test_path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(test,test_path)
   train_sources={int(r[1]) for r in train};validation_sources={int(r[1]) for r in validation};test_sources={int(r[1]) for r in test_rows}
   overlap={"train_validation":len(train_sources&validation_sources),"train_test":len(train_sources&test_sources),
            "validation_test":len(validation_sources&test_sources)}
   if any(overlap.values()):raise ValueError(f"source leakage for {graph}/{workload}: {overlap}")
   records.append({"graph":graph,"workload":workload,"train_batches":TRAIN_BATCHES,"validation_batches":VALIDATION_BATCHES,
    "train_rows":len(train),"validation_rows":len(validation),"test_rows":len(test_rows),"source_overlap":overlap,
    "train_path":str(train_path),"validation_path":str(validation_path),"test_path":str(test_path),
    "train_sha256":sha256(train_path),"validation_sha256":sha256(validation_path),"test_sha256":sha256(test_path),
    "calibration_source_path":str(calibration),"calibration_source_sha256":sha256(calibration),
    "test_source_path":str(test),"test_source_sha256":sha256(test)})
 manifest={"version":1,"unit":"query batch; 64 consecutive queries per batch","seed45":{"role":"model fitting and lambda selection",
  "train_rule":"batch % 4 != 3","validation_rule":"batch % 4 == 3"},"seed42":{"role":"frozen final test only; never fit or tune"},
  "counts":{"train_per_graph_workload":768,"validation_per_graph_workload":256,"test_per_graph_workload":1024},"records":records}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/"dataset_split_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 print(f"frozen {len(records)} graph/workload splits: train=11520 validation=3840 test=15360; source_overlap=0")

if __name__=="__main__":main()
