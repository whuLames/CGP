#!/usr/bin/env python3
"""Create the fixed 768-natural + 256-controlled-path SSSP workload."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):h.update(block)
    return h.hexdigest()


def queries(path):
    result={}
    with path.open() as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                row=next(csv.reader([line]));result[int(row[0])]=int(row[1])
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--candidates",type=Path,required=True)
    parser.add_argument("--completions",type=Path,required=True);parser.add_argument("--augmentation",type=Path,required=True)
    parser.add_argument("--identity",required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--capacity",type=int,default=128);parser.add_argument("--group-width",type=int,default=32)
    parser.add_argument("--seed",type=int,default=20261007);args=parser.parse_args()
    source_by_id=queries(args.candidates)
    with args.completions.open(newline="") as handle:
        rounds={int(row["query_id"]):int(row["service_rounds"]) for row in csv.DictReader(handle)}
    if source_by_id.keys()!=rounds.keys():raise ValueError("candidate and completion IDs differ")
    records=sorted(({"candidate_id":qid,"source":source,"rounds":rounds[qid]} for qid,source in source_by_id.items()),
                   key=lambda row:(row["rounds"],row["source"]))
    short=random.Random(args.seed).sample(records[:len(records)//2],768)
    augmentation=json.loads(args.augmentation.read_text());heads=augmentation["path_heads"][:256]
    lengths=augmentation["path_edges"][:256]
    if len(heads)!=256:raise ValueError("augmentation does not contain 256 controlled paths")
    random.Random(args.seed+1).shuffle(short)
    long=list(zip(heads,lengths));random.Random(args.seed+2).shuffle(long)
    if args.group_width!=32:raise ValueError("the fixed 24+8 interleave requires G=32")
    rows=[];audit=[]
    for group in range(32):
        for item in short[group*24:(group+1)*24]:
            audit.append({"order":len(rows),"source":item["source"],"label":"natural_short",
                          "reference_rounds":item["rounds"],"provenance":f"candidate:{item['candidate_id']}"})
            rows.append(item["source"])
        for source,length in long[group*8:(group+1)*8]:
            audit.append({"order":len(rows),"source":source,"label":"controlled_long",
                          "reference_rounds":length+1,"provenance":"appended_path"})
            rows.append(source)
    args.output.mkdir(parents=True,exist_ok=False);workload=args.output/"sssp_long_tail.csv"
    with workload.open("w",newline="") as handle:
        handle.write(f"# graph_identity={args.identity}\n# capacity={args.capacity}\n")
        handle.write("# 768 measured-natural-short + 256 controlled path queries; each G=32 is 24+8\n")
        handle.write("# id,source,score,offset,feature_key,algorithm,reference_rounds\n")
        writer=csv.writer(handle)
        for index,item in enumerate(audit):writer.writerow((3000000+index,item["source"],0,0,0,1,item["reference_rounds"]))
    with (args.output/"workload_audit.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    (args.output/"manifest.json").write_text(json.dumps({"schema":1,"N":1024,"M":args.capacity,"G":32,
        "natural_short":768,"controlled_long":256,"seed":args.seed,"graph_identity":args.identity,
        "candidate_sha256":digest(args.candidates),"completion_sha256":digest(args.completions),
        "augmentation_sha256":digest(args.augmentation),"workload_sha256":digest(workload)},indent=2)+"\n")


if __name__=="__main__":main()
