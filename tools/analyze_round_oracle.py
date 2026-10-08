#!/usr/bin/env python3
"""Summarize matched-state oracle runs and emit selector-training rows."""
import argparse
import csv
import gzip
import json
import math
import re
import statistics
from pathlib import Path


def parse_metrics(path: Path):
    text = path.read_text()
    lines = [line for line in text.splitlines() if "workload_ms=" in line]
    if not lines:
        raise ValueError(f"missing final metrics line: {path}")
    return {key: float(value) for key, value in re.findall(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)", lines[-1])}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def base_features(row, vertices, edges):
    frontier = float(row["frontier_vertices"]); vertex_pairs = float(row["vertex_pairs"])
    edge_pairs = float(row["edge_pairs"]); live = float(row["live_queries"])
    return {
        "log_frontier_vertices": math.log1p(frontier),
        "log_live_queries": math.log1p(live),
        "log_vertex_pairs_per_frontier": math.log1p(vertex_pairs/frontier if frontier else 0),
        "log_edge_pairs_per_vertex_pair": math.log1p(edge_pairs/vertex_pairs if vertex_pairs else 0),
        "log_edge_pairs_per_frontier": math.log1p(edge_pairs/frontier if frontier else 0),
        "log_graph_edges_per_vertex": math.log1p(edges/vertices if vertices else 0),
        "density": edge_pairs/(edges*live) if edges and live else 0,
    }


def graph_size(path: Path):
    vertices = (path / "csr_vlist.bin").stat().st_size // 4 - 1
    edges = (path / "csr_elist.bin").stat().st_size // 4
    return vertices, edges


def percentile(values,q):
    if not values:return 0.0
    values=sorted(values);position=(len(values)-1)*q;lower=math.floor(position);upper=math.ceil(position)
    return values[lower] if lower==upper else values[lower]*(upper-position)+values[upper]*(position-lower)


def case_parts(root, oracle):
    relative = oracle.relative_to(root)
    # dataset/workload/mM/refill/oracle/repR/oracle.csv
    if len(relative.parts) != 7:
        raise ValueError(f"unexpected result path: {oracle}")
    dataset, workload, m_label, refill, oracle_dir, repetition, filename = relative.parts
    if oracle_dir not in {"oracle", "oracle_paired"} or filename != "oracle.csv":
        raise ValueError(f"unexpected result path: {oracle}")
    return dataset, workload, int(m_label[1:]), refill, int(repetition[3:]), (
        "paired" if oracle_dir == "oracle_paired" else "core")


def selected_oracle_paths(root):
    paths=[]
    for config_path in sorted(root.glob("*/*/m*/*/config.json")):
        try: profile=json.loads(config_path.read_text()).get("oracle_profile","core")
        except (OSError,json.JSONDecodeError): continue
        oracle_dir="oracle_paired" if profile=="paired" else "oracle"
        paths.extend(sorted((config_path.parent/oracle_dir).glob("rep*/oracle.csv")))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root=args.input.resolve(); out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text()); training=[]; per_round={}; per_run={}
    oracle_paths=selected_oracle_paths(root)
    if not oracle_paths:
        raise ValueError("no completed oracle CSV files found")
    for path in oracle_paths:
        dataset,workload,capacity,refill,repetition,profile=case_parts(root,path)
        vertices,edges=graph_size(Path(manifest["datasets"][dataset]["graph"]))
        for row in read_csv(path):
            row.setdefault("oracle_profile",profile)
            row.update({"dataset":dataset,"workload":workload,"M":capacity,"Q":capacity,"G":32,
                        "refill_policy":refill,"repetition":repetition,"graph_vertices":vertices,
                        "graph_edges":edges,"active_slot_ratio":float(row["live_queries"])/capacity})
            row.update(base_features(row,vertices,edges))
            row["candidate_direction"]="pull" if row["kernel_family"]=="dense_pull" else "push"
            row["candidate_pipeline_adjusted_ms"]=float(row["pipeline_gpu_ms"])+float(row["adaptive_preparation_ms"])
            training.append(row)
            key=(dataset,workload,capacity,refill,profile,int(row["batch"]),int(row["round"]))
            per_round.setdefault(key,{})[repetition]=per_round.setdefault(key,{}).get(repetition,[])+[row]

    winner_rows=[]
    for key,repetitions in sorted(per_round.items()):
        winners={};margins={}
        for repetition,rows in repetitions.items():
            ordered=sorted(rows,key=lambda r:(float(r["candidate_pipeline_adjusted_ms"]),r["kernel_token"]))
            winners[repetition]=ordered[0]["kernel_token"]
            margins[repetition]=(float(ordered[1]["candidate_pipeline_adjusted_ms"])/
                                 float(ordered[0]["candidate_pipeline_adjusted_ms"])-1) if len(ordered)>1 else 0
            by_direction={direction:min((row for row in rows if row["candidate_direction"]==direction),
                key=lambda row:(float(row["candidate_pipeline_adjusted_ms"]),row["kernel_token"]),default=None)
                for direction in ("push","pull")}
            winner_direction=ordered[0]["candidate_direction"]
            production_direction=ordered[0]["production_direction"]
            run_key=key[:5]+(repetition,)
            item=per_run.setdefault(run_key,{"kernel_oracle_ms":0.0,"pipeline_oracle_gpu_ms":0.0,
                                             "pipeline_oracle_wall_ms":0.0,"rounds":0})
            item["kernel_oracle_ms"]+=min(float(r["kernel_gpu_ms"]) for r in rows)
            item["pipeline_oracle_gpu_ms"]+=float(ordered[0]["candidate_pipeline_adjusted_ms"])
            item["pipeline_oracle_wall_ms"]+=float(ordered[0]["pipeline_wall_ms"])+float(ordered[0]["adaptive_preparation_ms"])
            item["rounds"]+=1
            item.setdefault("direction_correct_rounds",0)
            item["direction_correct_rounds"]+=int(production_direction==winner_direction)
            item.setdefault("production_direction_regret_ms",0.0)
            production_best=by_direction.get(production_direction)
            item["production_direction_regret_ms"]+=(float(production_best["candidate_pipeline_adjusted_ms"])-
                float(ordered[0]["candidate_pipeline_adjusted_ms"])) if production_best else 0.0
            winners[(repetition,"push")]=by_direction["push"]["kernel_token"] if by_direction["push"] else ""
            winners[(repetition,"pull")]=by_direction["pull"]["kernel_token"] if by_direction["pull"] else ""
            winners[(repetition,"direction")]=winner_direction
            winners[(repetition,"production_direction")]=production_direction
        global_winners=[winners.get(0),winners.get(1)]
        stable=len(set(global_winners))==1 and None not in global_winners and all(value>.03 for value in margins.values())
        winner_rows.append(dict(zip(("dataset","workload","M","refill_policy","oracle_profile","batch","round"),key)) | {
            "winner_rep0":winners.get(0,""),"winner_rep1":winners.get(1,""),
            "best_push_rep0":winners.get((0,"push"),""),"best_push_rep1":winners.get((1,"push"),""),
            "best_pull_rep0":winners.get((0,"pull"),""),"best_pull_rep1":winners.get((1,"pull"),""),
            "winner_direction_rep0":winners.get((0,"direction"),""),
            "winner_direction_rep1":winners.get((1,"direction"),""),
            "production_direction_rep0":winners.get((0,"production_direction"),""),
            "production_direction_rep1":winners.get((1,"production_direction"),""),
            "production_direction_correct_rep0":int(winners.get((0,"direction"))==winners.get((0,"production_direction"))),
            "production_direction_correct_rep1":int(winners.get((1,"direction"))==winners.get((1,"production_direction"))),
            "margin_rep0":margins.get(0,""),"margin_rep1":margins.get(1,""),"stable_label":int(stable),
            "winner":global_winners[0] if stable else "tie_or_unstable"})

    fields=list(training[0])
    with gzip.open(out/"training_rows.csv.gz","wt",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(training)
    with (out/"oracle_winners.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(winner_rows[0]));writer.writeheader();writer.writerows(winner_rows)

    summary=[]
    for key,oracle in sorted(per_run.items()):
        dataset,workload,capacity,refill,profile,repetition=key
        for baseline in ("adaptive","iteration"):
            stdout=root/dataset/workload/f"m{capacity}"/refill/baseline/f"rep{repetition}"/"stdout.log"
            if not stdout.exists(): continue
            metrics=parse_metrics(stdout)
            completion_path=stdout.parent/"completions.csv";latencies=[]
            if completion_path.exists():latencies=[float(row["submit_to_completion_ms"]) for row in read_csv(completion_path)]
            fixed=max(0.0,metrics["workload_ms"]-metrics.get("copy_ms",0)-metrics.get("kernel_ms",0)-metrics.get("frontier_ms",0))
            modeled=fixed+oracle["pipeline_oracle_wall_ms"]
            summary.append({"dataset":dataset,"workload":workload,"M":capacity,"refill_policy":refill,
                "oracle_profile":profile,
                "repetition":repetition,"baseline":baseline,"rounds":oracle["rounds"],
                "baseline_workload_ms":metrics["workload_ms"],"baseline_kernel_gpu_ms":metrics.get("kernel_gpu_ms",0),
                **oracle,"fixed_host_ms":fixed,"modeled_oracle_workload_ms":modeled,
                "headroom_ms":metrics["workload_ms"]-modeled,
                "speedup_ceiling":metrics["workload_ms"]/modeled if modeled else 0,
                "group_refills":int(metrics.get("group_refills",0)),
                "refill_total_ms":metrics.get("recycle_ms",0),
                "refill_avg_per_group_ms":metrics.get("recycle_ms",0)/metrics.get("group_refills",1) if metrics.get("group_refills",0) else 0,
                "query_latency_p50_ms":percentile(latencies,.5),"query_latency_p95_ms":percentile(latencies,.95),
                "query_latency_p99_ms":percentile(latencies,.99)})
    if summary:
        with (out/"summary.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    audit={"schema":1,"oracle_files":len(oracle_paths),"training_rows":len(training),
           "logical_rounds":len(per_round),"stable_labels":sum(int(r["stable_label"]) for r in winner_rows),
           "candidate_counts":sorted({sum(1 for row in rows) for repetitions in per_round.values()
                                       for rows in repetitions.values()}),"formal_repetitions":2,
           "note":"modeled_oracle_workload_ms uses matched host fixed cost plus the sum of per-round minimum candidate pipeline wall time"}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(training),out/"training_rows.parquet",compression="zstd")
        audit["parquet"]="training_rows.parquet"
    except ImportError:
        audit["parquet"]=None;audit["parquet_note"]="pyarrow unavailable; lossless gzip CSV retained"
    (out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")


if __name__=="__main__":main()
