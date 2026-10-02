#!/usr/bin/env python3
"""Build auditable strong-tail BFS/SSSP workloads on derived graph views.

Natural source eccentricities in social graphs are too concentrated to form a
strong tail.  ``augment`` therefore makes a byte-for-byte-derived CSR view with
small, disconnected path components appended.  Original vertices, edges, and
weights are unchanged.  Path-head queries are real full-graph BFS/SSSP queries;
they simply have a controllable number of relaxation rounds.

The original graph is never modified.  The derived graph must be identified by
GraphWeft before ``workloads`` is run (the CLI prints ``graph=<identity>``).
"""
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import struct
from pathlib import Path

from prepare_workloads import core_keys, weighted_keys

BFS, SSSP = 0, 1


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def read_i32(path):
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError(f"unaligned int32 file: {path}")
    return struct.unpack(f"<{len(raw)//4}i", raw)


def path_lengths(count, minimum, maximum, alpha, seed):
    rng = random.Random(seed)
    values = [min(maximum, int(round(minimum * rng.paretovariate(alpha))))
              for _ in range(count)]
    # Stable order makes the first subset representative for Mixed as well.
    rng.shuffle(values)
    return values


def augment(args):
    src = args.graph.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    row_path, col_path, weight_path = (src / "csr_vlist.bin", src / "csr_elist.bin",
                                       src / "csr_weightlist.bin")
    rows = read_i32(row_path)
    if rows[0] != 0 or rows[-1] < 0:
        raise ValueError("only legacy int32 CSR input is supported")
    original_vertices, original_edges = len(rows) - 1, rows[-1]
    if col_path.stat().st_size != original_edges * 4:
        raise ValueError("CSR edge count mismatch")
    if weight_path.exists() and weight_path.stat().st_size != original_edges * 4:
        raise ValueError("CSR weight count mismatch")

    lengths = path_lengths(args.paths, args.min_edges, args.max_edges, args.alpha, args.seed)
    heads, appended_rows, appended_cols = [], [], []
    vertex = original_vertices
    edge = original_edges
    for edges in lengths:
        heads.append(vertex)
        for position in range(edges + 1):
            appended_rows.append(edge)
            if position < edges:
                appended_cols.append(vertex + position + 1)
                edge += 1
            if not args.directed and position > 0:
                appended_cols.append(vertex + position - 1)
                edge += 1
        vertex += edges + 1
    appended_rows.append(edge)
    if edge >= 2**31 or vertex >= 2**31:
        raise ValueError("derived CSR exceeds legacy int32 range")

    with (out / "csr_vlist.bin").open("wb") as f:
        f.write(struct.pack(f"<{len(rows)-1}i", *rows[:-1]))
        f.write(struct.pack(f"<{len(appended_rows)}i", *appended_rows))
    shutil.copyfile(col_path, out / "csr_elist.bin")
    with (out / "csr_elist.bin").open("ab") as f:
        f.write(struct.pack(f"<{len(appended_cols)}i", *appended_cols))
    if weight_path.exists():
        shutil.copyfile(weight_path, out / "csr_weightlist.bin")
    else:
        with (out / "csr_weightlist.bin").open("wb") as f:
            block = struct.pack("<1048576i", *([1] * 1048576))
            remaining = original_edges
            while remaining:
                take = min(remaining, 1048576)
                f.write(block[:take * 4]); remaining -= take
    with (out / "csr_weightlist.bin").open("ab") as f:
        f.write(struct.pack(f"<{len(appended_cols)}i", *([1] * len(appended_cols))))

    manifest = {
        "schema": 1, "construction": "disjoint_path_components",
        "directed": args.directed, "seed": args.seed, "pareto_alpha": args.alpha,
        "minimum_path_edges": args.min_edges, "maximum_path_edges": args.max_edges,
        "path_edges": lengths, "path_heads": heads,
        "original_graph": str(src), "original_vertices": original_vertices,
        "original_edges": original_edges, "derived_vertices": vertex,
        "derived_edges": edge,
        "original_files_sha256": {p.name: digest(p) for p in (row_path, col_path, weight_path)},
        "derived_files_sha256": {p.name: digest(p) for p in
                                  (out / "csr_vlist.bin", out / "csr_elist.bin",
                                   out / "csr_weightlist.bin")},
        "semantic_note": "Original CSR is unchanged as a prefix; appended components are disconnected paths."
    }
    (out / "augmentation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_candidates(query_path, completion_path):
    queries = {}
    with query_path.open() as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                row = next(csv.reader([line]))
                queries[int(row[0])] = int(row[1])
    with completion_path.open(newline="") as f:
        rounds = {int(r["query_id"]): int(r["service_rounds"]) for r in csv.DictReader(f)}
    if queries.keys() != rounds.keys():
        raise ValueError("candidate queries/completions do not match")
    return [{"candidate_id": qid, "source": source, "rounds": rounds[qid]}
            for qid, source in queries.items()]


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def stats(values):
    return {name: percentile(values, q) for name, q in
            (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99))} | {
                "min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def write_workload(path, identity, rows, note):
    with path.open("w", newline="") as f:
        f.write(f"# graph_identity={identity}\n# capacity=64\n# {note}\n")
        f.write("# id,source,score,offset,feature_key,algorithm,reference_rounds\n")
        csv.writer(f).writerows((r["id"], r["source"], r.get("score", 0),
                                 r.get("offset", 0), r.get("feature_key", 0), r["algorithm"],
                                 r["reference_rounds"]) for r in rows)


def workloads(args):
    graph_manifest = json.loads((args.graph / "augmentation_manifest.json").read_text())
    heads, lengths = graph_manifest["path_heads"], graph_manifest["path_edges"]
    if len(heads) < args.long_count:
        raise ValueError("derived graph has too few path components")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    all_rows, audit_rows = {}, []
    sources = {}
    for algorithm, name, query_path, completion_path, seed in (
            (BFS, "bfs", args.bfs_queries, args.bfs_completion, args.seed + 1),
            (SSSP, "sssp", args.sssp_queries, args.sssp_completion, args.seed + 2)):
        candidates = read_candidates(query_path, completion_path)
        candidates.sort(key=lambda r: (r["rounds"], r["source"]))
        short_pool = candidates[:len(candidates)//2]
        chosen = random.Random(seed).sample(short_pool, 1024 - args.long_count)
        rows = [{"id": algorithm * 1000000 + i, "source": r["source"],
                 "algorithm": algorithm, "reference_rounds": r["rounds"], "label": "short",
                 "provenance": f"candidate:{r['candidate_id']}"} for i, r in enumerate(chosen)]
        rows += [{"id": algorithm * 1000000 + len(rows) + i, "source": heads[i],
                  "algorithm": algorithm, "reference_rounds": lengths[i] + 1,
                  "label": "long", "provenance": f"synthetic_path:{i}"}
                 for i in range(args.long_count)]
        random.Random(seed + 100).shuffle(rows)
        all_rows[name] = rows
        sources[name] = {"query_path": str(query_path.resolve()),
                         "query_sha256": digest(query_path),
                         "completion_path": str(completion_path.resolve()),
                         "completion_sha256": digest(completion_path)}
        write_workload(out / f"{name}_strong_tail.csv", args.identity, rows,
                       f"strong tail; seed={seed}; {args.long_count}/1024 disconnected-path queries")

    mixed = []
    per_algorithm_long = args.long_count // 2
    for name in ("bfs", "sssp"):
        short = [r for r in all_rows[name] if r["label"] == "short"][:512-per_algorithm_long]
        long = [r for r in all_rows[name] if r["label"] == "long"][:per_algorithm_long]
        mixed += [{**r, "id": 2000000 + len(mixed) + i} for i, r in enumerate(short + long)]
    random.Random(args.seed + 200).shuffle(mixed)
    all_rows["mixed"] = mixed
    write_workload(out / "mixed_strong_tail.csv", args.identity, mixed,
                   f"algorithm-internal strong tail; seed={args.seed+200}; 512 BFS + 512 SSSP")

    report = {"schema": 1, "acceptance": {
        "long_median_over_short_median_minimum": 2.0,
        "long_fraction": args.long_count / 1024,
        "mixed_requires_each_algorithm": True}, "graph_identity": args.identity,
        "graph_manifest": str((args.graph / "augmentation_manifest.json").resolve()),
        "graph_manifest_sha256": digest(args.graph / "augmentation_manifest.json"),
        "selection_seed": args.seed, "candidate_provenance": sources, "workloads": {}}
    for workload, rows in all_rows.items():
        report["workloads"][workload] = {}
        for algorithm, name in ((BFS, "bfs"), (SSSP, "sssp")):
            subset = [r for r in rows if r["algorithm"] == algorithm]
            if not subset: continue
            short = [r["reference_rounds"] for r in subset if r["label"] == "short"]
            long = [r["reference_rounds"] for r in subset if r["label"] == "long"]
            ratio = percentile(long, .5) / percentile(short, .5)
            report["workloads"][workload][name] = {
                "queries": len(subset), "short_queries": len(short), "long_queries": len(long),
                "long_fraction": len(long) / len(subset), "all": stats(short + long),
                "short": stats(short), "long": stats(long),
                "long_median_over_short_median": ratio, "strong_tail": ratio >= 2}
        for order, row in enumerate(rows):
            audit_rows.append({"workload": workload, "order": order, **row})
    if not all(x["strong_tail"] for w in report["workloads"].values() for x in w.values()):
        raise RuntimeError("constructed workload failed strong-tail acceptance")
    with (out / "workload_audit.csv").open("w", newline="") as f:
        fields = ["workload", "order", "id", "source", "algorithm", "reference_rounds", "label", "provenance"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(audit_rows)
    (out / "distribution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps({
        "N": 1024, "M": 64, "long_count_homogeneous": args.long_count,
        "long_count_per_algorithm_mixed": per_algorithm_long, "seed": args.seed,
        "graph_identity": args.identity,
        "files_sha256": {p.name: digest(p) for p in out.glob("*.csv")}}, indent=2) + "\n")


def verify(args):
    report = json.loads((args.workloads / "distribution_report.json").read_text())
    result = {"schema": 1, "runs": {}}
    for name in ("bfs", "sssp", "mixed"):
        path = args.completions / f"{name}_completion.csv"
        with path.open(newline="") as f:
            completion_rows = list(csv.DictReader(f))
        observed = {int(r["query_id"]): int(r["service_rounds"]) for r in completion_rows}
        audit = []
        with (args.workloads / "workload_audit.csv").open(newline="") as f:
            audit = [r for r in csv.DictReader(f) if r["workload"] == name]
        if len(completion_rows) != len(observed) or len(observed) != 1024 or len(audit) != 1024 or \
                {int(r["id"]) for r in audit} != observed.keys():
            raise RuntimeError(f"{name}: completion IDs/count mismatch")
        expected = {int(r["id"]): r for r in audit}
        mismatched_rounds = [qid for qid, value in observed.items()
                             if value != int(expected[qid]["reference_rounds"])]
        mismatched_algorithms = [int(r["query_id"]) for r in completion_rows
                                 if int(r["algorithm"]) != int(expected[int(r["query_id"])]["algorithm"])]
        if mismatched_rounds or mismatched_algorithms:
            raise RuntimeError(f"{name}: frozen/observed metadata mismatch")
        fingerprint_path = args.completions / f"{name}_fingerprints.csv"
        fingerprint_rows = []
        if fingerprint_path.exists():
            with fingerprint_path.open(newline="") as f:
                fingerprint_rows = list(csv.DictReader(f))
            fingerprint_ids = [int(r["query_id"]) for r in fingerprint_rows]
            if len(fingerprint_ids) != 1024 or len(set(fingerprint_ids)) != 1024 or \
                    set(fingerprint_ids) != observed.keys():
                raise RuntimeError(f"{name}: fingerprint IDs/count mismatch")
        run = {}
        for algorithm, algorithm_name in ((BFS, "bfs"), (SSSP, "sssp")):
            rows = [r for r in audit if int(r["algorithm"]) == algorithm]
            if not rows: continue
            short = [observed[int(r["id"])] for r in rows if r["label"] == "short"]
            long = [observed[int(r["id"])] for r in rows if r["label"] == "long"]
            ratio = percentile(long, .5) / percentile(short, .5)
            run[algorithm_name] = {"all": stats(short + long), "short": stats(short),
                                   "long": stats(long), "long_fraction": len(long)/len(rows),
                                   "long_median_over_short_median": ratio,
                                   "strong_tail": ratio >= report["acceptance"]["long_median_over_short_median_minimum"]}
        result["runs"][name] = {"queries": len(observed),
                                 "completion_sha256": digest(path),
                                 "fingerprints": len(fingerprint_rows),
                                 "fingerprint_sha256": digest(fingerprint_path) if fingerprint_rows else None,
                                 "reference_round_mismatches": 0,
                                 "algorithm_mismatches": 0,
                                 "algorithms": run}
    if not all(x["strong_tail"] for w in result["runs"].values()
               for x in w["algorithms"].values()):
        raise RuntimeError("measured completions failed strong-tail acceptance")
    (args.workloads / "measured_distribution_report.json").write_text(json.dumps(result, indent=2) + "\n")


def annotate_predicted(args):
    """Freeze graph-only length keys while retaining the strong-tail membership.

    The predictor indexes describe the unmodified source graph.  Appended path
    components are unreachable from its landmarks, so their natural frozen key
    is zero.  This mirrors the online predictors' unreachable encoding without
    using reference_rounds as a scheduling input.
    """
    source = args.workloads.resolve()
    graph_manifest = json.loads(args.graph_manifest.read_text())
    original_vertices = int(graph_manifest["original_vertices"])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    parsed = {}
    for workload in ("bfs", "sssp", "mixed"):
        rows = []
        with (source / f"{workload}_strong_tail.csv").open() as handle:
            for line in handle:
                if line.strip() and not line.startswith("#"):
                    fields = next(csv.reader([line]))
                    rows.append({"id": int(fields[0]), "source": int(fields[1]),
                                 "score": float(fields[2]), "offset": int(fields[3]),
                                 "feature_key": int(fields[4]), "algorithm": int(fields[5]),
                                 "reference_rounds": int(fields[6])})
        parsed[workload] = rows

    bfs_sources = {row["source"] for rows in parsed.values() for row in rows
                   if row["algorithm"] == BFS and row["source"] < original_vertices}
    sssp_sources = {row["source"] for rows in parsed.values() for row in rows
                    if row["algorithm"] == SSSP and row["source"] < original_vertices}
    bfs = core_keys(args.core_index, bfs_sources)
    sssp = weighted_keys(args.weighted_index, sssp_sources)
    keys = {BFS: bfs, SSSP: sssp}

    for workload, rows in parsed.items():
        for row in rows:
            row["feature_key"] = (keys[row["algorithm"]][row["source"]]
                                  if row["source"] < original_vertices else 0)
        write_workload(out / f"{workload}_strong_tail.csv", args.identity, rows,
                       "strong tail with frozen graph-only length-prediction keys; "
                       "appended unreachable components use key=0")

    shutil.copyfile(source / "workload_audit.csv", out / "workload_audit.csv")
    shutil.copyfile(source / "distribution_report.json", out / "distribution_report.json")
    manifest = {
        "schema": 1, "graph_identity": args.identity,
        "source_workloads": str(source), "source_manifest_sha256": digest(source / "manifest.json"),
        "original_vertices": original_vertices,
        "prediction": {
            "bfs": "core-distance-v1", "bfs_index": str(args.core_index.resolve()),
            "bfs_index_sha256": digest(args.core_index),
            "sssp": "weighted-boundary-v4", "sssp_index": str(args.weighted_index.resolve()),
            "sssp_index_sha256": digest(args.weighted_index),
            "appended_component_key": 0,
            "uses_reference_rounds": False,
        },
        "files_sha256": {p.name: digest(p) for p in out.glob("*.csv")},
    }
    (out / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("augment")
    p.add_argument("--graph", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--directed", action="store_true"); p.add_argument("--paths", type=int, default=64)
    p.add_argument("--min-edges", type=int, default=64); p.add_argument("--max-edges", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.5); p.add_argument("--seed", type=int, default=20260929)
    p.set_defaults(func=augment)
    p = sub.add_parser("workloads")
    p.add_argument("--graph", type=Path, required=True); p.add_argument("--identity", required=True)
    p.add_argument("--bfs-queries", type=Path, required=True); p.add_argument("--bfs-completion", type=Path, required=True)
    p.add_argument("--sssp-queries", type=Path, required=True); p.add_argument("--sssp-completion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--long-count", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260929); p.set_defaults(func=workloads)
    p = sub.add_parser("verify")
    p.add_argument("--workloads", type=Path, required=True); p.add_argument("--completions", type=Path, required=True)
    p.set_defaults(func=verify)
    p = sub.add_parser("annotate-predicted")
    p.add_argument("--workloads", type=Path, required=True)
    p.add_argument("--graph-manifest", type=Path, required=True)
    p.add_argument("--identity", required=True)
    p.add_argument("--core-index", type=Path, required=True)
    p.add_argument("--weighted-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=annotate_predicted)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__": main()
