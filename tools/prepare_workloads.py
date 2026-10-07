#!/usr/bin/env python3
"""Create GraphWeft's frozen N=1024 workloads without changing the graph.

The command is deliberately split in two.  ``sample`` creates the ordinary
workloads and candidate pools.  Run both candidate files through GraphWeft
with M=64 and --completion_output, then ``mixed`` selects and freezes the
natural short/long workload from those measured completion rounds.
"""
import argparse
import csv
import json
import hashlib
import mmap
import random
import struct
from pathlib import Path

BFS, SSSP = 0, 1
CORE_MAGIC = 0x434F524544495354
WEIGHTED_MAGIC = 0x57434F5245444953
U16 = 65535


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20): value.update(block)
    return value.hexdigest()


def core_keys(path, sources):
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        magic, version, vertices, tiers, hubs = struct.unpack_from("<QIIII", data, 0)
        if magic != CORE_MAGIC or version != 1 or tiers != 4:
            raise ValueError(f"invalid core-distance index: {path}")
        base = 24 + 4 * tiers + 4 * hubs
        if len(data) != base + 2 * tiers * vertices:
            raise ValueError(f"invalid core-distance index length: {path}")
        result = {}
        for source in sources:
            if source >= vertices: raise ValueError("source exceeds core index")
            key = 0
            for tier in range(tiers):
                value = struct.unpack_from("<H", data, base + 2 * (tier * vertices + source))[0]
                key = (key << 16) | (0 if value == U16 else value + 1)
            result[source] = key
        return result


def weighted_keys(path, sources):
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        magic, version, vertices, core_tiers, hubs, count = struct.unpack_from("<QIIIII", data, 0)
        if magic != WEIGHTED_MAGIC or version != 4:
            raise ValueError(f"invalid weighted-boundary index: {path}")
        landmark_tiers = struct.unpack_from(f"<{count}I", data, 28 + 4 * core_tiers)
        base = 28 + 4 * core_tiers + 4 * count + 4 * hubs + 4 * landmark_tiers[-1]
        maximum_hop = base + 4 * core_tiers * vertices + 2 * core_tiers * vertices
        maximum_distance = maximum_hop + 2 * count * vertices
        mean_hop = maximum_distance + 4 * count * vertices
        reachable = mean_hop + 2 * count * vertices
        if len(data) != reachable + 2 * count * vertices:
            raise ValueError(f"invalid weighted-boundary index length: {path}")
        tier = count - 1; result = {}
        for source in sources:
            if source >= vertices: raise ValueError("source exceeds weighted index")
            maximum = struct.unpack_from("<H", data, maximum_hop + 2 * (tier * vertices + source))[0]
            mean = struct.unpack_from("<H", data, mean_hop + 2 * (tier * vertices + source))[0]
            result[source] = mean + 128 * maximum
        return result


def sources_with_edges(path: Path):
    if path.is_dir():
        raw = (path / "csr_vlist.bin").read_bytes()
        if len(raw) % 4:
            raise ValueError("csr_vlist.bin is not an int32 offset array")
        offsets = struct.unpack(f"<{len(raw)//4}i", raw)
        return [v for v in range(len(offsets) - 1) if offsets[v + 1] > offsets[v]]
    seen = set()
    with path.open() as f:
        for line in f:
            line = line.partition("#")[0].strip()
            if line:
                seen.add(int(line.split()[0]))
    return sorted(seen)


def write_queries(path, identity, rows, note, capacity=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write(f"# graph_identity={identity}\n# capacity={capacity}\n# {note}\n")
        f.write("# id,source,score,offset,feature_key,algorithm,reference_rounds\n")
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)


def sample(args):
    eligible = sources_with_edges(args.graph)
    if len(eligible) < 5120:
        raise ValueError(f"need at least 5120 positive-outdegree vertices, found {len(eligible)}")
    ordinary = random.Random(args.ordinary_seed).sample(eligible, 1024)
    excluded = set(ordinary)
    candidate_population = [v for v in eligible if v not in excluded]
    candidates = random.Random(43).sample(candidate_population, 4096)
    candidate_set = set(candidates)
    calibration_population = [v for v in candidate_population if v not in candidate_set]
    calibration = random.Random(45).sample(calibration_population, 1024)
    out = args.output
    write_queries(out / "bfs.csv", args.identity,
                  [(i, s, 0, 0, 0, BFS, 0) for i, s in enumerate(ordinary)],
                  f"seed={args.ordinary_seed}; uniform without replacement",args.capacity)
    write_queries(out / "sssp.csv", args.identity,
                  [(100000 + i, s, 0, 0, 0, SSSP, 0) for i, s in enumerate(ordinary)], "same sources/order as bfs.csv",args.capacity)
    for name, algorithm, base in (("bfs_candidates.csv", BFS, 200000), ("sssp_candidates.csv", SSSP, 300000)):
        write_queries(out / name, args.identity,
                      [(base + i, s, 0, 0, 0, algorithm, 0) for i, s in enumerate(candidates)],
                      f"seed=43; excludes ordinary sources; measure with M={args.capacity}",args.capacity)
    write_queries(out / "calibration_bfs.csv", args.identity,
                  [(400000 + i, s, 0, 0, 0, BFS, 0) for i, s in enumerate(calibration)],
                  "seed=45; excludes ordinary and Mixed-tail candidate pools; mapping calibration only",args.capacity)
    write_queries(out / "calibration_sssp.csv", args.identity,
                  [(500000 + i, s, 0, 0, 0, SSSP, 0) for i, s in enumerate(calibration)],
                  "same calibration sources/order as calibration_bfs.csv",args.capacity)
    (out / "manifest.json").write_text(json.dumps({
        "N": 1024, "M": args.capacity, "ordinary_seed": args.ordinary_seed, "candidate_seed": 43,
        "mixed_shuffle_seed": 44, "calibration_seed": 45, "graph_identity": args.identity,
        "positive_outdegree_vertices": len(eligible), "candidate_count_per_algorithm": 4096,
    }, indent=2) + "\n")


def read_queries(path):
    result = {}
    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            row = next(csv.reader([line]))
            result[int(row[0])] = {"id": int(row[0]), "source": int(row[1]), "score": float(row[2]),
                                   "feature_key": int(row[4]), "algorithm": int(row[5])}
    return result


def read_rounds(path):
    with path.open(newline="") as f:
        return {int(r["query_id"]): int(r["service_rounds"]) for r in csv.DictReader(f)}


def choose(candidates, rounds, seed):
    records = [{**q, "rounds": rounds[qid]} for qid, q in candidates.items()]
    records.sort(key=lambda x: (x["rounds"], x["source"]))
    short_pool = records[:len(records)//2]
    short = random.Random(seed).sample(short_pool, 448)
    # The ascending (rounds, source) order gives a deterministic opposite end
    # even when every candidate has the same round count.
    long = records[-64:]
    tagged = [(x, "short") for x in short] + [(x, "long") for x in long]
    short_ids = {x["id"] for x in short}
    long_ids = {x["id"] for x in long}
    if short_ids & long_ids:
        raise ValueError("short and long selections overlap")
    return tagged


def median(values):
    values = sorted(values); n = len(values)
    return values[n//2] if n % 2 else (values[n//2-1] + values[n//2]) / 2


def grouping_quality(records, key):
    ordered = sorted(records, key=key); waits = []; spans = []
    for begin in range(0, len(ordered), 8):
        rounds = [x["rounds"] for x in ordered[begin:begin + 8]]
        maximum = max(rounds); waits.append(sum(maximum - x for x in rounds)); spans.append(maximum - min(rounds))
    return {"groups": len(waits), "slot_rounds": sum(waits), "mean_span": sum(spans) / len(spans),
            "median_span": median(spans), "max_span": max(spans)}


def mixed(args):
    selected = []
    for index, (query_file, completion_file) in enumerate(((args.bfs_queries, args.bfs_completion),
                                                            (args.sssp_queries, args.sssp_completion))):
        selected += choose(read_queries(query_file), read_rounds(completion_file), 4300 + index)
    random.Random(44).shuffle(selected)
    bfs_sources = [x[0]["source"] for x in selected if x[0]["algorithm"] == BFS]
    sssp_sources = [x[0]["source"] for x in selected if x[0]["algorithm"] == SSSP]
    if args.defer_features:
        bfs_predicted = {source: 0 for source in bfs_sources}
    elif args.core_index:
        bfs_predicted = core_keys(args.core_index, bfs_sources)
    elif args.bfs_features:
        bfs_predicted = {q["source"]: q["feature_key"] for q in read_queries(args.bfs_features).values()}
    else: raise ValueError("provide --core-index or --bfs-features")
    if args.defer_features:
        sssp_predicted = {source: 0 for source in sssp_sources}
    elif args.weighted_index:
        sssp_predicted = weighted_keys(args.weighted_index, sssp_sources)
    elif args.sssp_features:
        sssp_predicted = {q["source"]: q["feature_key"] for q in read_queries(args.sssp_features).values()}
    else: raise ValueError("provide --weighted-index or --sssp-features")
    predicted = {BFS: bfs_predicted, SSSP: sssp_predicted}
    rows, audit = [], []
    for order, (record, label) in enumerate(selected):
        feature_key = predicted[record["algorithm"]][record["source"]]
        rows.append((record["id"], record["source"], record["score"], 0, feature_key,
                     record["algorithm"], record["rounds"]))
        audit.append({"order": order, **record, "feature_key": feature_key, "label": label})
    write_queries(args.output / "mixed_tail.csv", args.identity, rows,
                  "448 short + 64 longest per algorithm; shuffled seed=44; reference_rounds is diagnostic only")
    for algorithm, name in ((BFS, "bfs"), (SSSP, "sssp")):
        write_queries(args.output / f"mixed_tail_{name}_selected.csv", args.identity,
                      [row for row in rows if row[5] == algorithm],
                      "selected Mixed-tail members before same-algorithm grouping")
    with (args.output / "mixed_tail_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order", "id", "source", "score", "feature_key", "algorithm", "rounds", "label"])
        w.writeheader(); w.writerows(audit)
    report = {}
    for algorithm, name in ((BFS, "bfs"), (SSSP, "sssp")):
        short = [x["rounds"] for x in audit if x["algorithm"] == algorithm and x["label"] == "short"]
        long = [x["rounds"] for x in audit if x["algorithm"] == algorithm and x["label"] == "long"]
        ratio = median(long) / median(short) if median(short) else None
        algorithm_rows = [x for x in audit if x["algorithm"] == algorithm]
        report[name] = {"short_min": min(short), "short_median": median(short), "short_max": max(short),
                        "long_min": min(long), "long_median": median(long), "long_max": max(long),
                        "R_tail": ratio, "strong_tail": ratio is not None and ratio >= 2,
                        "grouping": {
                            "fifo": grouping_quality(algorithm_rows, key=lambda x: x["order"]),
                            "predicted": grouping_quality(algorithm_rows, key=lambda x: (x["feature_key"], x["source"])),
                            "oracle": grouping_quality(algorithm_rows, key=lambda x: (x["rounds"], x["source"])),
                        }}
    report["indexes"] = {"deferred": args.defer_features}
    if not args.defer_features:
        for name, path in (("core", args.core_index or args.bfs_features),
                           ("weighted", args.weighted_index or args.sssp_features)):
            report["indexes"][name] = str(path); report["indexes"][name + "_sha256"] = sha256(path)
    (args.output / "mixed_tail_report.json").write_text(json.dumps(report, indent=2) + "\n")


def annotate(args):
    records = list(read_queries(args.input).values())
    sources = [q["source"] for q in records]
    if args.algorithm == "bfs": keys = core_keys(args.index, sources); algorithm = BFS
    else: keys = weighted_keys(args.index, sources); algorithm = SSSP
    rows = [(q["id"], q["source"], q["score"], 0, keys[q["source"]], algorithm, 0) for q in records]
    write_queries(args.output, args.identity, rows,
                  f"graph-only {args.algorithm} prediction keys from {args.index}; sha256={sha256(args.index)}")


def merge_features(args):
    original = list(read_queries(args.input).values())
    features = read_queries(args.features)
    by_source = {q["source"]: q["feature_key"] for q in features.values()}
    if len(by_source) != len(features):
        raise ValueError("feature file contains duplicate sources")
    algorithm = BFS if args.algorithm == "bfs" else SSSP
    missing = [q["source"] for q in original if q["source"] not in by_source]
    if missing:
        raise ValueError(f"feature file is missing {len(missing)} sources")
    rows = [(q["id"], q["source"], q["score"], 0, by_source[q["source"]], algorithm, 0)
            for q in original]
    write_queries(args.output, args.identity, rows,
                  f"original input order with graph-built {args.algorithm} keys from {args.features}; "
                  f"sha256={sha256(args.features)}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("sample")
    p.add_argument("--graph", type=Path, required=True); p.add_argument("--identity", required=True)
    p.add_argument("--output", type=Path, required=True);p.add_argument("--capacity",type=int,default=64)
    p.add_argument("--ordinary-seed",type=int,default=42);p.set_defaults(func=sample)
    p = sub.add_parser("mixed")
    p.add_argument("--identity", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bfs-queries", type=Path, required=True); p.add_argument("--bfs-completion", type=Path, required=True)
    p.add_argument("--sssp-queries", type=Path, required=True); p.add_argument("--sssp-completion", type=Path, required=True)
    p.add_argument("--core-index", type=Path); p.add_argument("--weighted-index", type=Path)
    p.add_argument("--bfs-features", type=Path); p.add_argument("--sssp-features", type=Path)
    p.add_argument("--defer-features", action="store_true")
    p.set_defaults(func=mixed)
    p = sub.add_parser("annotate")
    p.add_argument("--identity", required=True); p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--algorithm", choices=("bfs", "sssp"), required=True)
    p.add_argument("--index", type=Path, required=True); p.set_defaults(func=annotate)
    p = sub.add_parser("merge-features")
    p.add_argument("--identity", required=True); p.add_argument("--input", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--algorithm", choices=("bfs", "sssp"), required=True)
    p.set_defaults(func=merge_features)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
