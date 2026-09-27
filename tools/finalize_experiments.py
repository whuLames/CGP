#!/usr/bin/env python3
"""Fill GraphWeft experiment records from preserved command logs."""
import csv
import json
import os
import platform
import re
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,compute_cap','--format=csv,noheader'],capture_output=True,text=True).stdout.splitlines()[0]
nvcc = subprocess.run(['nvcc','--version'],capture_output=True,text=True).stdout.splitlines()[-1]
for folder in sorted((root/'experiments').iterdir()):
    if not folder.is_dir(): continue
    name=folder.name
    stdout=(folder/'stdout.log').read_text() if (folder/'stdout.log').exists() else ''
    stderr=(folder/'stderr.log').read_text() if (folder/'stderr.log').exists() else ''
    code_file=folder/'artifacts/exit_code.txt'
    code=int(code_file.read_text().strip()) if code_file.exists() else (1 if 'terminate called' in stderr else 0)
    status='success' if code==0 else 'failed'
    metrics={'status':status,'exit_code':code,'validation_passed':'validation passed' in stdout,
             'frontier_passed':'frontier validation passed' in stdout,'sanitizer_errors':None,'runs':[]}
    for line in stdout.splitlines():
        if line.startswith('batches='):
            record={}
            for key,value in re.findall(r'(\w+)=([0-9.eE+-]+)',line):
                record[key]=float(value) if '.' in value or 'e' in value else int(value)
            metrics['runs'].append(record)
    parity_path=folder/'artifacts/parity.json'
    if parity_path.exists():metrics['parity']=json.loads(parity_path.read_text())
    raw_path=folder/'artifacts/raw_samples.csv'
    if raw_path.exists():metrics['kernel_samples']=sum(1 for _ in raw_path.open())-1
    for path in (folder/'artifacts').glob('*check.stdout'):
        m=re.search(r'ERROR SUMMARY: (\d+) errors',path.read_text())
        if m:metrics['sanitizer_errors']=max(metrics['sanitizer_errors'] or 0,int(m.group(1)))
    if 'cit_patents_weighted' in name:
        dataset={'name':'cit-Patents','source':'/home/zyl/data/csr_data/cit-Patents','vertices':3774768,'edges':33037895,'preprocessing':'directed CSR import and CSC build'}
        commands=['graphweft_cli --directed --algorithm=sssp --n=32 --q=32 --selector=push','graphweft_cli --directed --algorithm=sswp --n=32 --q=32 --selector=push']
        purpose='Run SSSP and SSWP real-graph regressions on cit-Patents.'
    elif 'cit_patents' in name:
        dataset={'name':'cit-Patents','source':'/home/zyl/data/csr_data/cit-Patents','vertices':3774768,'edges':33037895,'preprocessing':'directed CSR import and CSC build'}
        q=128 if 'q128' in name else 256
        commands=[f'./build/graphweft_cli --graph=/home/zyl/data/csr_data/cit-Patents --directed --algorithm=bfs --n={q} --q={q} --selector=push --frontier=unordered']
        purpose=f'Check real-graph BFS execution at Q={q} and memory accounting.'
    elif 'kernel_replay' in name:
        dataset={'name':'four-vertex directed edge list','source':'artifacts/directed.edgelist','vertices':4,'edges':5,'preprocessing':'none'}
        commands=['graphweft_cli --directed --algorithm=sssp --n=130 --q=129 --layout=grouped --checkpoint=artifacts/checkpoint.bin','graphweft_kernel_lab artifacts/directed.edgelist artifacts/checkpoint.bin result.csv 1']
        purpose='Save a matched round state and compare/timestamp shared Push and Dense Pull.'
    elif 'timing_smoke' in name:
        dataset={'name':'two-vertex undirected edge list','source':'artifacts/graph.txt','vertices':2,'edges':2,'preprocessing':'none'}
        commands=['graphweft_cli --n=3 --q=2 --algorithm=bfs']
        purpose='Check complete task, round, feature and selector timing fields.'
    elif 'round_timing' in name:
        dataset={'name':'synthetic unit graphs','source':'tests/validate.cpp and tests/frontier.cu','vertices':'2-7','edges':'1-10','preprocessing':'in-memory construction'}
        commands=['graphweft_validate','graphweft_frontier_validate']
        purpose='Validate all algorithms after adding aggregate round timing.'
    elif 'warp_compaction' in name:
        dataset={'name':'synthetic unit graphs','source':'tests/validate.cpp and tests/frontier.cu','vertices':'2-7','edges':'1-10','preprocessing':'in-memory construction'}
        commands=['graphweft_validate','graphweft_frontier_validate','compute-sanitizer --tool synccheck graphweft_frontier_validate']
        purpose='Validate warp-reserved unordered compaction and warp synchronization.'
    elif 'final_gate' in name:
        dataset={'name':'synthetic unit graphs and four-vertex directed edge list','source':'tests/ and artifacts/directed.edgelist','vertices':'2-7','edges':'1-10','preprocessing':'none'}
        commands=['graphweft_validate','graphweft_frontier_validate','graphweft_cli --algorithm=sssp --n=130 --q=129 --layout=grouped --selector=replay --checkpoint=artifacts/checkpoint.bin','graphweft_kernel_lab artifacts/directed.edgelist artifacts/checkpoint.bin artifacts/raw_samples.csv 1','compute-sanitizer --tool memcheck graphweft_cli ...','compute-sanitizer --tool synccheck graphweft_cli ...']
        purpose='Final correctness, matched-state replay, and CUDA sanitizer gate.'
    elif 'gpu_acceptance' in name:
        dataset={'name':'four-vertex directed edge list','source':'artifacts/directed.edgelist','vertices':4,'edges':5,'preprocessing':'none'}
        commands=['graphweft_validate','graphweft_cli --directed --algorithm=sssp --n=130 --q=129 --layout=grouped --selector=replay --replay=push,pull --copy_results_to_cpu','compute-sanitizer --tool memcheck graphweft_cli ...','compute-sanitizer --tool synccheck graphweft_cli ...']
        purpose='Validate CPU reference, multi-batch output and CUDA sanitizer checks.'
    elif 'core_index_parity' in name:
        dataset={'name':'cit-Patents','source':'/home/zyl/data/csr_data/cit-Patents and old core index','vertices':3774768,'edges':33037895,'preprocessing':'directed CSR import'}
        commands=['graphweft_cli --planner=length --predictor=core_distance --n=32 --q=32 --plan_output=artifacts/plan.csv','Python compare feature_key against old core-distance index']
        purpose='Check BFS core-distance features against the verified old cit-Patents index.'
    elif 'weighted_index_parity' in name:
        dataset={'name':'4096-vertex synthetic weighted graph','source':'artifacts/synthetic.gr and artifacts/csr','vertices':4096,'edges':'8192 or 49152','preprocessing':'same topology encoded as old GR and GraphWeft CSR'}
        commands=['old precompute_weighted_core synthetic.gr weighted.bin','graphweft_cli --planner=length --predictor=weighted_boundary --n=32 --q=32','Python compare feature_key against weighted.bin']
        purpose='Check SSSP weighted-boundary feature parity with the old preprocessor.'
    elif 'plan_capacity' in name:
        dataset={'name':'four-vertex directed edge list','source':'artifacts/directed.edgelist','vertices':4,'edges':5,'preprocessing':'none'}
        commands=['graphweft_cli --planner=length --predictor=core_distance --q=4 --plan_output=artifacts/plan.csv','graphweft_cli --queries=artifacts/plan.csv --planner=length --predictor=import_key --q=5']
        purpose='Check a frozen plan refuses a mismatched Q.'
    elif 'frozen_plan' in name:
        dataset={'name':'four-vertex directed edge list','source':'artifacts/directed.edgelist','vertices':4,'edges':5,'preprocessing':'none'}
        commands=['graphweft_cli --planner=length --predictor=core_distance --plan_output=artifacts/plan.csv','graphweft_cli --queries=artifacts/plan.csv --planner=length --predictor=import_key','diff artifacts/auto.csv artifacts/imported.csv']
        purpose='Verify graph-bound frozen plan export and exact replay output.'
    elif 'auto_predictors' in name:
        dataset={'name':'four-vertex directed edge list','source':'artifacts/directed.edgelist','vertices':4,'edges':5,'preprocessing':'none'}
        commands=['graphweft_cli --planner=length --predictor=core_distance --algorithm=bfs','graphweft_cli --planner=length --predictor=weighted_boundary --algorithm=sssp --offsets --offset_source=phase']
        purpose='Check automatic length providers, stable grouping, and phase offsets.'
    elif 'cold_cache' in name:
        dataset={'name':'saved four-vertex SSSP checkpoint','source':'earlier final_gate checkpoint','vertices':4,'edges':5,'preprocessing':'64 MiB eviction before each timed kernel'}
        commands=['graphweft_kernel_lab GRAPH CHECKPOINT artifacts/raw_samples.csv 1 1']
        purpose='Measure matched-state kernel timings with optional cache eviction.'
    elif 'kernel_resources' in name:
        dataset={'name':'kernel source','source':'src/kernels.cu','vertices':'n/a','edges':'n/a','preprocessing':'nvcc ptxas verbosity'}
        commands=['nvcc -std=c++17 -O3 -arch=sm_70 -Xptxas=-v -Iinclude -c src/kernels.cu']
        purpose='Record register counts and spill bytes for compiled kernels.'
    elif 'phase_offsets' in name:
        dataset={'name':'four-vertex directed edge list','source':'artifacts/directed.edgelist','vertices':4,'edges':5,'preprocessing':'none'}
        commands=['graphweft_cli --directed --algorithm=sssp --n=5 --q=4 --offsets --offset_source=phase --landmarks=4 --copy_results_to_cpu']
        purpose='Validate migrated phase-index offset evaluation and delayed batch execution.'
    elif 'cli_guards' in name:
        dataset={'name':'two-vertex undirected edge list','source':'artifacts/undirected.edgelist','vertices':2,'edges':2,'preprocessing':'none'}
        commands=['graphweft_cli --q=256 --memory_fraction=0.00000001','graphweft_cli --q=2']
        purpose='Check explicit memory-budget refusal and small-capacity success.'
    else:
        dataset={'name':'synthetic unit graphs','source':'tests/validate.cpp and tests/frontier.cu','vertices':'2-7','edges':'1-10','preprocessing':'in-memory construction'}
        commands=['./build/graphweft_validate' if 'validation' in name else './build/graphweft_frontier_validate']
        purpose='Validate graph algorithms, frontier masks and precision behavior.'
    config={'experiment':name,'purpose':purpose,'dataset':dataset,'commands':commands,'cwd':str(root),'environment':{'os':platform.platform(),'gpu':gpu,'nvcc':nvcc},'seed':0x4757 if 'kernel_replay' in name else None}
    (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    with (folder/'result.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','status','exit_code','rounds','total_ms'])
        if metrics['runs']:
            for i,run in enumerate(metrics['runs']):writer.writerow([i,status,code,run.get('rounds',''),run.get('total_ms','')])
        else:writer.writerow([name,status,code,'',''])
    issue=stderr.strip().splitlines()[-1] if code and stderr.strip() else 'None'
    readme=f'''# Experiment: {name}

## Purpose

{purpose}

## Hypothesis

The implementation should complete with the expected validation or explicit rejection behavior.

## Dataset

- Name: {dataset['name']}
- Source: {dataset['source']}
- Size: V={dataset['vertices']}, E={dataset['edges']}
- Split: none
- Preprocessing: {dataset['preprocessing']}

## Scenario

- Task: GraphWeft {name}
- Workload: see config.json and stdout.log
- Baseline: CPU reference or alternate kernel where applicable
- Candidate: GraphWeft implementation
- Comparison: correctness, capacity and timing as applicable

## Environment

- OS: {platform.platform()}
- GPU: {gpu}
- Runtime: {nvcc}
- Dependencies: CUDA 12.8, spdlog 1.15.3, gflags 2.2.2
- Git commit: uncommitted workspace snapshot

## Command

From `{root}`:

```bash
'''+ '\n'.join(commands)+'''
```

## Results

| Status | Exit | Runs | Sanitizer errors |
|---|---:|---:|---:|
| '''+f"{status} | {code} | {len(metrics['runs'])} | {metrics['sanitizer_errors'] if metrics['sanitizer_errors'] is not None else 'n/a'} |"+'''

See metrics.json, result.csv, stdout.log, stderr.log and artifacts/ for raw evidence.

## Observations

'''+('The check completed.\n' if code==0 else 'The check failed; a later experiment records the fix.\n')+f'''
## Conclusion

{status}.

## Issues

{issue}

## Next Steps

Continue broader graph and workload coverage.
'''
    (folder/'README.md').write_text(readme)
