#!/usr/bin/env python3
"""Download, extract, symmetrize, validate, and index one catalog graph."""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

GIB=1<<30


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");temporary.replace(path)


def pause(root,reason,extra):
    state={"status":"paused","reason":reason,"utc":datetime.now(timezone.utc).isoformat(),
           "free_bytes":shutil.disk_usage(root).free,**extra}
    atomic_json(root/"PAUSED_INSUFFICIENT_SPACE.json",state)
    print("\n*** GRAPHWEFT DATA PREPARATION PAUSED: insufficient /data space. ***",flush=True)
    print(json.dumps(state,indent=2),flush=True)


def run_monitored(command,root,log,low=25*GIB):
    with log.open("w") as output:
        process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,text=True)
        while process.poll() is None:
            if shutil.disk_usage(root).free<low:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                pause(root,"runtime free space below 25 GiB",{"command":command,"log":str(log)})
                return False
            time.sleep(10)
    if process.returncode:raise RuntimeError(f"command failed ({process.returncode}): {command}; see {log}")
    return True


def extract_monitored(command,target_path,root,log,low=25*GIB):
    with target_path.open("wb") as target,log.open("w") as errors:
        process=subprocess.Popen(command,stdout=target,stderr=errors)
        while process.poll() is None:
            if shutil.disk_usage(root).free<low:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                pause(root,"runtime free space below 25 GiB",{"command":command,"log":str(log)})
                return False
            time.sleep(10)
    if process.returncode:raise RuntimeError(f"extraction failed ({process.returncode}); see {log}")
    return True


def choose_member(archive):
    allowed={".mtx",".edges",".edge",".txt",".csv"};blocked=("readme","label","node","attr","feat","coord")
    with zipfile.ZipFile(archive) as bundle:
        candidates=[item for item in bundle.infolist() if not item.is_dir() and
                    Path(item.filename).suffix.lower() in allowed and
                    not any(word in item.filename.lower() for word in blocked)]
    if not candidates:raise RuntimeError("archive has no supported edge-list or MatrixMarket member")
    return max(candidates,key=lambda item:item.file_size)


def allocation_bytes(vertices,edges,capacity):
    words=(capacity+63)//64
    return (8*(vertices+1)+8*edges+8*vertices*capacity+16*vertices*words+12*vertices+
            max(4*capacity,8*words)+4*capacity+2*capacity+capacity+4*capacity+
            ((capacity+31)//32)*32+vertices+4*vertices+40)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--catalog",type=Path,required=True)
    parser.add_argument("--name",required=True);parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--converter",type=Path,required=True);args=parser.parse_args()
    catalog=json.loads(args.catalog.read_text());entries={entry["name"]:entry for entry in catalog["datasets"]}
    if args.name not in entries:raise ValueError(f"unknown dataset: {args.name}")
    entry=entries[args.name];root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    dataset=root/"datasets"/args.name;status=root/"status";temporary=root/"tmp"/args.name
    status.mkdir(parents=True,exist_ok=True);temporary.mkdir(parents=True,exist_ok=True)
    complete=status/f"{args.name}.json"
    if complete.exists() and json.loads(complete.read_text()).get("status")=="success":
        print("already complete",args.name);return
    expected_archive=int(entry.get("archive_bytes",0));expected_raw=int(entry.get("raw_bytes",expected_archive*5))
    expected_csr=entry.get("vertices",0)*4+entry.get("edges",0)*16
    projected=expected_archive+expected_raw+expected_csr+30*GIB
    if shutil.disk_usage(root).free<projected:
        pause(root,"projected peak would violate 30 GiB reserve",{"dataset":args.name,"projected_bytes":projected})
        raise SystemExit(2)
    archive=temporary/(args.name+".zip");edge_text=temporary/(args.name+".input")
    record={"status":"running","entry":entry,"started_utc":datetime.now(timezone.utc).isoformat()}
    atomic_json(complete,record)
    try:
        if not run_monitored(["curl","-fL","--retry","5","--continue-at","-","--output",str(archive),entry["url"]],
                             root,temporary/"download.log"):raise SystemExit(2)
        member=choose_member(archive)
        if not extract_monitored(["unzip","-p",str(archive),member.filename],edge_text,root,temporary/"extract.log"):
            raise SystemExit(2)
        indexing=entry.get("indexing","auto");vertices=str(entry.get("vertices",0));seed=str(catalog.get("weight_seed",20261007))
        if not run_monitored([str(args.converter),str(edge_text),str(dataset),indexing,vertices,seed],
                             root,temporary/"convert.log"):raise SystemExit(2)
        conversion=json.loads((dataset/"conversion_manifest.json").read_text())
        files={name:sha256(dataset/name) for name in ("csr_vlist.bin","csr_elist.bin","csr_weightlist.bin")}
        identities=[]
        for other in (root/"status").glob("*.json"):
            if other==complete:continue
            try:value=json.loads(other.read_text())
            except (OSError,json.JSONDecodeError):continue
            if value.get("files_sha256")==files:identities.append(value.get("dataset",other.stem))
        budget=int(.8*32*GIB)
        allocation={str(capacity):allocation_bytes(conversion["vertices"],conversion["symmetric_edges"],capacity)
                    for capacity in (128,256)}
        record.update({"status":"success","dataset":args.name,"archive_member":member.filename,
                       "archive_sha256":sha256(archive),"conversion":conversion,"files_sha256":files,
                       "duplicate_of":identities,"allocation_bytes":allocation,
                       "fits_v100_32g_80pct":{key:value<=budget for key,value in allocation.items()},
                       "completed_utc":datetime.now(timezone.utc).isoformat()})
        atomic_json(complete,record)
        archive.unlink();edge_text.unlink()
        print(json.dumps(record,indent=2))
    except Exception as error:
        record.update({"status":"failed","dataset":args.name,"error":repr(error),
                       "failed_utc":datetime.now(timezone.utc).isoformat()})
        atomic_json(complete,record)
        raise


if __name__=="__main__":main()
