"""One-command independent audit or isolated full rebuild. Run from the Git checkout."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/"runs/paper_closure"
STAGES=["budgets","predictability","robustness","ksources","bridge","evidence"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["verify","reproduce"])
    args=p.parse_args()
    if args.action=="verify":
        # All per-WP verifiers reconstruct verification manifests; check existing seals before invoking them.
        subprocess.run([sys.executable,"-B",str(ROOT/"scripts/closure_evidence.py"),"--sealed"],check=True)
        for name in STAGES:
            subprocess.run([sys.executable,"-B",str(ROOT/"scripts"/("closure_"+name+".py")),"--verify"],check=True)
        return
    # A private fresh output root also prevents mixing partial and completed archives on retries.
    with tempfile.TemporaryDirectory(prefix=".closure-rebuild-",dir=ROOT) as temp:
        output=Path(temp)/"runs/paper_closure"
        env=dict(os.environ,CLOSURE_OUTPUT=str(output),CLOSURE_DOC_ROOT=str(Path(temp)/"docs"))
        for name in STAGES:
            subprocess.run([sys.executable,"-B","-u",str(ROOT/"scripts"/("closure_"+name+".py"))],env=env,check=True)
        checked=0
        for original in ARCHIVE.glob("wp*/*"):
            if original.name in ("protocol.json","verification.json") or not original.is_file():
                continue
            other=output/original.relative_to(ARCHIVE)
            if original.suffix==".json":
                a,b=[json.loads(p.read_text(encoding="utf8")) for p in (original,other)]
                if original.parent.name=="wp3_robustness" and original.name=="summary.json":
                    a.pop("stress_wall_seconds");b.pop("stress_wall_seconds")
                assert a==b,original
            elif original.suffix==".gz":
                with gzip.open(original,"rb") as a,gzip.open(other,"rb") as b:
                    assert hashlib.file_digest(a,"sha256").digest()==hashlib.file_digest(b,"sha256").digest(),original
            else:
                assert original.read_bytes()==other.read_bytes(),original
            checked+=1
        assert (Path(temp)/"docs/closure_evidence_pack.md").read_bytes()==(ROOT/"docs/closure_evidence_pack.md").read_bytes()
        assert json.loads((output/"master_summary.json").read_text(encoding="utf8"))==json.loads((ARCHIVE/"master_summary.json").read_text(encoding="utf8"))
        print("Fresh rebuild matches",checked,"scientific outputs plus the evidence pack and master index. Protocol paths and elapsed time are machine-specific.")


if __name__=="__main__":
    main()

