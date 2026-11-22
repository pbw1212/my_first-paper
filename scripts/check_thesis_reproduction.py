"""Re-run the frozen workflow in isolation and byte-compare its scientific outputs."""

from pathlib import Path
from tempfile import TemporaryDirectory

from threadpoolctl import threadpool_limits

import thesis_holdout as experiment
from thesis_holdout import sha, write_json


def check():
    reference=experiment.OUT
    with TemporaryDirectory() as tmp, threadpool_limits(limits=1,user_api="blas"):
        experiment.OUT=Path(tmp)
        experiment.freeze()
        for external in (False,True):
            experiment.generate(external)
            experiment.predict(external)
        files=sorted(p.relative_to(tmp) for p in Path(tmp).rglob("*") if p.is_file())
        digests={}
        for rel in files:
            digest=sha(Path(tmp)/rel)
            assert sha(reference/rel)==digest,str(rel)
            digests[rel.as_posix()]=digest
    write_json(reference/"reproduction.json",dict(files=len(files),sha256=digests))
    print(len(files),"files reproduced identically")


if __name__=="__main__":
    check()
