"""Recover archive headers returned by Python's email parser as Header objects."""
import gzip
import hashlib
import io
import json
import mailbox
from concurrent.futures import ThreadPoolExecutor
from email.header import Header
from email.parser import BytesHeaderParser
from pathlib import Path
import tempfile

import role_evidence_time_holdout as holdout
from public_thread_archive import download,parse_message


class TextHeaders:
    def __init__(self,msg):
        self.msg=msg
        self.converted=set()

    def get(self,key,default=""):
        value=self.msg.get(key,default)
        if isinstance(value,Header):
            self.converted.add(key)
            return str(value)
        return value


def recover(url):
    label=url.rsplit("/",1)[1].split(".")[0]
    dst=holdout.OUT/"headers"/(label+".jsonl.gz")
    meta=dst.with_suffix(".json")
    if dst.exists() and meta.exists():
        cfg=holdout.fg.load(meta)
        assert cfg["url"]==url and cfg["headers_sha256"]==holdout.fg.digest(dst)
        return cfg
    raw=download(url)
    data=gzip.decompress(raw)
    assert data.startswith(b"From ")
    rows=[]
    converted=0
    counter=0
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/"mailbox"
        p.write_bytes(data)
        parser=BytesHeaderParser()
        box=mailbox.mbox(p,create=False,factory=lambda h:parser.parse(io.BytesIO(h.read()),headersonly=True))
        try:
            for msg in box:
                counter+=1
                wrapper=TextHeaders(msg)
                result=parse_message(wrapper)
                if wrapper.converted:
                    converted+=1
                else:
                    assert result==parse_message(msg)
                if result is not None:
                    rows.append(result)
        finally:
            box.close()
    assert rows
    tmp=dst.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress("".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows).encode(),mtime=0))
    tmp.replace(dst)
    cfg=dict(url=url,sha256=hashlib.sha256(raw).hexdigest(),messages=counter,parsed=len(rows),
             converted_messages=converted,headers_sha256=holdout.fg.digest(dst))
    holdout.fg.write_json(meta,cfg)
    return cfg


if __name__=="__main__":
    spec=holdout.fg.load(holdout.OUT/"protocol.json")
    assert spec==holdout.config()
    (holdout.OUT/"headers").mkdir(exist_ok=True)
    with ThreadPoolExecutor(3) as pool:
        meta=list(pool.map(recover,spec["urls"]))
    holdout.fg.write_json(holdout.OUT/"archives.json",meta)
    holdout.fg.write_json(holdout.OUT/"ingestion.json",dict(
        correction="Convert non-string email.header.Header values to text before the original parser. Unchanged URL manifest and frozen models.",
        converted_messages=sum(v.get("converted_messages",0) for v in meta),archives=len(meta),
        parsed=sum(v["parsed"] for v in meta),messages=sum(v["messages"] for v in meta),script=holdout.fg.digest(Path(__file__))))
