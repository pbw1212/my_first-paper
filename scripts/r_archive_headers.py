"""Keep an opaque ETH Pipermail sender token without decoding the protected address."""
import gzip
import hashlib
import io
import mailbox
import re
import tempfile
from collections import Counter
from email.parser import BytesHeaderParser
from pathlib import Path

from public_thread_archive import download,parse_message,token


def archive_sender(header):
    value=" ".join(header.split())
    comment=re.fullmatch(r"(.*?)\s+\(.*\)",value)
    return (comment[1] if comment else value).strip()


def prepare(args):
    _,url,_,expected=args
    raw=download(url)
    checksum=hashlib.sha256(raw).hexdigest()
    assert checksum==expected
    plain=gzip.decompress(raw)
    assert plain.startswith(b"From ")
    parser=BytesHeaderParser()
    rows=[]
    skipped=Counter()
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"mbox"
        path.write_bytes(plain)
        box=mailbox.mbox(path,create=False,factory=lambda h:parser.parse(io.BytesIO(h.read()),headersonly=True))
        try:
            count=len(box)
            for m in box:
                who=archive_sender(m.get("From",""))
                if not who:
                    skipped["empty_sender"]+=1
                    continue
                del m["From"]
                m["From"]=token(who)+"@archive.invalid"
                r=parse_message(m)
                if r is None:
                    skipped["date_or_message_id"]+=1
                else:
                    rows.append(r)
        finally:
            box.close()
    assert len(rows)+sum(skipped.values())==count and (rows or not count)
    return rows,dict(url=url,sha256=checksum,messages=count,parsed=len(rows),skipped=dict(skipped))
