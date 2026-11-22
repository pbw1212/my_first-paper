"""Fetch fixed monthly Pipermail archives and retain only pseudonymous threading metadata."""
import calendar
import gzip
import hashlib
import io
import json
import mailbox
import re
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from email.parser import BytesHeaderParser
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"runs/public_thread_transfer"
LISTS = ("python-dev","python-ideas")
MONTHS = [(2016,m) for m in range(9,13)]+[(y,m) for y in (2017,2018) for m in range(1,13)]+[(2019,1)]
IDS = re.compile(r"<([^<>\s]+)>")


def token(s):
    return hashlib.sha256(s.encode()).hexdigest()


def download(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=45) as reply:
                return reply.read()
        except OSError:
            if attempt==2:
                raise
            time.sleep(1+attempt)


def parse_message(msg):
    refs = IDS.findall(msg.get("References",""))
    parent = IDS.findall(msg.get("In-Reply-To",""))
    own = IDS.findall(msg.get("Message-ID",""))
    addr = parseaddr(msg.get("From","").replace(" at ","@"),strict=False)[1].lower().strip()
    if len(own)!=1 or addr.count("@")!=1:
        return None
    try:
        date = parsedate_to_datetime(msg.get("Date",""))
        if not date or date.tzinfo is None:
            return None
        t = int(date.astimezone(timezone.utc).timestamp())
    except (TypeError,ValueError,OverflowError):
        return None
    # Unresolved reply headers and subject-only replies cannot create new roots.
    reply = bool(msg.get("In-Reply-To") or msg.get("References") or re.search(r"\bre\s*:",msg.get("Subject",""),re.I))
    return [token(own[0]),token(addr)[:20],t,token(parent[-1] if parent else refs[-1]) if parent or refs else None,reply]


def prepare_month(args):
    name,url,key = args
    raw = download(url)
    if raw.startswith(b"\x1f\x8b"):
        data = gzip.decompress(raw)
    else:
        data = raw
    assert data.startswith(b"From "),url
    parser = BytesHeaderParser()
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/"mailbox"
        path.write_bytes(data)
        box = mailbox.mbox(path,create=False,factory=lambda f:parser.parse(io.BytesIO(f.read()),headersonly=True))
        try:
            count = len(box)
            records = [r for msg in box if (r:=parse_message(msg)) is not None]
        finally:
            box.close()
    return name,key,records,dict(url=url,sha256=hashlib.sha256(raw).hexdigest(),messages=count,parsed=len(records))


def prepare():
    tasks = []
    for name in LISTS:
        base = f"https://mail.python.org/pipermail/{name}/"
        page = download(base).decode()
        links = set(re.findall(r'href="([^"]+)"',page,re.I))
        for y,m in MONTHS:
            key = f"{y}-{calendar.month_name[m]}"
            file = next(v for v in (key+".txt",key+".txt.gz") if v in links)
            tasks.append((name,base+file,key))
    all_records = {n:[] for n in LISTS}
    audit = {n:{} for n in LISTS}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name,key,records,info in pool.map(prepare_month,tasks):
            all_records[name].extend(records)
            audit[name][key] = info
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in all_records.items():
        path = OUT/(name+"-headers.jsonl.gz")
        encoded = "".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows).encode()
        path.write_bytes(gzip.compress(encoded,mtime=0))
    (OUT/"archive.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    prepare()
