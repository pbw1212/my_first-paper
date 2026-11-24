"""Recount dataset inventory from the inputs identified by the locked archive."""
from datetime import datetime, timezone
from pathlib import Path
import json

from closure_common import ARCH, REPO, dept4, fg, libc, rl, sha
from closure_discussions import finish, messages, LISTS
from probe_cascade_yield import load_events
from masked_pair_benchmark import company

RUN = REPO / "runs"
DEST = REPO / "paper_dasfaa" / "evidence" / "inventory.json"


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def stream_stats(stream):
    nodes = {x for _, u, v in stream for x in (u, v)}
    times = [t for t, _, _ in stream]
    return dict(nodes=len(nodes), records=len(stream), days=round((max(times)-min(times))/86400, 1))


def main():
    seals = load(RUN/"paper_closure/wp2_predictability/protocol.json")["inputs"]
    for path in (ARCH/"data/raw").glob("*"):
        key = "archive/data/raw/" + path.name
        if key in seals:
            assert sha(path) == seals[key], path
    sim = {}
    for n, filename in [("Dept2", "email-Eu-core-temporal-Dept2.txt.gz"),
                        ("Dept4","email-Eu-core-temporal-Dept4.txt.gz"), ("CollegeMsg","CollegeMsg.txt.gz")]:
        sim[n] = stream_stats(load_events(ARCH/"data/raw"/filename))
    sim["company"] = stream_stats(company())
    enron = ARCH/"runs/temporal_transfer/Enron/contacts.txt.gz"
    prov = load(RUN/"temporal_transfer/Enron/provenance.json")
    assert sha(enron)==prov["contacts_sha256"]
    sim["Enron"] = stream_stats(load_events(enron))
    assert (sim["Enron"]["nodes"],sim["Enron"]["records"])==(prov["vertices"],prov["distinct"])
    network = load(RUN/"paper_closure/wp2_predictability/summary.json")["networks"]
    real = {}
    for key,s in network.items():
        if not key.startswith("real "):
            continue
        name = key[5:]
        msg = messages(name,False) if name in LISTS else finish.cohort(name)[0]
        months = set(s["keys"])
        selected = [a for a,t,_,_ in msg.values()
                    if datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m") in months]
        real[name] = dict(nodes=len(set(selected)),records=len(selected),start=min(months),end=max(months))
        if name in LISTS:
            late=messages(name,True)
            periods={"2019-02","2019-03","2019-04"}
            authors=[a for a,t,_,_ in late.values()
                     if datetime.fromtimestamp(t,timezone.utc).strftime("%Y-%m") in periods]
            real[name+"_late"]=dict(nodes=len(set(authors)),records=len(authors),start=min(periods),end=max(periods))
        if name=="libc-alpha":
            # All resolved nonself reply opportunities in the evaluation years,
            # before restricting to each month's closed roster and fortnight.
            from public_thread_transfer import thread_forest
            _,contacts=thread_forest(msg)
            stream=[(t,u,v) for t,u,v in contacts
                    if datetime.fromtimestamp(t,timezone.utc).strftime("%Y-%m") in months]
            sim["GNU libc"]=stream_stats(stream)
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text(json.dumps(dict(sim=sim,real=real),indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
