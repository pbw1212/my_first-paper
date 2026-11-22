"""Exact conditional partner distribution given a complete, parent-labelled simulation trace."""
import numpy as np
from scipy.special import softmax

from build_dept3_pilot import P


class CompleteTrace:
    def __init__(self,graph,stream):
        self.nodes=graph["nodes"]
        self.index={v:i for i,v in enumerate(self.nodes)}
        edges=[(t,self.index[u],self.index[v]) for t,batch in stream for u,v in batch]
        self.times,self.senders,self.targets=np.array(edges,dtype=np.int64).T
        assert np.all(self.senders!=self.targets)

    def evaluate(self,lab,check=False,prior=None,return_distribution=False):
        # Parents that never joined during the trace must have been present at time zero.
        times=np.full(len(self.nodes),np.iinfo(np.int64).max,dtype=np.int64)
        parents=set()
        for batch in lab["trajectory"]:
            for node,incoming in batch["joins"]:
                times[self.index[node]]=batch["t"]
                parents.update(incoming)
        unseen=np.flatnonzero(times==np.iinfo(np.int64).max)
        mandatory=sorted(parents.intersection(self.nodes[i] for i in unseen))
        assert 1<=len(mandatory)<=2 and set(mandatory)<=set(lab["sources"])
        if len(mandatory)==2:
            if return_distribution:
                return mandatory[0],mandatory[1:],np.ones(1)
            return dict(mandatory=mandatory,mass=1.,hit=True)
        a=self.index[mandatory[0]]
        candidates=unseen[unseen!=a]
        times[a]=np.iinfo(np.int64).min
        t,u,v=self.times,self.senders,self.targets
        susceptible=t<=times[v]
        active=t>times[u]
        n=len(self.nodes)
        outgoing=np.bincount(u[susceptible],minlength=n)
        blocked=np.bincount(v[active & susceptible],minlength=n)
        extra=(outgoing-blocked)[candidates]
        if check:
            # Count all exposures explicitly for each hypothetical seed, independently of the shortcut.
            base=int(np.sum(active & susceptible))
            for b,delta in zip(candidates,extra):
                exact=int(np.sum((active | (u==b)) & susceptible & (v!=b)))
                assert exact-base==delta
        log_weight=extra*np.log1p(-P)
        if prior is not None:
            log_weight+=np.log(prior[candidates])
        prob=softmax(log_weight)
        if return_distribution:
            return mandatory[0],[self.nodes[i] for i in candidates],prob
        order=np.lexsort((candidates,-prob))
        other=next(v for v in lab["sources"] if v!=mandatory[0])
        top=[self.nodes[candidates[i]] for i in order[:11]]
        cdf=np.cumsum(prob[order])
        return dict(mandatory=mandatory,partners=top,legal_partners=len(prob),
                    k50=int(np.searchsorted(cdf,.50))+1,k90=int(np.searchsorted(cdf,.90))+1,
                    mass=float(prob[order[:11]].sum()),hit=other in top)
