import numpy as np, json, time
from app.research_lab import edge_lab as lab
from scipy import stats
d=lab.load("data/wingo_docx_clean.csv"); n=len(d); rng=np.random.default_rng(42)
START,REFIT=1500,500; kinds=["LogReg","LightGBM","RandForest"]
def synth(kind,p):
    # digits uniform, but Big/Small outcome shaped by the planted rule
    h=np.zeros(n,int); big=np.zeros(n,int); big[0]=rng.integers(2)
    for t in range(n):
        if kind=="shuffle": b=rng.integers(2)
        elif kind=="repeat": b=big[t-1] if (t and rng.random()<p) else rng.integers(2)
        elif kind=="streak_reversal":              # only after a run of >=3 equal sizes: reverse with prob p (applies to ~1/4 of rounds)
            if t>=3 and big[t-1]==big[t-2]==big[t-3]: b=1-big[t-1] if rng.random()<p else big[t-1]
            else: b=rng.integers(2)
        big[t]=b; h[t]=rng.integers(5,10) if b else rng.integers(0,5)
    return h,big
def run(kind,p):
    h,big=synth(kind,p); X,_=lab.features(h,d.rnd.values,d.gap.values)
    P=lab.walk_forward(X,big,2,kinds,START,REFIT); y=big[START:]; E=np.mean(list(P.values()),axis=0); P["Ensemble"]=E
    out={}
    for k,pp in P.items():
        acc=((pp[:,1]>0.5)==y).mean(); out[k]=acc
    # confidence-filtered ensemble: top 25% by confidence
    c=np.abs(E[:,1]-0.5); m=c>=np.quantile(c,0.75); fa=((E[m,1]>0.5)==y[m]).mean(); ok=((E[:,1]>0.5)==y)
    pv=stats.binomtest(int(ok.sum()),len(y),0.5).pvalue; pf=stats.binomtest(int(((E[m,1]>0.5)==y[m]).sum()),int(m.sum()),0.5).pvalue
    return out,fa,pv,pf,len(y)
cases=[("shuffle",0.5),("repeat",0.10),("repeat",0.20),("repeat",0.40),("streak_reversal",0.65),("streak_reversal",0.80)]
res=[]
for kind,p in cases:
    t0=time.time(); out,fa,pv,pf,ny=run(kind,p)
    # true best-possible accuracy for reference
    ceil={"shuffle":0.5,"repeat":0.5+p/2,"streak_reversal":None}[kind]
    if kind=="streak_reversal": ceil=0.75*0.5+0.25*p
    print(f"{kind:16s} p={p:.2f} theoretical ceiling {ceil*100:4.1f}% | "+" ".join(f"{k}:{v*100:4.1f}%" for k,v in out.items())+f" | ensemble top25% conf: {fa*100:4.1f}% (p={pf:.3g}) | all-rounds p={pv:.3g}  [{time.time()-t0:.0f}s]",flush=True)
    res.append(dict(kind=kind,p=p,ceiling=ceil,acc=out,top25=fa,p_all=pv,p_top25=pf))
json.dump(res,open("controls.json","w"),default=float,indent=1)
