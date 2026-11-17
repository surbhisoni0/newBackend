import numpy as np, pickle, json
from app.research_lab import edge_lab as lab
from scipy import stats
o=pickle.load(open("main_preds.pkl","rb")); y=o["y"]["bs"]; n=len(y); half=n//2
P=dict(o["bs"]); P["Ensemble(all)"]=np.mean(list(o["bs"].values()),axis=0)
QS=[0,50,70,80,90,95]                      # keep top (100-q)% most confident
rng=np.random.default_rng(0)
res={}
print("Big/Small, confidence = |P(Big)-0.5|.  Threshold (quantile) is CHOSEN ON DEV (min 150 picks) and applied once to FINAL.\n")
print(f"{'model':14s} {'chosen q':>8s} {'DEV acc':>14s} {'FINAL acc':>20s}  p(FINAL vs 50%)")
for k,p in P.items():
    conf=np.abs(p[:,1]-0.5); pred=(p[:,1]>0.5).astype(int); ok=(pred==y)
    dev=slice(0,half); fin=slice(half,n)
    best=None
    for q in QS:
        thr=np.quantile(conf[dev],q/100); m=conf[dev]>=thr
        if m.sum()<150: continue
        a=ok[dev][m].mean()
        if best is None or a>best[1]: best=(q,a,thr,m.sum())
    q,a,thr,nd=best; mf=conf[fin]>=thr; kf=int(ok[fin][mf].sum()); nf=int(mf.sum())
    lo,hi=lab.wilson(kf,nf); pv=stats.binomtest(kf,nf,0.5).pvalue
    print(f"{k:14s} {q:>7d}% {a*100:6.2f}% (n={nd:4d}) {kf/nf*100:6.2f}% [{lo*100:4.1f}-{hi*100:4.1f}] (n={nf:4d})  p={pv:.3f}")
    res[k]=dict(q=q,dev_acc=a,final_acc=kf/nf,final_n=nf,p=pv)
# accuracy by confidence band, all test rounds, ensemble
conf=np.abs(P["Ensemble(all)"][:,1]-0.5); ok=((P["Ensemble(all)"][:,1]>0.5).astype(int)==y)
print("\nEnsemble accuracy by confidence decile (all 3586 test rounds):")
edges=np.quantile(conf,np.linspace(0,1,11)); dec=[]
for i in range(10):
    m=(conf>=edges[i])&(conf<=edges[i+1]); dec.append(ok[m].mean())
print("  ",[f"{x*100:.1f}" for x in dec])
# NULL: how good does a pure coin-flip look if we cherry-pick the best (model x confidence-slice) cell post hoc?
def best_cell(okmat):          # okmat: models x n boolean correct, confidence random
    best=0
    for r in okmat:
        c=rng.random(len(r))
        for q in (50,70,80,90,95):
            m=c>=np.quantile(c,q/100); best=max(best,r[m].mean())
    return best
sims=[best_cell(rng.random((6,n))<0.5) for _ in range(300)]
print("\nNULL (pure coin flips, 6 'models', 5 slices, best cell picked post hoc over %d rounds): median best-cell accuracy %.1f%%, 95th pct %.1f%%, max %.1f%%"%(n,np.median(sims)*100,np.percentile(sims,95)*100,max(sims)*100))
real_best=max(((P[k][:,1]>0.5).astype(int)==y)[np.abs(P[k][:,1]-0.5)>=np.quantile(np.abs(P[k][:,1]-0.5),q/100)].mean() for k in P for q in (50,70,80,90,95))
print("REAL best post-hoc cell (cherry-picked, NOT valid evidence): %.1f%%"%(real_best*100))
res["null_best_cell_p95"]=float(np.percentile(sims,95)); res["real_best_cell"]=float(real_best)
# small-slice illustration: how often does a 30-bet window hit >=70%?
w=30; hits=np.mean([ (rng.random(w)<0.5).mean()>=0.7 for _ in range(200000)])
print("A fair coin scores >=70%% over a 30-prediction window %.1f%% of the time; over 100 predictions: %.4f%%"%(hits*100, np.mean([(rng.random(100)<0.5).mean()>=0.7 for _ in range(200000)])*100))
json.dump(res,open("conf_results.json","w"),default=float,indent=1)
