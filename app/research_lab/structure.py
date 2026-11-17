import numpy as np, json
from app.research_lab import edge_lab as lab
from scipy import stats
d=lab.load("data/wingo_docx_clean.csv"); h=d.number.values; n=len(h)
idx=d.period_id.astype("int64").values-d.period_id.astype("int64").values[0]   # true round index incl. gaps
out={}
# 1. digit vs (round index mod m) and vs round-of-day mod m, m=2..120  (chi-square of 10-digit dist across residues)
ps=[]
for src,name in ((idx,"index"),(d.rnd.values,"round_of_day")):
    for m in range(2,121):
        tab=np.zeros((m,10)); np.add.at(tab,(src%m,h),1)
        tab=tab[tab.sum(1)>0]; chi,p,_,_=stats.chi2_contingency(tab); ps.append((p,name,m))
ps.sort(); K=len(ps)
print("1) digit vs time-index modulus: %d tests, smallest raw p=%.4f (%s mod %d), Bonferroni p=%.3f; #raw p<0.05: %d (expected ~%.0f)"%(K,ps[0][0],ps[0][1],ps[0][2],min(1,ps[0][0]*K),sum(p<0.05 for p,_,_ in ps),0.05*K))
out["modulus_scan"]=dict(tests=K,min_p=ps[0][0],bonf=min(1,ps[0][0]*K),n_raw_sig=sum(p<0.05 for p,_,_ in ps))
# 2. determinism: fraction of repeated length-k contexts whose next digit equals the previous occurrence's next digit (null: 10%)
for k in (1,2,3,4):
    seen={}; hit=tot=0
    for t in range(k,n):
        c=tuple(h[t-k:t])
        if c in seen: tot+=1; hit+= seen[c]==h[t]
        seen[c]=h[t]
    p=stats.binomtest(hit,tot,0.1).pvalue
    print(f"2) context length {k}: repeat-same-next = {hit}/{tot} = {hit/tot*100:.2f}%  (null 10%, p={p:.3f})")
    out[f"determinism_k{k}"]=dict(hit=hit,tot=tot,p=p)
# 3. linear structure: is h[t] a linear function mod 10 of earlier digits?  (LCG-like) -> test best a,b for h[t]=(a*h[t-1]+b)%10 and 2-lag version
best=0
for a in range(10):
    for b in range(10):
        acc=np.mean((a*h[:-1]+b)%10==h[1:]); best=max(best,acc)
print("3) best h[t]=(a*h[t-1]+b) mod 10 over 100 pairs: %.2f%% (chance 10%%, 100 tries so ~11-12%% by luck)"%(best*100))
best2=0
for a in range(10):
    for b in range(10):
        for c in range(10):
            acc=np.mean((a*h[1:-1]+b*h[:-2]+c)%10==h[2:]); best2=max(best2,acc)
print("   best 2-lag linear-mod-10 over 1000 combos: %.2f%%"%(best2*100)); out["lcg"]=dict(k1=best,k2=best2)
# 4. cross-day regime: Big share by day & whether it carries into the next day
dd=d.period_id.str[:8]; print("4) Big share by day:",{k:round(float((g.number>=5).mean()),3) for k,g in d.groupby(dd)})
# 5. gap-aware: Big/Small repeat rate only between truly consecutive rounds
cons=d.gap.values[1:]==0; big=(h>=5).astype(int)
rep=(big[1:]==big[:-1])[cons]; print("5) P(same size as previous round) on %d truly consecutive pairs: %.2f%% (p=%.3f)"%(cons.sum(),rep.mean()*100,stats.binomtest(int(rep.sum()),int(cons.sum()),0.5).pvalue))
out["repeat_rate"]=float(rep.mean())
json.dump(out,open("structure.json","w"),default=float,indent=1)
