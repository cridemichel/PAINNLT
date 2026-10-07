# Figura 1 del report: mappa la struttura nativa 2JPZ sui siti CG.
# Input nella cartella corrente: hybrid_2jpz_tel26_noions.gro (tutorials/tel26_unfold),
# tel26_topology.lp2.json e cg_priors.lp2.json (tutorials/tel26_unfold/ref).
# Ordine: python3 mkcg.py; python3 render.py side 30; python3 render.py top 0; python3 render.py ml 30; python3 compose.py
import json, numpy as np
MASS={'C':12.011,'N':14.007,'O':15.999,'P':30.974,'H':1.008}
top=json.load(open('tel26_topology.lp2.json'))
mp=top['mapping']['residues']
atoms=[]
L=open('hybrid_2jpz_tel26_noions.gro').read().splitlines()
n=int(L[1])
for l in L[2:2+n]:
    resid=int(l[0:5]); rn=l[5:10].strip(); an=l[10:15].strip()
    x=np.array([float(l[20:28]),float(l[28:36]),float(l[36:44])])
    atoms.append((resid,rn,an,x))
res={}
for a in atoms: res.setdefault(a[0],[]).append(a)
seq="TTAGGGTTAGGGTTAGGGTTAGGGTT"
sites=[]  # (resid, resname, sitename, type, xyz)
TYPES=top['mapping']['site_types']
for r in sorted(res):
    rn=res[r][0][1]; base='D'+seq[r-1]
    for sname, names in mp[base].items():
        sel=[a for a in res[r] if names==['*'] or a[2] in names]
        m=np.array([MASS[a[2].lstrip("0123456789")[0]] for a in sel]); X=np.array([a[3] for a in sel])
        sites.append((r,base,sname,TYPES[sname],(m[:,None]*X).sum(0)/m.sum()))
print(len(sites))
json.dump([(s[0],s[1],s[2],s[3],list(s[4])) for s in sites],open('cg_sites.json','w'))
# PDB (Angstrom)
short={'CG_DA':'A','CG_DT':'T','CG_DG_S':'S','CG_DG_B1':'B1','CG_DG_B2':'B2','CG_DG_B3':'B3','CG_DG_B4':'B4','CG_DG_B5':'B5'}
with open('cg_native.pdb','w') as f:
    for i,s in enumerate(sites):
        x=s[4]*10
        f.write("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00%6.2f          %2s\n"%(i+1,short[s[2]],s[1],s[0],x[0],x[1],x[2],s[3],'C'))
    f.write("END\n")
# convert gro to pdb (Angstrom) for AA context
with open('aa_native.pdb','w') as f:
    for i,a in enumerate(atoms):
        x=a[3]*10; rn=a[1][:3] if a[1][:2] in('DT','DA','DG') else a[1]
        rn={'DT5':'DT','DT3':'DT'}.get(a[1],a[1])[:3]
        el=a[2].lstrip("0123456789")[0]
        f.write("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"%(i+1,a[2] if len(a[2])==4 else ' '+a[2],rn,a[0],x[0],x[1],x[2],el))
    f.write("END\n")
# checks: B3-B3 distances in tetrads
idx={(s[0],s[2]):np.array(s[4]) for s in sites}
for t in [[4,12,16,22],[5,11,17,23],[6,10,18,24]]:
    d=[np.linalg.norm(idx[(a,'CG_DG_B3')]-idx[(b,'CG_DG_B3')]) for i,a in enumerate(t) for b in t[i+1:]]
    print(t, np.round(d,3))
