"""Loader a livello di SITO (non centroide) + contatti Hoogsteen di TEL22."""
import struct, numpy as np
NUC = 22
# (res_i, res_j, site_i, site_j) 0-based, da TEL22_TETRAD_CONTACTS; B3-B3 implicito
CONTACTS = [(1,9,2,4),(1,13,4,2),(1,21,4,2),(9,13,2,4),(9,21,2,4),(13,21,4,2),
            (2,8,4,2),(2,14,2,4),(2,20,2,4),(8,14,4,2),(8,20,4,2),(14,20,2,4),
            (3,7,2,4),(3,15,4,2),(3,19,4,2),(7,15,2,4),(7,19,2,4),(15,19,4,2)]
TETRAD_OF = [0]*6 + [1]*6 + [2]*6

def mi(d, L): return d - L*np.round(d/L)

def load_reference_sites(path):
    """-> S (T, M, 6, 3) con NaN per molecole non-guanina; L (T,3); ncopy"""
    buf = open(path,"rb").read(); off=[0]
    def take(f):
        v=struct.unpack_from(f,buf,off[0]); off[0]+=struct.calcsize(f); return v
    (T,)=take("i"); box=[]; S=[]
    for _ in range(T):
        nm,_n=take("ii"); box.append(take("3f"))
        row=np.full((nm,6,3),np.nan)
        for m in range(nm):
            _mid,ns=take("ii"); take("3f"); take("3f"); take("3f")
            blk=np.frombuffer(buf,dtype=np.int32,count=ns*4,offset=off[0]).reshape(ns,4); off[0]+=ns*16
            if ns==6: row[m]=blk[:,1:].copy().view(np.float32).astype(np.float64)
        S.append(row)
    return np.asarray(S), np.asarray(box,np.float64), nm//NUC

def load_sample_sites(path):
    z=np.load(path); sites,smol,sidx=z["sites"],z["site_molecule"],z["site_index"]
    L=np.asarray(z["box"],np.float64); T=sites.shape[0]; M=int(smol.max())+1
    S=np.full((T,M,6,3),np.nan)
    for m in np.flatnonzero(np.bincount(smol,minlength=M)==6):
        sel=np.flatnonzero(smol==m); order=np.argsort(sidx[sel])
        S[:,m]=sites[:,sel[order],:]
    return S, L, M//NUC, z["time_ps"]

def contact_distances(S, L, ncopy, frames):
    """-> d33 (n, 18), d24 (n, 18): B3-B3 e B2/B4 orientata, per (frame,copia)."""
    d33=[]; d24=[]
    for t in frames:
        Lt = L[t] if L.ndim==2 else L
        for c in range(ncopy):
            r33=[]; r24=[]
            for (ri,rj,si,sj) in CONTACTS:
                gi=S[t,c*NUC+ri]; gj=S[t,c*NUC+rj]
                if np.isnan(gi).any() or np.isnan(gj).any():
                    r33.append(np.nan); r24.append(np.nan); continue
                r33.append(np.linalg.norm(mi(gj[3]-gi[3],Lt)))
                r24.append(np.linalg.norm(mi(gj[sj]-gi[si],Lt)))
            d33.append(r33); d24.append(r24)
    return np.asarray(d33), np.asarray(d24)
