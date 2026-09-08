"""Coordinate collettive di TEL22 per il confronto termodinamico.

Due famiglie, con ruoli diversi e deliberatamente distinti.

TICA (criterio)
  Proiezione sulle componenti indipendenti a tempo ritardato delle distanze a
  coppie fra gli ancoraggi di residuo.  E' la scelta della letteratura per un
  CG di biomolecola ripiegata (Majewski et al., Nat. Commun. 2023, CGSchNet /
  TorchMD-Net: TICA sulle distanze Ca a coppie, FES 2D su griglia 80x80, piu'
  RMSD alla nativa).  Il pregio e' che le coordinate NON sono scelte a mano:
  si apprendono dal riferimento e sono i gradi di liberta' piu' lenti, cioe'
  quelli che un modello CG deve riprodurre.  Per questo la convergenza si
  giudica qui.

Q e Rg (diagnostica)
  Frazione di contatti nativi di Hoogsteen in forma liscia, e raggio di
  girazione.  Non sono la metrica della letteratura, ma si leggono
  fisicamente - TIC1 no - e separano i due modi di fallimento osservati su
  TEL22: legami persi a sagoma invariata (Q scende, Rg fermo) contro
  espansione (entrambi si muovono).  Vanno riportate, non usate come criterio.

Unita': nm, come tutto il resto di TEL22.
"""
from __future__ import annotations

import struct
import numpy as np

NUC = 22          # residui per copia
MAX_SITES = 6     # guanina: S, B1..B5; A e T hanno un solo sito

# Riuso il grafo di contatti validato invece di ridefinirlo qui.
from _hb_common import CONTACTS, TETRAD_OF, mi

# I 12 legami H di Hoogsteen del ciclo, come nello script 40.
CYCLE_1B = {0: [(2, 10), (10, 22), (22, 14), (14, 2)],
            1: [(3, 9), (9, 21), (21, 15), (15, 3)],
            2: [(4, 8), (8, 20), (20, 16), (16, 4)]}
HB_IDX = [k for k, (ri, rj, _, _) in enumerate(CONTACTS)
          if any({ri + 1, rj + 1} == set(p) for p in CYCLE_1B[TETRAD_OF[k]])]
assert len(HB_IDX) == 12


# ── caricamento: tutti i siti, non solo le guanine ───────────────────────────
# I loader di _hb_common scartano le molecole con ns != 6, perche' servono solo
# i contatti fra guanine.  Rg e gli ancoraggi richiedono anche A e T, quindi qui
# si conserva tutto in un array riempito con NaN.

def load_reference(path):
    """-> S (T, M, 6, 3) con NaN nei posti vuoti, L (T, 3), ncopy."""
    buf = open(path, "rb").read()
    off = [0]

    def take(fmt):
        v = struct.unpack_from(fmt, buf, off[0])
        off[0] += struct.calcsize(fmt)
        return v

    (T,) = take("i")
    box, frames = [], []
    for _ in range(T):
        nm, _n = take("ii")
        box.append(take("3f"))
        row = np.full((nm, MAX_SITES, 3), np.nan)
        for m in range(nm):
            _mid, ns = take("ii")
            take("3f"); take("3f"); take("3f")
            blk = np.frombuffer(buf, dtype=np.int32, count=ns * 4,
                                offset=off[0]).reshape(ns, 4)
            off[0] += ns * 16
            row[m, :ns] = blk[:, 1:].copy().view(np.float32).astype(np.float64)
        frames.append(row)
    return np.asarray(frames), np.asarray(box, np.float64), nm // NUC


def load_samples(path):
    """-> S (T, M, 6, 3), L, ncopy, t (ps)."""
    z = np.load(path)
    sites, smol, sidx = z["sites"], z["site_molecule"], z["site_index"]
    L = np.asarray(z["box"], np.float64)
    T, M = sites.shape[0], int(smol.max()) + 1
    S = np.full((T, M, MAX_SITES, 3), np.nan)
    for m in range(M):
        sel = np.flatnonzero(smol == m)
        if sel.size == 0:
            continue
        order = np.argsort(sidx[sel])
        S[:, m, :sel.size] = sites[:, sel[order], :]
    return S, L, M // NUC, z["time_ps"]


def _box_at(L, t):
    return L[t] if L.ndim == 2 else L


# ── feature per TICA: distanze a coppie fra ancoraggi di residuo ─────────────
# L'ancoraggio e' il primo sito di ogni residuo (S per la guanina), quindi la
# definizione e' uniforme fra guanine e non-guanine.  22 residui -> 231 coppie,
# la stessa granularita' delle distanze Ca a coppie del lavoro sulle proteine.

def anchor_distances(S, L, ncopy):
    """-> (T*ncopy, 231): distanze fra ancoraggi, per (frame, copia)."""
    T = S.shape[0]
    iu, ju = np.triu_indices(NUC, k=1)
    out = np.empty((T * ncopy, iu.size))
    r = 0
    for t in range(T):
        Lt = _box_at(L, t)
        for c in range(ncopy):
            a = S[t, c * NUC:(c + 1) * NUC, 0, :]          # (22, 3)
            d = mi(a[iu] - a[ju], Lt)
            out[r] = np.linalg.norm(d, axis=1)
            r += 1
    return out


# ── diagnostica interpretabile: Q liscio e Rg ───────────────────────────────

def native_hb_distances(S, L, ncopy, stride=5):
    """Distanza nativa mediana dei 12 legami H, dal riferimento."""
    from _hb_common import contact_distances
    d33, _ = contact_distances(S, L, ncopy, range(0, S.shape[0], stride))
    return np.nanmedian(d33[:, HB_IDX], axis=0)          # (12,)


def smooth_Q(S, L, ncopy, native, beta=25.0, lam=1.2):
    """Frazione liscia di contatti nativi intatti, per (frame, copia).

    Q = (1/N) sum_k 1 / (1 + exp(beta (r_k - lam r_k^nat)))

    lam = 1.2 riproduce la soglia dura di 0.60 nm dello script 40 rispetto alla
    nativa mediana di ~0.499 nm, quindi Q e' la versione continua dello stesso
    criterio.  beta = 25 nm^-1 da' una transizione di ~0.08 nm: abbastanza
    ripida da distinguere intatto da rotto, abbastanza morbida da non
    reintrodurre la discretizzazione che rende un Q a soglia inutilizzabile
    come asse di un istogramma (12 legami -> soli 13 valori possibili).
    """
    from _hb_common import contact_distances
    T = S.shape[0]
    d33, _ = contact_distances(S, L, ncopy, range(T))
    r = d33.reshape(T * ncopy, -1)[:, HB_IDX]
    q = 1.0 / (1.0 + np.exp(beta * (r - lam * native[None, :])))
    return np.nanmean(q, axis=1)


def radius_of_gyration(S, L, ncopy):
    """Rg per (frame, copia), su tutti i siti fisici della copia.

    Le posizioni sono srotolate rispetto al primo sito della copia prima di
    calcolare il centro di massa: senza questo, una copia a cavallo del bordo
    periodico darebbe un Rg enorme e spurio.
    """
    T = S.shape[0]
    out = np.empty(T * ncopy)
    r = 0
    for t in range(T):
        Lt = _box_at(L, t)
        for c in range(ncopy):
            blk = S[t, c * NUC:(c + 1) * NUC].reshape(-1, 3)
            p = blk[np.isfinite(blk).all(axis=1)]
            p = p[0] + mi(p - p[0], Lt)                   # srotolamento
            out[r] = np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, axis=1)))
            r += 1
    return out


# ── RMSD alla nativa, con sovrapposizione di Kabsch ─────────────────────────

def kabsch_rmsd(P, Q):
    """RMSD fra due insiemi (N,3) dopo rotazione+traslazione ottimale."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    Pr = Pc @ (V @ D @ Wt)
    return float(np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))))


def rmsd_to_native(S, L, ncopy, native_anchors):
    """RMSD degli ancoraggi alla struttura nativa, per (frame, copia)."""
    T = S.shape[0]
    out = np.empty(T * ncopy)
    r = 0
    for t in range(T):
        Lt = _box_at(L, t)
        for c in range(ncopy):
            a = S[t, c * NUC:(c + 1) * NUC, 0, :]
            a = a[0] + mi(a - a[0], Lt)
            out[r] = kabsch_rmsd(a, native_anchors)
            r += 1
    return out


def mean_native_anchors(S, L, ncopy, stride=5):
    """Struttura nativa di riferimento: media degli ancoraggi sul riferimento,
    ogni copia srotolata e sovrapposta alla prima copia del primo frame."""
    L0 = _box_at(L, 0)
    ref = S[0, 0:NUC, 0, :]
    ref = ref[0] + mi(ref - ref[0], L0)
    acc = np.zeros_like(ref)
    n = 0
    for t in range(0, S.shape[0], stride):
        Lt = _box_at(L, t)
        for c in range(ncopy):
            a = S[t, c * NUC:(c + 1) * NUC, 0, :]
            a = a[0] + mi(a - a[0], Lt)
            Pc, Qc = a - a.mean(0), ref - ref.mean(0)
            V, _, Wt = np.linalg.svd(Pc.T @ Qc)
            d = np.sign(np.linalg.det(V @ Wt))
            acc += Pc @ (V @ np.diag([1.0, 1.0, d]) @ Wt)
            n += 1
    return acc / n


def load_site_types(path):
    """-> (M, 6) int con -1 nei posti vuoti: il tipo CG di ogni sito.

    Il tipo sta nella colonna 0 dei blocchi del dataset binario; i loader di
    _hb_common la scartano perche' a loro servono solo le posizioni.  Mapping
    TEL22: A -> 0, T -> 1, G -> 2..7 nell'ordine S, B1, B2, B3, B4, B5.
    Quindi B3, il sito dei legami H di Hoogsteen, e' il tipo 5.
    """
    buf = open(path, "rb").read()
    off = [0]

    def take(fmt):
        v = struct.unpack_from(fmt, buf, off[0])
        off[0] += struct.calcsize(fmt)
        return v

    (T,) = take("i")
    nm, _n = take("ii")
    take("3f")
    types = np.full((nm, MAX_SITES), -1, dtype=np.int64)
    for m in range(nm):
        _mid, ns = take("ii")
        take("3f"); take("3f"); take("3f")
        blk = np.frombuffer(buf, dtype=np.int32, count=ns * 4,
                            offset=off[0]).reshape(ns, 4)
        off[0] += ns * 16
        types[m, :ns] = blk[:, 0]
    return types


SITE_NAME = {0: "A", 1: "T", 2: "S", 3: "B1", 4: "B2", 5: "B3", 6: "B4", 7: "B5"}
