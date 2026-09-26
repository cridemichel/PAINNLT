#!/usr/bin/env python3
"""g(r) di riferimento contro g(r) del modello ML: il test strutturale primario.

PERCHE' E' LA METRICA PIU' FONDAMENTALE
  Per un modello CG di coppia il teorema di Henderson lega univocamente g(r) al
  potenziale: due potenziali di coppia che danno la stessa g(r) sono lo stesso
  potenziale.  E' anche l'osservabile su cui itera l'IBI.  Quindi la
  sovrapposizione fra la g(r) del riferimento all-atom mappato su CG e quella
  della dinamica ESPResSo con potenziale ML e' la condizione strutturale
  necessaria: se non si sovrappongono, il modello e' sbagliato, senza appello.

  E' NECESSARIA MA NON SUFFICIENTE, e va detto.  Henderson vale per un
  potenziale di coppia puro; qui l'Hamiltoniana ha prior bonded, WCA, Morse e
  un residuo ML a molti corpi.  Due modelli diversi possono dare g(r) quasi
  identiche e paesaggi di energia libera diversi - e' esattamente per questo
  che la letteratura sulle proteine (Majewski et al. 2023) valuta anche la FES
  su TICA.  Le due misure sono complementari: g(r) per la struttura di coppia,
  FES su TICA (script 43) per i gradi di liberta' collettivi lenti.

DUE CONTRIBUTI DA NON MESCOLARE
  intra  distanze fra siti della STESSA copia.  Non e' una g(r) in senso
         liquido: e' dominata dalla topologia e dal ripiegamento, e non si
         normalizza per la densita'.  Riportata come distribuzione P(r).
         E' qui che vive la struttura del quadruplex.
  inter  distanze fra siti di copie DIVERSE.  Questa e' la g(r) propria, con
         normalizzazione 4 pi r^2 rho.  Con 10 copie in scatola la statistica
         e' povera e gli effetti di taglia finita non sono trascurabili: va
         letta come controllo di aggregazione, non come misura fine.

IL RIFERIMENTO
  E' tel22_dataset.bin, cioe' la traiettoria GROMACS GIA' MAPPATA sui siti CG
  con lo stesso mapping usato per costruire il dataset di training.  Confrontare
  con l'atomistico grezzo non avrebbe senso: i siti CG sono definiti da quel
  mapping.

USO
  python3 44_tel22_rdf.py <dataset.bin> A=<run>/samples.npz [B=...] \
      [--rmax 2.0] [--bins 200] [--types] [--json out.json] [--nuc 26]

NON E' SOLO TEL22
  Di specifico al TEL22 c'e' solo il numero di residui per copia, che serve a
  spezzare la lista delle molecole in copie: --nuc lo cambia.  I canali di
  tipo (S, B1..B5) valgono per qualunque sistema, perche' sono chimica del
  nucleotide e non della piega.  Per il TEL26 ibrido:

      python3 44_tel22_rdf.py tel26_dataset.bin run=samples.npz --nuc 26

  Restano legate al 143D le coordinate collettive dello script 43 -- Q sul
  ciclo delle tetradi, RMSD -- che passano da _hb_common e richiedono il
  registro delle tetradi del sistema, non solo il suo numero di residui.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _tel22_cv as cv
from _tel22_cv import (SITE_NAME, load_reference, load_samples,
                       load_site_types, _box_at)
# NUC si legge come cv.NUC e non per nome: --nuc la cambia a runtime, e un
# "from ... import NUC" congelerebbe il valore all'importazione.
from _hb_common import mi


# --no-same-residue: esclude dall'INTRA le coppie di siti dello stesso
# residuo.  Nel CG la guanina e' un corpo rigido: le sue 15 distanze interne
# sono fisse, nel riferimento fluttuano.  Quelle coppie misurano il mapping,
# non il campo di forza, e nessun prior puo' avvicinarle.
SKIP_SAME_RES = False


def _mol_of_site(blk, ok):
    return np.repeat(np.arange(blk.shape[0]), blk.shape[1])[ok.ravel()]


def load_run(path, skip_ps=0.0, stride=1):
    """Traiettoria di un modello, senza il transiente iniziale e sottocampionata.

    skip_ps: scarta i frame con t < skip_ps.  Dopo aver acceso il potenziale ML
             il sistema rilassa nel paesaggio del modello per decine di ps
             (sul TEL26 D=64, E_ML si stabilizza dopo ~60 ps): mediare anche
             quel tratto mescola l'ensemble del modello con la struttura di
             partenza, che e' quella di riferimento, e gonfia l'accordo.
    stride:  un frame ogni `stride`.  Con log_interval = 20 passi i frame sono
             a 0.02 ps, fortemente correlati: uno ogni 10 non perde statistica
             e riduce il costo dell'analisi di dieci volte.
    """
    S, L, nc, t = load_samples(path)
    t = np.asarray(t, dtype=float)
    keep = np.flatnonzero(t >= float(skip_ps))[::max(1, int(stride))]
    if keep.size == 0:
        raise SystemExit(f"[ERROR] {path}: nessun frame con t >= {skip_ps} ps "
                         f"(la traiettoria arriva a {t.max():.1f} ps)")
    L = np.asarray(L)
    if L.ndim == 2 and L.shape[0] == S.shape[0]:
        L = L[keep]
    print(f"  {path}: {keep.size} frame da t = {t[keep[0]]:.1f} a {t[keep[-1]]:.1f} ps")
    return S[keep], L, nc, t[keep]


def _copy_of_site(nmol_total: int, ncopy: int) -> np.ndarray:
    """Indice di copia per ciascuna molecola."""
    return np.repeat(np.arange(ncopy), cv.NUC)


# Canali di tipo: la g(r) totale mescola S-S, B3-B3, zucchero-base e
# guanina-adenina, cosi' errori in canali diversi possono cancellarsi.  Il
# canale che porta l'informazione sul quadruplex e' B3-B3 (tipo 5), dove stanno
# i legami H di Hoogsteen; B5-B5 (tipo 7) e' l'impilamento fra tetradi.
CHANNELS = [("tutti i siti", None),
            ("B3-B3 (legami H)", (5, 5)),
            ("B5-B5 (stacking)", (7, 7)),
            ("S-S (backbone)", (2, 2))]


def accumulate_channels(S, L, ncopy, types, rmax, nbins, stride=1):
    """Come accumulate(), ma per canale di tipo, in un solo passaggio.

    -> dict canale -> (h_intra, h_inter), piu' (nframes, volume, edges, n_inter)
    """
    T, M = S.shape[0], S.shape[1]
    cop = np.repeat(np.arange(ncopy), cv.NUC)
    edges = np.linspace(0.0, rmax, nbins + 1)
    acc = {name: [np.zeros(nbins), np.zeros(nbins)] for name, _ in CHANNELS}
    n_inter = {name: 0 for name, _ in CHANNELS}
    vol = 0.0
    nf = 0
    for t in range(0, T, stride):
        Lt = _box_at(L, t)
        blk = S[t]
        ok = np.isfinite(blk).all(axis=2)
        pos = blk[ok]
        who = np.repeat(cop, blk.shape[1])[ok.ravel()]
        typ = types[ok]
        mol = _mol_of_site(blk, ok)
        n = pos.shape[0]
        if n < 2:
            continue
        for i in range(0, n, 512):
            d = mi(pos[i:i + 512, None, :] - pos[None, :, :], Lt)
            r = np.linalg.norm(d, axis=2)
            gi = np.arange(i, min(i + 512, n))
            upper = gi[:, None] < np.arange(n)[None, :]
            same = who[i:i + 512, None] == who[None, :]
            if SKIP_SAME_RES:
                same = same & (mol[i:i + 512, None] != mol[None, :])
            ta, tb = typ[i:i + 512, None], typ[None, :]
            for name, pair in CHANNELS:
                if pair is None:
                    sel = upper
                else:
                    a, b = pair
                    sel = upper & (((ta == a) & (tb == b)) | ((ta == b) & (tb == a)))
                si, so = sel & same, sel & ~same
                acc[name][0] += np.histogram(r[si], bins=edges)[0]
                acc[name][1] += np.histogram(r[so], bins=edges)[0]
                n_inter[name] += int(so.sum())
        vol += float(np.prod(Lt))
        nf += 1
    return acc, n_inter, nf, vol / max(nf, 1), edges


# Nomi dei tipi del mapping G-quadruplex del framework (DA, DT un sito; DG
# sei).  Un tipo fuori tabella si stampa col suo numero.
TYPE_NAMES = {0: "DA", 1: "DT", 2: "S", 3: "B1", 4: "B2", 5: "B3", 6: "B4", 7: "B5"}


def accumulate_type_pairs_intra(S, L, ncopy, types, rmax, nbins, stride=1):
    """Istogrammi INTRA-copia per OGNI coppia di tipi, in un passaggio.

    -> (hist[ntype, ntype, nbins] triangolare superiore, edges, ntype)
    Serve a capire dove sta lo scarto residuo della P(r) totale: i quattro
    canali di --types ne coprono solo una parte.
    """
    T = S.shape[0]
    cop = np.repeat(np.arange(ncopy), cv.NUC)
    edges = np.linspace(0.0, rmax, nbins + 1)
    dr = edges[1] - edges[0]
    ntype = int(np.max(types)) + 1
    hist = np.zeros(ntype * ntype * nbins, dtype=np.int64)
    for t in range(0, T, stride):
        Lt = _box_at(L, t)
        blk = S[t]
        ok = np.isfinite(blk).all(axis=2)
        pos = blk[ok]
        who = np.repeat(cop, blk.shape[1])[ok.ravel()]
        typ = types[ok].astype(np.int64)
        mol = _mol_of_site(blk, ok)
        n = pos.shape[0]
        for i in range(0, n, 512):
            d = mi(pos[i:i + 512, None, :] - pos[None, :, :], Lt)
            r = np.linalg.norm(d, axis=2)
            gi = np.arange(i, min(i + 512, n))
            sel = (gi[:, None] < np.arange(n)[None, :]) & (who[i:i + 512, None] == who[None, :]) \
                & (r < rmax)
            if SKIP_SAME_RES:
                sel &= mol[i:i + 512, None] != mol[None, :]
            ta = np.broadcast_to(typ[i:i + 512, None], r.shape)[sel]
            tb = np.broadcast_to(typ[None, :], r.shape)[sel]
            lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
            b = np.minimum((r[sel] / dr).astype(np.int64), nbins - 1)
            hist += np.bincount((lo * ntype + hi) * nbins + b, minlength=hist.size)
    return hist.reshape(ntype, ntype, nbins), edges, ntype


def plot_channels(path, edges, ref, runs, kind="intra", title="", scores=None):
    """Curve per canale, riferimento in continuo e modelli tratteggiati.

    ref/runs: dict canale -> curva.  runs e' [(etichetta, dict)].
    scores:   opzionale, etichetta -> canale -> sovrapposizione; se c'e',
              ogni pannello riporta in legenda il valore della sua curva,
              cosi' il grafico si legge senza la tabella accanto.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = 0.5 * (edges[:-1] + edges[1:])
    names = [n for n, _ in CHANNELS]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    ylab = "P(r)   (nm$^{-1}$)" if kind == "intra" else "g(r)"
    for ax, name in zip(axes.ravel(), names):
        ax.plot(r, ref[name], color="k", lw=2.2, label="all-atom (mappato su CG)")
        for lab, cur in runs:
            shown = lab
            if scores and lab in scores and name in scores[lab]:
                shown = f"{lab}  (sovr. {scores[lab][name]:.3f})"
            ax.plot(r, cur[name], lw=1.6, ls="--", label=shown)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("r  (nm)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, r[-1])
        ax.grid(alpha=0.25, lw=0.5)
        if kind == "inter":
            ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    fig.suptitle(f"{title}  -  "
                 + ("distanze intra-copia  P(r):  struttura del ripiegamento"
                    if kind == "intra"
                    else "g(r) inter-copia:  struttura di soluzione"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def accumulate(S, L, ncopy, rmax, nbins, stride=1):
    """-> (h_intra, h_inter, npairs_intra, npairs_inter, nframes, volume medio)

    Un solo passaggio sui frame, contando le distanze in shell radiali e
    separando le coppie intra-copia da quelle inter-copia.
    """
    T, M = S.shape[0], S.shape[1]
    cop = _copy_of_site(M, ncopy)
    edges = np.linspace(0.0, rmax, nbins + 1)
    h_intra = np.zeros(nbins)
    h_inter = np.zeros(nbins)
    n_intra = n_inter = 0
    vol = 0.0
    nf = 0
    for t in range(0, T, stride):
        Lt = _box_at(L, t)
        blk = S[t]                                   # (M, 6, 3)
        ok = np.isfinite(blk).all(axis=2)            # (M, 6)
        pos = blk[ok]                                # (Ns, 3)
        who = np.repeat(cop, blk.shape[1])[ok.ravel()]
        mol = _mol_of_site(blk, ok)
        n = pos.shape[0]
        if n < 2:
            continue
        # Distanze a coppie con immagine minima, a blocchi per non allocare
        # una matrice n x n x 3 intera.
        for i in range(0, n, 512):
            d = mi(pos[i:i + 512, None, :] - pos[None, :, :], Lt)
            r = np.linalg.norm(d, axis=2)
            same = who[i:i + 512, None] == who[None, :]
            # solo la meta' superiore, per non contare due volte
            gi = np.arange(i, min(i + 512, n))
            upper = gi[:, None] < np.arange(n)[None, :]
            m_intra = upper & same
            m_inter = upper & ~same
            if SKIP_SAME_RES:
                m_intra &= mol[i:i + 512, None] != mol[None, :]
            h_intra += np.histogram(r[m_intra], bins=edges)[0]
            h_inter += np.histogram(r[m_inter], bins=edges)[0]
            n_intra += int(m_intra.sum())
            n_inter += int(m_inter.sum())
        vol += float(np.prod(Lt))
        nf += 1
    return h_intra, h_inter, n_intra, n_inter, nf, (vol / max(nf, 1)), edges


def normalize(h_intra, h_inter, n_inter, nframes, volume, edges, nsites):
    """P(r) per l'intra, g(r) con 4 pi r^2 rho per l'inter."""
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    # intra: distribuzione normalizzata, nessuna densita' in gioco
    p_intra = h_intra / max(h_intra.sum(), 1.0) / dr
    # inter: numero atteso di coppie in shell per un gas ideale alla stessa densita'
    shell = 4.0 * np.pi * r ** 2 * dr
    # coppie inter possibili per frame
    pairs_per_frame = n_inter / max(nframes, 1)
    rho_pairs = pairs_per_frame / volume if volume > 0 else 0.0
    ideal = rho_pairs * shell * nframes
    with np.errstate(divide="ignore", invalid="ignore"):
        g_inter = np.where(ideal > 0, h_inter / ideal, np.nan)
    return r, p_intra, g_inter


def overlap_metrics(ref, mod, r):
    """Sovrapposizione fra due curve sulla stessa griglia.

    integral_overlap  integrale del minimo fra le due curve normalizzate: 1 =
                      identiche, 0 = disgiunte.  E' la misura piu' diretta di
                      "si sovrappongono".
    l1                distanza L1 relativa.
    rmse              scarto quadratico medio sulla regione dove il
                      riferimento ha supporto.
    first_peak_shift  spostamento del primo massimo, in nm: dice se la
                      struttura e' traslata anche quando l'ampiezza torna.
    """
    m = np.isfinite(ref) & np.isfinite(mod)
    if not np.any(m):
        return {k: float("nan") for k in
                ("integral_overlap", "l1_relative", "rmse", "first_peak_shift_nm")}
    a, b, rr = ref[m], mod[m], r[m]
    dr = float(np.mean(np.diff(rr))) if rr.size > 1 else 1.0
    na = a / max(a.sum() * dr, 1e-300)
    nb = b / max(b.sum() * dr, 1e-300)
    ov = float(np.sum(np.minimum(na, nb)) * dr)
    l1 = float(np.sum(np.abs(a - b)) / max(np.sum(np.abs(a)), 1e-300))
    sup = a > 0.05 * a.max()
    rmse = float(np.sqrt(np.mean((a[sup] - b[sup]) ** 2))) if np.any(sup) else float("nan")
    # ATTENZIONE: e' l'argmax GLOBALE, non il primo picco, malgrado il nome
    # della chiave.  Su una P(r) intra larga e multimodale -- tutte le coppie
    # di siti dentro una copia -- basta un piccolo spostamento di peso perche'
    # salti da un modo all'altro e riporti frazioni di nanometro senza che la
    # struttura sia cambiata di molto.  Gli indicatori robusti sono la
    # sovrapposizione e la L1.
    shift = float(rr[np.argmax(b)] - rr[np.argmax(a)])
    return {"integral_overlap": ov, "l1_relative": l1, "rmse": rmse,
            "first_peak_shift_nm": shift}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("runs", nargs="+", help="etichetta=percorso/samples.npz")
    ap.add_argument("--rmax", type=float, default=2.0, help="nm (default 2.0)")
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--stride", type=int, default=2,
                    help="stride sui frame del riferimento (default 2)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--all-pairs", dest="all_pairs", action="store_true",
                    help="tabella INTRA per ogni coppia di tipi: peso nel totale, "
                         "sovrapposizione per corsa e contributo allo scarto")
    ap.add_argument("--types", action="store_true",
                    help="scomponi per canale di tipo (tutti, B3-B3 legami H, "
                         "B5-B5 stacking, S-S backbone).  La curva totale li "
                         "mescola, quindi errori di segno opposto in canali "
                         "diversi possono cancellarsi e farla sembrare giusta")
    ap.add_argument("--plot", default=None,
                    help="prefisso dei PNG per canale (richiede --types)")
    ap.add_argument("--skip-ps", dest="skip_ps", type=float, default=0.0,
                    help="scarta i primi N ps di ogni traiettoria di modello "
                         "(transiente dopo l'accensione del ML); il riferimento "
                         "non e' toccato")
    ap.add_argument("--run-stride", dest="run_stride", type=int, default=1,
                    help="un frame ogni N nelle traiettorie di modello (default 1)")
    ap.add_argument("--title", default=None,
                    help="titolo dei grafici; default: dal nome del dataset "
                         "(tel26_dataset.bin -> TEL26)")
    ap.add_argument("--nuc", type=int, default=None,
                    help="residui per copia (default 22, cioe' TEL22; 26 per il "
                         "TEL26).  Serve solo a spezzare le molecole in copie, "
                         "quindi la g(r) vale per qualunque G-quadruplex")
    ap.add_argument("--no-same-residue", dest="no_same_res", action="store_true",
                    help="escludi dall'INTRA le coppie di siti dello stesso residuo "
                         "(interne al corpo rigido della guanina: fisse nel CG)")
    ap.add_argument("--ref-range", dest="ref_range", default=None, metavar="A:B",
                    help="usa come riferimento solo la frazione [A, B) dei frame "
                         "del dataset (es. 0:0.5).  Con una corsa '@ref:0.5:1' da' il "
                         "tetto di sovrapposizione: meta' del riferimento contro "
                         "l'altra meta', cioe' il rumore statistico del riferimento")
    args = ap.parse_args()
    global SKIP_SAME_RES
    SKIP_SAME_RES = bool(args.no_same_res)
    if SKIP_SAME_RES:
        print("[INFO] INTRA senza le coppie dello stesso residuo (--no-same-residue)")
    if args.nuc:
        cv.set_nuc(args.nuc)
    print(f"[INFO] {cv.NUC} residui per copia")

    runs = {}
    for spec in args.runs:
        if "=" not in spec:
            ap.error(f"atteso etichetta=percorso, ricevuto {spec!r}")
        k, v = spec.split("=", 1)
        runs[k] = v

    print(f"  riferimento: {args.dataset}")
    S_full, L_full, ncr = load_reference(args.dataset)

    def frac_slice(spec, what):
        try:
            lo, hi = (float(x) for x in spec.split(":"))
        except ValueError:
            raise SystemExit(f"[ERROR] {what}: atteso A:B con frazioni, ricevuto {spec!r}")
        if not 0.0 <= lo < hi <= 1.0:
            raise SystemExit(f"[ERROR] {what}: serve 0 <= A < B <= 1, ricevuto {spec!r}")
        T = S_full.shape[0]
        return int(round(lo * T)), int(round(hi * T))

    if args.ref_range:
        i0, i1 = frac_slice(args.ref_range, "--ref-range")
        Sr, Lr = S_full[i0:i1], L_full[i0:i1]
        print(f"  riferimento ristretto ai frame {i0}-{i1} (--ref-range {args.ref_range})")
    else:
        Sr, Lr = S_full, L_full

    def get_run(path):
        """Corsa di un modello, o una fetta del riferimento con '@ref:A:B'."""
        if path.startswith("@ref:"):
            i0, i1 = frac_slice(path[5:], path)
            st = max(1, int(args.stride))
            print(f"  {path}: frame {i0}-{i1} del riferimento, stride {st}")
            return S_full[i0:i1:st], L_full[i0:i1:st], ncr, None
        if "#" in path:
            # finestra temporale di una corsa: percorso#T0:T1 (ps).  Due finestre
            # della stessa corsa danno la dispersione statistica della corsa CG.
            path, win = path.split("#", 1)
            try:
                t0, t1 = (float(x) for x in win.split(":"))
            except ValueError:
                raise SystemExit(f"[ERROR] finestra {win!r}: attesa T0:T1 in ps")
            S, L, nc, t = load_run(path, max(t0, args.skip_ps), args.run_stride)
            keep = np.flatnonzero(t < t1)
            if keep.size == 0:
                raise SystemExit(f"[ERROR] {path}: nessun frame in [{t0}, {t1}) ps")
            L = np.asarray(L)
            if L.ndim == 2 and L.shape[0] == S.shape[0]:
                L = L[keep]
            print(f"    finestra {t0:g}-{t1:g} ps: {keep.size} frame")
            return S[keep], L, nc, t[keep]
        return load_run(path, args.skip_ps, args.run_stride)

    hi, he, ni, ne, nf, vol, edges = accumulate(Sr, Lr, ncr, args.rmax,
                                                args.bins, args.stride)
    nsites = int(np.isfinite(Sr[0]).all(axis=2).sum())
    r, p_ref, g_ref = normalize(hi, he, ne, nf, vol, edges, nsites)
    print(f"  {Sr.shape[0]} frame x {ncr} copie, {nsites} siti/frame, "
          f"{nf} frame usati (stride {args.stride})")
    print(f"  coppie: {ni} intra, {ne} inter | volume medio {vol:.1f} nm^3")

    report = {"reference": args.dataset, "rmax_nm": args.rmax,
              "bins": args.bins, "runs": []}
    rows = []
    for label, path in runs.items():
        S, L, nc, _t = get_run(path)
        hi2, he2, ni2, ne2, nf2, vol2, _ = accumulate(S, L, nc, args.rmax, args.bins, 1)
        _, p_mod, g_mod = normalize(hi2, he2, ne2, nf2, vol2, edges, nsites)
        e = {"run": label, "path": path,
             "intra": overlap_metrics(p_ref, p_mod, r),
             "inter": overlap_metrics(g_ref, g_mod, r)}
        report["runs"].append(e)
        rows.append(e)

    print(f"\n  INTRA-copia  P(r): struttura del ripiegamento  "
          f"[sovrapp. 1.000 = identica]")
    print(f"  {'run':<16}{'sovrapp.':>10}{'L1 rel.':>10}{'RMSE':>10}"
          f"{'shift picco':>14}")
    print("  " + "-" * 60)
    for e in sorted(rows, key=lambda x: -x["intra"]["integral_overlap"]):
        m = e["intra"]
        print(f"  {e['run']:<16}{m['integral_overlap']:>10.4f}"
              f"{m['l1_relative']:>10.4f}{m['rmse']:>10.4f}"
              f"{m['first_peak_shift_nm']:>13.3f} nm")

    print(f"\n  INTER-copia  g(r): struttura di soluzione  "
          f"[statistica povera con {ncr} copie, controllo di aggregazione]")
    print(f"  {'run':<16}{'sovrapp.':>10}{'L1 rel.':>10}{'RMSE':>10}"
          f"{'shift picco':>14}")
    print("  " + "-" * 60)
    for e in sorted(rows, key=lambda x: -x["inter"]["integral_overlap"]):
        m = e["inter"]
        print(f"  {e['run']:<16}{m['integral_overlap']:>10.4f}"
              f"{m['l1_relative']:>10.4f}{m['rmse']:>10.4f}"
              f"{m['first_peak_shift_nm']:>13.3f} nm")

    if args.types:
        # I tipi si leggono dal dataset di riferimento: le traiettorie prodotte
        # portano gli stessi siti nello stesso ordine, per costruzione.
        types = load_site_types(args.dataset)
        acc_r, nint_r, nf_r, vol_r, _ = accumulate_channels(
            Sr, Lr, ncr, types, args.rmax, args.bins, args.stride)
        ref_curves = {"intra": {}, "inter": {}}
        for name, _ in CHANNELS:
            _, p_c, g_c = normalize(acc_r[name][0], acc_r[name][1],
                                    nint_r[name], nf_r, vol_r, edges, nsites)
            ref_curves["intra"][name] = p_c
            ref_curves["inter"][name] = g_c

        run_curves = {"intra": [], "inter": []}
        per_channel = {}
        for label, path in runs.items():
            S, L, nc, _t = get_run(path)
            acc_m, nint_m, nf_m, vol_m, _ = accumulate_channels(
                S, L, nc, types, args.rmax, args.bins, 1)
            ci, ce = {}, {}
            per_channel[label] = {}
            for name, _ in CHANNELS:
                _, p_c, g_c = normalize(acc_m[name][0], acc_m[name][1],
                                        nint_m[name], nf_m, vol_m, edges, nsites)
                ci[name], ce[name] = p_c, g_c
                per_channel[label][name] = {
                    "intra": overlap_metrics(ref_curves["intra"][name], p_c, r),
                    "inter": overlap_metrics(ref_curves["inter"][name], g_c, r),
                }
            run_curves["intra"].append((label, ci))
            run_curves["inter"].append((label, ce))

        for kind, titolo in (("intra", "INTRA-copia P(r)"),
                             ("inter", "INTER-copia g(r)")):
            print(f"\n  Per canale, {titolo}   [sovrapposizione]")
            header = f"  {'canale':<20}" + "".join(f"{lab:>12}" for lab in runs)
            print(header)
            print("  " + "-" * (len(header) - 2))
            for name, _ in CHANNELS:
                row = f"  {name:<20}"
                for lab in runs:
                    row += f"{per_channel[lab][name][kind]['integral_overlap']:>12.4f}"
                print(row)

        print("\n  B3-B3 sono i legami di Hoogsteen dentro le tetradi, B5-B5")
        print("  l'impilamento fra tetradi adiacenti, S-S il backbone.  Uno")
        print("  scarto concentrato su un canale dice quale termine sbaglia;")
        print("  uno scarto distribuito e' il coarse-graining in se'.")
        report["per_channel"] = per_channel

        if args.plot:
            for kind in ("intra", "inter"):
                out = f"{args.plot}_{kind}.png"
                title = args.title or pathlib.Path(args.dataset).name.split("_")[0].upper()
                scores = {lab: {name: per_channel[lab][name][kind]["integral_overlap"]
                                for name, _ in CHANNELS} for lab in runs}
                plot_channels(out, edges, ref_curves[kind], run_curves[kind], kind,
                              title=title, scores=scores)
                print(f"  grafico -> {out}")

    if args.all_pairs:
        types_ap = load_site_types(args.dataset)
        h_ref, edges_ap, ntype = accumulate_type_pairs_intra(
            Sr, Lr, ncr, types_ap, args.rmax, args.bins, args.stride)
        r_ap = 0.5 * (edges_ap[:-1] + edges_ap[1:])
        dr_ap = edges_ap[1] - edges_ap[0]
        total_ref = h_ref.sum()
        h_runs = {}
        for label, path in runs.items():
            S, L, nc, _t = get_run(path)
            h_runs[label], _, _ = accumulate_type_pairs_intra(
                S, L, nc, types_ap, args.rmax, args.bins, 1)
        rows_ap = []
        for a in range(ntype):
            for b in range(a, ntype):
                hr = h_ref[a, b]
                if hr.sum() == 0:
                    continue
                w = hr.sum() / total_ref
                pr = hr / hr.sum() / dr_ap
                ovs = {}
                for label in runs:
                    hm = h_runs[label][a, b]
                    pm = hm / max(hm.sum(), 1) / dr_ap
                    ovs[label] = overlap_metrics(pr, pm, r_ap)["integral_overlap"]
                name = f"{TYPE_NAMES.get(a, a)}-{TYPE_NAMES.get(b, b)}"
                rows_ap.append((name, w, ovs))
        first = next(iter(runs))
        rows_ap.sort(key=lambda x: -x[1] * (1.0 - x[2][first]))
        print(f"\n  INTRA per coppia di tipi (r < {args.rmax} nm), ordinate per "
              f"peso x (1 - sovrapp.) di '{first}'")
        header = f"  {'coppia':<8}{'peso':>7}" + "".join(f"{lab:>12}" for lab in runs) \
            + f"{'scarto':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, w, ovs in rows_ap:
            print(f"  {name:<8}{w:7.3f}" + "".join(f"{ovs[lab]:>12.4f}" for lab in runs)
                  + f"{w * (1.0 - ovs[first]):9.4f}")
        print("  peso = frazione delle coppie intra del riferimento; scarto = peso x "
              "(1 - sovrapposizione): dove conviene lavorare.")
        report["all_pairs_intra"] = [
            {"pair": n, "weight": float(w), "overlap": {k: float(v) for k, v in o.items()}}
            for n, w, o in rows_ap]

    print("\n  g(r) sovrapposta e' NECESSARIA, non sufficiente: Henderson vale")
    print("  per un potenziale di coppia puro, qui c'e' anche un residuo ML a")
    print("  molti corpi. Leggere insieme alla FES su TICA (script 43).")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n  report -> {args.json}")


if __name__ == "__main__":
    main()
