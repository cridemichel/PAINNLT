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
      [--rmax 2.0] [--bins 200] [--types] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _tel22_cv import (NUC, SITE_NAME, load_reference, load_samples,
                       load_site_types, _box_at)
from _hb_common import mi


def _copy_of_site(nmol_total: int, ncopy: int) -> np.ndarray:
    """Indice di copia per ciascuna molecola."""
    return np.repeat(np.arange(ncopy), NUC)


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
    cop = np.repeat(np.arange(ncopy), NUC)
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
        n = pos.shape[0]
        if n < 2:
            continue
        for i in range(0, n, 512):
            d = mi(pos[i:i + 512, None, :] - pos[None, :, :], Lt)
            r = np.linalg.norm(d, axis=2)
            gi = np.arange(i, min(i + 512, n))
            upper = gi[:, None] < np.arange(n)[None, :]
            same = who[i:i + 512, None] == who[None, :]
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


def plot_channels(path, edges, ref, runs, kind="intra"):
    """Curve per canale, riferimento in continuo e modelli tratteggiati.

    ref/runs: dict canale -> curva.  runs e' [(etichetta, dict)].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = 0.5 * (edges[:-1] + edges[1:])
    names = [n for n, _ in CHANNELS]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    ylab = "P(r)   (nm$^{-1}$)" if kind == "intra" else "g(r)"
    for ax, name in zip(axes.ravel(), names):
        ax.plot(r, ref[name], color="k", lw=2.2, label="riferimento all-atom")
        for lab, cur in runs:
            ax.plot(r, cur[name], lw=1.6, ls="--", label=lab)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("r  (nm)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, r[-1])
        ax.grid(alpha=0.25, lw=0.5)
        if kind == "inter":
            ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    axes.ravel()[0].legend(frameon=False, fontsize=9)
    fig.suptitle("TEL22  -  "
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
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        if "=" not in spec:
            ap.error(f"atteso etichetta=percorso, ricevuto {spec!r}")
        k, v = spec.split("=", 1)
        runs[k] = v

    print(f"  riferimento: {args.dataset}")
    Sr, Lr, ncr = load_reference(args.dataset)
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
        S, L, nc, _t = load_samples(path)
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

    print("\n  g(r) sovrapposta e' NECESSARIA, non sufficiente: Henderson vale")
    print("  per un potenziale di coppia puro, qui c'e' anche un residuo ML a")
    print("  molti corpi. Leggere insieme alla FES su TICA (script 43).")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n  report -> {args.json}")


if __name__ == "__main__":
    main()
