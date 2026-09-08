#!/usr/bin/env python3
"""Confronto termodinamico di TEL22 allineato alla letteratura CG.

CRITERIO: FES su TICA
  TICA sulle distanze a coppie fra ancoraggi di residuo, FES 2D sulle prime due
  componenti, griglia 80x80.  E' cio' che fa il lavoro piu' vicino a TEL22 -
  Majewski et al., Nat. Commun. 2023 (CGSchNet / TorchMD-Net): TICA sulle
  distanze Ca a coppie, griglia 80x80, piu' RMSD alla nativa.  Le coordinate
  non sono scelte a mano ma apprese dal riferimento, e sono i gradi di liberta'
  piu' lenti: quelli che un modello CG deve riprodurre.  La convergenza si
  giudica QUI.

DIAGNOSTICA: Q e Rg
  Frazione liscia di contatti nativi di Hoogsteen e raggio di girazione.  Non
  sono la metrica della letteratura e non decidono nulla, ma si leggono
  fisicamente mentre TIC1 no, e separano i due modi di fallimento osservati su
  TEL22: legami persi a sagoma invariata contro espansione.

Metriche scalari: js_divergence, fes_mse_kbt2, fes_rmse_kbt e la copertura di
massa del riferimento vengono dal modulo Ala2, importato invece che riscritto,
cosi' i numeri TEL22 e Ala2 sono la stessa computazione.  La JS e' definita
come in letteratura (Bereau/Kremer, arXiv:1907.04082): simmetrica e finita.

USO
  python3 43_tel22_tica_fes.py <dataset.bin> A=<run>/samples.npz [B=... ...]
      [--lag N] [--bins N] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Nucleo metrico condiviso con Ala2: una sola implementazione, cosi' i numeri
# dei due sistemi sono confrontabili per costruzione.
sys.path.insert(0, str(HERE.parents[2] / "ala2_cgnet" / "diagnostics" / "scripts"))

from _tel22_cv import (anchor_distances, load_reference, load_samples,
                       mean_native_anchors, native_hb_distances,
                       radius_of_gyration, rmsd_to_native, smooth_Q)
from _tica import TICA
from analyze_ala2_fes_ab import (KBT_KJ_MOL, js_divergence, probability,
                                 surface_metrics)


def hist2d(x, y, ex, ey):
    counts, _, _ = np.histogram2d(x, y, bins=(ex, ey))
    return counts.astype(np.float64)


def edges_from(x, y, bins, pad=0.02):
    """Bordi dai soli dati di RIFERIMENTO: il modello va giudicato sulla griglia
    del riferimento, altrimenti un modello che esplora regioni assurde si
    ridefinirebbe la griglia addosso e diluirebbe il proprio errore."""
    def one(v):
        lo, hi = float(np.min(v)), float(np.max(v))
        m = pad * (hi - lo) if hi > lo else 1.0
        return np.linspace(lo - m, hi + m, bins + 1)
    return one(x), one(y)


def per_copy_counts(x, y, ncopy, ex, ey):
    """Istogrammi per copia: sono le repliche appaiate del bootstrap."""
    n = x.size // ncopy
    out = []
    for c in range(ncopy):
        sel = slice(c, x.size, ncopy)      # righe ordinate (frame, copia)
        out.append(hist2d(x[sel], y[sel], ex, ey))
    assert n > 0
    return out


def compare(label, ref_counts, run_counts, pseudocount, min_ref):
    m = surface_metrics(ref_counts, run_counts, pseudocount, min_ref)
    return {"run": label, **m}


def fes(counts, pseudocount):
    """-> F in kBT, con il minimo a zero."""
    F = -np.log(probability(counts, pseudocount))
    return F - F.min()


def make_figure(path, ex, ey, panels, pseudocount, fmax=8.0):
    """Figura nella grammatica visiva di CGnet (Wang et al. 2019, fig. 6-7):
    pannelli FES affiancati con la STESSA scala di colore - riferimento, poi
    baseline a pochi corpi, poi modello - piu' un profilo 1D lungo la prima
    coordinata, che nel loro caso e' la coordinata di folding.

    'panels' e' una lista [(etichetta, counts)] con il riferimento per primo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig = plt.figure(figsize=(3.1 * n + 0.6, 6.4))
    gs = fig.add_gridspec(2, n, height_ratios=[1.25, 1.0], hspace=0.32,
                          wspace=0.12)
    xc = 0.5 * (ex[:-1] + ex[1:])
    yc = 0.5 * (ey[:-1] + ey[1:])
    surfaces = [(lab, fes(c, pseudocount)) for lab, c in panels]

    im = None
    for k, (lab, F) in enumerate(surfaces):
        ax = fig.add_subplot(gs[0, k])
        Fm = np.ma.masked_greater(F.T, fmax)
        im = ax.pcolormesh(xc, yc, Fm, cmap="nipy_spectral", vmin=0.0, vmax=fmax,
                           shading="auto")
        ax.set_title(lab, fontsize=11)
        ax.set_xlabel("TIC 1")
        if k == 0:
            ax.set_ylabel("TIC 2")
        else:
            ax.set_yticklabels([])
        ax.set_xlim(xc[0], xc[-1]); ax.set_ylim(yc[0], yc[-1])
    cb = fig.colorbar(im, ax=fig.axes[:n], fraction=0.022, pad=0.012)
    cb.set_label("energia libera  (kBT)")

    # Profilo 1D lungo TIC1: marginalizzazione su TIC2, come il pannello (d)
    # della loro figura sulla chignolina.
    ax = fig.add_subplot(gs[1, :])
    for lab, c in panels:
        p = probability(c, pseudocount).sum(axis=1)
        F1 = -np.log(p); F1 -= F1.min()
        ax.plot(xc, F1, lw=2.0 if lab == panels[0][0] else 1.6,
                ls="-" if lab == panels[0][0] else "--", label=lab)
    ax.set_xlabel("TIC 1  (coordinata lenta)")
    ax.set_ylabel("energia libera  (kBT)")
    ax.set_xlim(xc[0], xc[-1])
    ax.set_ylim(0, fmax)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("runs", nargs="+", help="etichetta=percorso/samples.npz")
    ap.add_argument("--lag", type=int, default=10,
                    help="lag TICA in frame del riferimento (default 10)")
    ap.add_argument("--bins", type=int, default=80,
                    help="lato della griglia FES (default 80, come in letteratura)")
    ap.add_argument("--pseudocount", type=float, default=0.5)
    ap.add_argument("--min-reference-count", type=int, default=1)
    ap.add_argument("--json", default=None)
    ap.add_argument("--plot", default=None,
                    help="figura FES nella grammatica di CGnet (png)")
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        if "=" not in spec:
            ap.error(f"atteso etichetta=percorso, ricevuto {spec!r}")
        k, v = spec.split("=", 1)
        runs[k] = v

    # ── riferimento ─────────────────────────────────────────────────────────
    Sr, Lr, ncr = load_reference(args.dataset)
    print(f"  riferimento: {Sr.shape[0]} frame x {ncr} copie")

    feat_r = anchor_distances(Sr, Lr, ncr)
    # Una traiettoria per copia: le coppie (t, t+lag) non devono attraversare
    # il confine fra copie.
    trajs = [feat_r[c::ncr] for c in range(ncr)]
    tica = TICA(lag=args.lag, n_components=2).fit(trajs)
    print(f"  TICA: lag={args.lag} frame | rango C0={tica.rank_}/{feat_r.shape[1]}"
          f" | autovalori={np.round(tica.eigenvalues_, 4).tolist()}"
          f" | tempi impliciti={np.round(tica.timescales_, 1).tolist()} frame")
    if np.any(tica.timescales_ > 0.5 * Sr.shape[0]):
        print("  [ATTENZIONE] tempo implicito comparabile alla lunghezza del "
              "riferimento: la componente lenta non e' campionata, FES inaffidabile")

    zr = tica.transform(feat_r)
    ex, ey = edges_from(zr[:, 0], zr[:, 1], args.bins)
    ref_counts = hist2d(zr[:, 0], zr[:, 1], ex, ey)

    native = native_hb_distances(Sr, Lr, ncr)
    nat_anchors = mean_native_anchors(Sr, Lr, ncr)
    qr = smooth_Q(Sr, Lr, ncr, native)
    gr = radius_of_gyration(Sr, Lr, ncr)
    qex, qey = edges_from(qr, gr, args.bins)
    ref_qg = hist2d(qr, gr, qex, qey)
    rr = rmsd_to_native(Sr, Lr, ncr, nat_anchors)

    print(f"  riferimento: Q={qr.mean():.3f}+-{qr.std():.3f}  "
          f"Rg={gr.mean():.3f}+-{gr.std():.3f} nm  "
          f"RMSD={rr.mean():.3f} nm")

    # ── run ─────────────────────────────────────────────────────────────────
    report = {"criterion": "tica_fes", "lag": args.lag, "bins": args.bins,
              "tica_eigenvalues": tica.eigenvalues_.tolist(),
              "tica_timescales_frames": tica.timescales_.tolist(),
              "reference": {"frames": int(Sr.shape[0]), "copies": int(ncr),
                            "Q_mean": float(qr.mean()), "Rg_mean_nm": float(gr.mean()),
                            "rmsd_mean_nm": float(rr.mean())},
              "runs": []}

    rows = []
    panel_counts = [("riferimento all-atom", ref_counts)]
    for label, path in runs.items():
        S, L, nc, _t = load_samples(path)
        z = tica.transform(anchor_distances(S, L, nc))
        counts = hist2d(z[:, 0], z[:, 1], ex, ey)
        crit = compare(label, ref_counts, counts, args.pseudocount,
                       args.min_reference_count)

        q = smooth_Q(S, L, nc, native)
        g = radius_of_gyration(S, L, nc)
        diag = surface_metrics(ref_qg, hist2d(q, g, qex, qey),
                              args.pseudocount, args.min_reference_count)
        rmsd = rmsd_to_native(S, L, nc, nat_anchors)

        # Errore standard sulle copie: la dispersione fra copie e' l'unica
        # barra d'errore disponibile da una singola run.
        qs = np.array([q[c::nc].mean() for c in range(nc)])
        gs = np.array([g[c::nc].mean() for c in range(nc)])
        rs = np.array([rmsd[c::nc].mean() for c in range(nc)])
        sem = lambda v: v.std(ddof=1) / np.sqrt(v.size)

        entry = {"run": label, "path": path, "copies": int(nc),
                 "criterion_tica": crit,
                 "diagnostic_q_rg": diag,
                 "Q_mean": float(qs.mean()), "Q_sem": float(sem(qs)),
                 "Rg_mean_nm": float(gs.mean()), "Rg_sem_nm": float(sem(gs)),
                 "rmsd_mean_nm": float(rs.mean()), "rmsd_sem_nm": float(sem(rs)),
                 "rmsd_min_nm": float(rmsd.min())}
        report["runs"].append(entry)
        rows.append(entry)
        panel_counts.append((label, counts))

    # ── tabelle ─────────────────────────────────────────────────────────────
    print(f"\n  CRITERIO - FES su TICA ({args.bins}x{args.bins}, "
          f"griglia definita dal riferimento)")
    print(f"  {'run':<16}{'JS (nats)':>12}{'FES RMSE':>11}{'FES MSE':>10}"
          f"{'massa rif.':>12}")
    print("  " + "-" * 61)
    for r in sorted(rows, key=lambda e: e["criterion_tica"]["js_divergence_nats"]):
        c = r["criterion_tica"]
        print(f"  {r['run']:<16}{c['js_divergence_nats']:>12.4f}"
              f"{c['fes_rmse_kbt']:>11.3f}{c['fes_mse_kbt2']:>10.3f}"
              f"{100 * c['reference_mass_covered_by_sampled_bins']:>11.1f}%")
    print("  JS piu' bassa = piu' vicino al riferimento.  FES in kBT "
          f"(kBT = {KBT_KJ_MOL:.4f} kJ/mol).")

    print("\n  DIAGNOSTICA - non e' il criterio, si legge per interpretare")
    print(f"  {'run':<16}{'Q':>14}{'Rg (nm)':>16}{'RMSD (nm)':>16}{'JS(Q,Rg)':>11}")
    print("  " + "-" * 73)
    for r in rows:
        print(f"  {r['run']:<16}{r['Q_mean']:>8.3f}+-{r['Q_sem']:<5.3f}"
              f"{r['Rg_mean_nm']:>9.3f}+-{r['Rg_sem_nm']:<6.3f}"
              f"{r['rmsd_mean_nm']:>9.3f}+-{r['rmsd_sem_nm']:<6.3f}"
              f"{r['diagnostic_q_rg']['js_divergence_nats']:>11.4f}")
    print(f"  riferimento:    {qr.mean():>8.3f}       {gr.mean():>9.3f}"
          f"        {rr.mean():>9.3f}")

    if args.plot:
        make_figure(args.plot, ex, ey, panel_counts, args.pseudocount)
        print(f"\n  figura -> {args.plot}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n  report -> {args.json}")


if __name__ == "__main__":
    main()
