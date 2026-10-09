#!/usr/bin/env python3
"""Input del pilota AA: TEL26 a 400 K senza il K+ del canale, 4 copie.

PERCHE'
    Nelle corse di Giulia a 400 K la tetrade 3 si apre appena il suo sito di
    K+ si svuota, mentre le tetradi 1 e 2, con un K+ fra loro (resid 28120),
    restano chiuse per 746 ns (02_tetrads_k.py).  Il pilota toglie quel K+ dal
    canale per vedere se anche loro si aprono: unfolding completo in tempi
    accessibili.

COSA FA (solo numpy; gli input sono le copie in AA_unfold/input)
    - dalla struttura di fine equilibratura sposta il K+ --k-resid nel bulk:
      scambia la sua posizione con quella dell'ossigeno di una molecola d'acqua
      lontana dal DNA (> --min-dna nm) e dagli altri K+ (> --min-k nm); l'acqua
      viene traslata rigidamente nel sito lasciato libero.  Ordine degli atomi
      e topologia invariati, carica totale neutra;
    - scrive index.ndx con i gruppi System e DNA_K (DNA + K+, per l'xtc);
    - scrive em.mdp (steepest descent) e md_rep<N>.mdp per --nrep copie, da
      MD.mdp di Giulia: 400 K, velocita' generate con seed diversi, barostato
      C-rescale, --ns ns, xtc del solo DNA_K ogni 10 ps, niente trr;
    - copia .top e kions.itp nella cartella di lavoro.

    --cation Li: tutti i 25 K+ diventano Li+ (Joung-Cheatham per SPC/E, come il K+ della
    topologia: R_min/2 = 0,791 A, eps = 0,3367344 kcal/mol, frcmod.ionsjc_spce), nelle stesse
    posizioni; nessuno spostamento dal canale.  Il Li+ non stabilizza il G-quadruplex: lo
    stato aperto a 400 K senza che un K+ della soluzione rientri nel canale (pilota K+, 8/10).
    Il gruppo dell'xtc si chiama ancora DNA_K (DNA + cationi) per 04 e 12.
    Esito (9/10): anche il Li+ JC rientra nel sito fra le tetradi 1 e 2 in ~20 ns (occupazione
    0,94-0,97): con ioni a carica fissa la desolvatazione dei cationi piccoli e' sottostimata.

    --exclude-r R (nm): canale vietato ai cationi.  Per ogni catione una coordinata di pull
    (distanza dal baricentro dei 12 O6 del core, tetradi 1-3) con potenziale flat-bottom-high:
    zero per r >= R, armonico (--exclude-k) per r < R.  I siti del canale stanno entro ~0,5 nm
    dal baricentro (sopra 1 e sotto 3 compresi), i fosfati oltre 1,0 nm: con R = 0,8 il legame
    nei solchi e sui fosfati resta libero.  Scrive in index.ndx i gruppi O6core e CAT01..CATnn
    e la sezione pull in em.mdp e md_rep*.mdp.  Dopo l'apertura il baricentro degli O6 resta
    definito: il vincolo tiene solo i cationi fuori da una sfera di raggio R attorno ad esso.

USO (Leonardo, login, dopo source hpc/env_leonardo.sh)
    python3 03_make_pilot_inputs.py --input /leonardo_work/IscrB_G4MES/cdemiche/AA_unfold/input \\
        --out /leonardo_work/IscrB_G4MES/cdemiche/AA_unfold/pilot_noK400
    python3 03_make_pilot_inputs.py --input ... --out .../pilot_excl400 --exclude-r 0.8   # canale vietato
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

import numpy as np

DNA_RES = {"DT5", "DT", "DA", "DG", "DT3"}


def read_gro(path):
    L = pathlib.Path(path).read_text().splitlines()
    n = int(L[1])
    rows = L[2:2 + n]
    resid = np.array([int(l[0:5]) for l in rows])
    resn = [l[5:10].strip() for l in rows]
    name = [l[10:15].strip() for l in rows]
    xyz = np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])] for l in rows])
    box = np.array([float(x) for x in L[2 + n].split()[:3]])
    return L, rows, resid, resn, name, xyz, box


def write_gro(path, L, rows, xyz, title):
    out = [title, L[1]]
    for l, x in zip(rows, xyz):
        out.append(f"{l[:20]}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}{l[44:]}")
    out.append(L[2 + len(rows)])
    pathlib.Path(path).write_text("\n".join(out) + "\n")


def set_mdp(text, values):
    """sostituisce o aggiunge chiavi (confronto con '-' e '_' equivalenti)"""
    norm = lambda k: k.strip().lower().replace("_", "-")
    want = {norm(k): (k, v) for k, v in values.items()}
    out, seen = [], set()
    for line in text.splitlines():
        body = line.split(";")[0]
        if "=" in body:
            k = norm(body.split("=")[0])
            if k in want:
                out.append(f"{want[k][0]:<24s} = {want[k][1]}")
                seen.add(k)
                continue
        out.append(line)
    for k, (orig, v) in want.items():
        if k not in seen:
            out.append(f"{orig:<24s} = {v}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="cartella con equil_hybrid_tel26.gro, .top, kions.itp, MD.mdp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gro", default="equil_hybrid_tel26.gro")
    ap.add_argument("--top", default="ibrido_noions_spce.top")
    ap.add_argument("--k-resid", type=int, default=28120, help="K+ da togliere dal canale")
    ap.add_argument("--min-dna", type=float, default=2.5)
    ap.add_argument("--min-k", type=float, default=1.0)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--ns", type=float, default=200.0)
    ap.add_argument("--temp", type=float, default=400.0)
    ap.add_argument("--seed0", type=int, default=4001)
    ap.add_argument("--cation", choices=("K", "Li"), default="K",
                    help="K: toglie il K+ --k-resid dal canale; Li: tutti i K+ diventano Li+")
    ap.add_argument("--exclude-r", type=float, default=0.0,
                    help="raggio (nm) della sfera attorno al baricentro O6 vietata ai cationi (0 = niente)")
    ap.add_argument("--exclude-k", type=float, default=5000.0, help="costante del muro (kJ/mol/nm^2)")
    args = ap.parse_args()
    inp, out = pathlib.Path(args.input), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    L, rows, resid, resn, name, xyz, box = read_gro(inp / args.gro)
    n = len(rows)
    dna = np.array([r in DNA_RES for r in resn])
    isk = np.array([r == "K" for r in resn])
    if args.cation == "Li":
        lines = []
        for l in rows:
            if l[5:10].strip() == "K":
                l = l[:5] + f"{'LI':<5s}" + f"{'LI':>5s}" + l[15:]
            lines.append(l)
        write_gro(out / "start.gro", L, lines, xyz, f"TEL26 400 K, {int(isk.sum())} K+ -> Li+")
        rows_out = lines
        print(f"[INFO] {int(isk.sum())} K+ -> Li+ (stesse posizioni)")
        new = xyz
    kidx = np.where(isk & (resid == args.k_resid))[0] if args.cation == "K" else np.array([0])
    if len(kidx) != 1:
        raise SystemExit(f"[ERROR] K+ resid {args.k_resid} non trovato")
    k = kidx[0]

    def mind(p, sel):
        d = xyz[sel] - p
        d -= box * np.round(d / box)
        return np.sqrt((d ** 2).sum(1)).min()

    if args.cation == "K":
        ow = np.where((np.array(resn) == "SOL") & (np.array(name) == "OW"))[0]
        others = np.where(isk & (np.arange(n) != k))[0]
        best = None
        for o in ow[np.random.default_rng(0).permutation(len(ow))]:
            dd, dk = mind(xyz[o], dna), mind(xyz[o], others)
            if dd > args.min_dna and dk > args.min_k:
                score = min(dd, 2 * dk)
                if best is None or score > best[0]:
                    best = (score, o, dd, dk)
        if best is None:
            raise SystemExit("[ERROR] nessuna acqua abbastanza lontana: ridurre --min-dna / --min-k")
        _, o, dd, dk = best
        wat = np.where(resid == resid[o])[0]                      # OW, HW1, HW2 della stessa molecola
        new = xyz.copy()
        shift = xyz[k] - xyz[o]
        new[wat] = xyz[wat] + shift
        new[k] = xyz[o]
        print(f"[INFO] K+ resid {args.k_resid} (atomo {k + 1}) dal canale a {xyz[o].round(3)}: "
              f"{dd:.2f} nm dal DNA, {dk:.2f} nm dal K+ piu' vicino; acqua resid {resid[o]} nel canale")
        write_gro(out / "start.gro", L, rows, new, f"TEL26 400 K senza K+ {args.k_resid} nel canale")

    # indice: System e DNA_K (numerazione 1-based)
    def block(title, idx):
        lines = [f"[ {title} ]"]
        for i in range(0, len(idx), 15):
            lines.append(" ".join(f"{j + 1:6d}" for j in idx[i:i + 15]))
        return "\n".join(lines)
    ndx = block("System", np.arange(n)) + "\n" + block("DNA_K", np.where(dna | isk)[0]) + "\n"
    pull = {}
    if args.exclude_r > 0:
        core = {4, 5, 6, 10, 11, 12, 16, 17, 18, 22, 23, 24}
        o6 = np.array([i for i in range(n) if dna[i] and name[i] == "O6" and resid[i] in core])
        cat = np.where(isk)[0]
        if len(o6) != 12:
            raise SystemExit(f"[ERROR] {len(o6)} O6 del core invece di 12")
        c = new[o6].mean(0)
        d = new[cat] - c
        d -= box * np.round(d / box)
        r = np.sqrt((d ** 2).sum(1))
        if args.exclude_r >= box.min() / 2:
            raise SystemExit("[ERROR] --exclude-r maggiore di meta' scatola")
        ndx += block("O6core", o6) + "\n" + "".join(block(f"CAT{j + 1:02d}", [i]) + "\n" for j, i in enumerate(cat))
        pull = {"pull": "yes", "pull-ncoords": str(len(cat)), "pull-ngroups": str(len(cat) + 1),
                "pull-group1-name": "O6core", "pull-nstxout": "50000", "pull-nstfout": "0"}
        for j in range(len(cat)):
            g = j + 1
            pull.update({f"pull-group{g + 1}-name": f"CAT{g:02d}",
                         f"pull-coord{g}-type": "flat-bottom-high", f"pull-coord{g}-geometry": "distance",
                         f"pull-coord{g}-groups": f"1 {g + 1}", f"pull-coord{g}-dim": "Y Y Y",
                         f"pull-coord{g}-start": "no", f"pull-coord{g}-init": f"{args.exclude_r:g}",
                         f"pull-coord{g}-rate": "0", f"pull-coord{g}-k": f"{args.exclude_k:g}"})
        inside = np.sort(r)[:4]
        print(f"[INFO] canale vietato: sfera di {args.exclude_r:g} nm attorno al baricentro dei 12 O6, "
              f"k {args.exclude_k:g} kJ/mol/nm^2, {len(cat)} cationi; nella partenza i piu' vicini a "
              + ", ".join(f"{x:.3f}" for x in inside) + f" nm ({int((r < args.exclude_r).sum())} dentro)")
    (out / "index.ndx").write_text(ndx)
    print(f"[INFO] index.ndx: System {n} atomi, DNA_K {int((dna | isk).sum())} atomi"
          + (", O6core e CAT01..CAT%02d" % int(isk.sum()) if pull else ""))

    md = (inp / "MD.mdp").read_text()
    em = set_mdp(md, {"integrator": "steep", "nsteps": "5000", "emtol": "500", "emstep": "0.01",
                      "Tcoupl": "no", "pcoupl": "no", "gen_vel": "no", "constraints": "none",
                      "nstxout": "0", "nstvout": "0", "nstxout-compressed": "0", "nstenergy": "100", **pull})
    (out / "em.mdp").write_text(em)
    nsteps = int(round(args.ns * 1000 / 0.002))
    for r in range(1, args.nrep + 1):
        txt = set_mdp(md, {"nsteps": str(nsteps), "continuation": "no",
                           "gen_vel": "yes", "gen_temp": f"{args.temp:g}", "gen_seed": str(args.seed0 + r),
                           "ref_t": f"{args.temp:g}", "pcoupl": "C-rescale", "tau_p": "2",
                           "nstxout": "0", "nstvout": "0", "nstfout": "0", "nstlog": "50000",
                           "nstenergy": "5000", "nstxout-compressed": "5000", "compressed-x-grps": "DNA_K",
                           **pull})
        (out / f"md_rep{r}.mdp").write_text(txt)
    if args.cation == "K":
        for f in (args.top, "kions.itp"):
            shutil.copy2(inp / f, out / f)
    else:
        # Li+ di Joung-Cheatham (SPC/E): sigma = 2 R_min/2 / 2^(1/6), eps in kJ/mol
        sig = 2 * 0.0791 / 2 ** (1 / 6)
        eps = 0.3367344 * 4.184
        top = (inp / args.top).read_text().splitlines()
        out_top, in_mol = [], False
        for ln in top:
            if ln.split() and ln.split()[0] == "K" and len(ln.split()) >= 6 and not in_mol:
                out_top.append(ln)
                out_top.append(f" LI        LI          6.94      0.0000    A    {sig:.5e}   {eps:.5e} ; Li+ JC/SPC/E")
                continue
            if ln.strip() == '#include "./kions.itp"':
                ln = '#include "./liions.itp"'
            if ln.strip().startswith("[ molecules ]"):
                in_mol = True
            if in_mol and ln.split()[:1] == ["K"]:
                ln = ln.replace("K", "LI", 1)
            out_top.append(ln)
        if not any(" LI " in ln for ln in out_top) or '#include "./liions.itp"' not in out_top:
            raise SystemExit("[ERROR] topologia inattesa: tipo K o include di kions.itp non trovati")
        (out / args.top).write_text("\n".join(out_top) + "\n")
        (out / "liions.itp").write_text("[ moleculetype ]\n; molname       nrexcl\nLI              1\n\n[ atoms ]\n"
                                        "; id    at type         res nr  residu name     at name  cg nr  charge\n"
                                        "1       LI              1       LI              LI       1      1.00000\n")
        print(f"[INFO] Li+: sigma {sig:.6f} nm, eps {eps:.6f} kJ/mol, massa 6.94 -> {args.top}, liions.itp")
    print(f"[DONE] {out}: start.gro, index.ndx, em.mdp, md_rep1..{args.nrep}.mdp ({args.ns:g} ns, {args.temp:g} K), "
          f"{args.top}, {'kions' if args.cation == 'K' else 'liions'}.itp")


if __name__ == "__main__":
    main()
