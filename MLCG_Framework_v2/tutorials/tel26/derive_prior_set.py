#!/usr/bin/env python3
"""Nuovo insieme di prior per le sole corse SOLO-PRIOR, senza ricostruire il dataset.

PERCHE'
    build_cg_dataset.py fa due cose: stima i prior (Boltzmann inversion dei
    legami, WCA) e sottrae le loro forze dal riferimento per ottenere il
    residuo da allenare.  Per una corsa con i soli prior serve solo la prima,
    e dei prior cambiano soltanto i contatti pair-specific (Morse/LJ delle
    tetradi e dell'impilamento): legami, angoli e WCA restano quelli
    dell'insieme di partenza.  Ricostruire il dataset -- 25-30 minuti di CPU
    per rileggere 14 GB di forze -- non serve.

    Questo script prende i cg_priors di un insieme esistente, sostituisce
    TUTTI i contatti pair-specific con quelli della topologia del nuovo
    insieme (e il blocco debye_huckel, se c'e') e copia rigid_bodies_info.
    Il dataset del nuovo insieme NON esiste: 04/05 usano quello canonico come
    configurazione iniziale solo se il residuo ML non e' attivo, e 03 si
    rifiuta di allenare.  Quando un insieme merita il ML, si costruisce il suo
    dataset con lo stadio dataset, come sempre.

USO (in tutorials/tel26)
    python3 derive_prior_set.py --base b3stack --set b3core
        legge  cg_priors.b3stack.json, rigid_bodies_info.b3stack.json,
               tel26_topology.b3core.json
        scrive cg_priors.b3core.json, rigid_bodies_info.b3core.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "preprocessing"))
from prior_kernels import normalize_debye_huckel  # noqa: E402

CONTACT_TYPES = ("morse", "lj")


def names(s):
    sfx = f".{s}" if s else ""
    return (f"tel26_topology{sfx}.json", f"cg_priors{sfx}.json", f"rigid_bodies_info{sfx}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="insieme di partenza ('' = canonico)")
    ap.add_argument("--set", required=True, help="nome del nuovo insieme")
    ap.add_argument("--force", action="store_true", help="sovrascrivi file esistenti")
    args = ap.parse_args()
    if not args.set or args.set == args.base:
        sys.exit("[ERROR] --set deve essere un nome nuovo")

    _, base_priors, base_rb = names(args.base)
    new_topo, new_priors, new_rb = names(args.set)
    for f in (base_priors, base_rb, new_topo):
        if not Path(f).is_file():
            sys.exit(f"[ERROR] manca {f}")
    for f in (new_priors, new_rb):
        if Path(f).exists() and not args.force:
            sys.exit(f"[ERROR] {f} esiste gia' (usa --force)")
    if Path(f"tel26_{args.set}_dataset.bin").exists():
        sys.exit(f"[ERROR] esiste gia' tel26_{args.set}_dataset.bin: i prior di quell'insieme "
                 "vengono dal suo dataset, non si derivano")

    priors = json.loads(Path(base_priors).read_text())
    topo = json.loads(Path(new_topo).read_text())

    kept = [b for b in priors["bonds"] if str(b.get("type", "harmonic")).lower() not in CONTACT_TYPES]
    removed = len(priors["bonds"]) - len(kept)
    contacts = []
    for b in topo["bonds"]:
        if str(b.get("type", "harmonic")).lower() not in CONTACT_TYPES:
            continue
        if b.get("exclude_wca", False):
            sys.exit("[ERROR] un contatto con exclude_wca=true cambierebbe le esclusioni WCA: "
                     "serve il dataset completo")
        if b.get("type") == "morse" and not all(isinstance(b.get(k), (int, float))
                                                for k in ("D", "a", "r0", "r_cut")):
            sys.exit(f"[ERROR] Morse con parametri non numerici (r0='auto'?): {b}")
        contacts.append(dict(b))
    if not contacts:
        sys.exit(f"[ERROR] {new_topo} non ha contatti pair-specific")
    priors["bonds"] = kept + contacts

    dh = normalize_debye_huckel(topo.get("debye_huckel"), topo["mapping"]["site_types"])
    if dh:
        priors["debye_huckel"] = dh
    else:
        priors.pop("debye_huckel", None)
    priors["derived_prior_set"] = {
        "base": args.base or "canonico", "topology": new_topo,
        "note": "contatti pair-specific e DH dalla topologia; legami, angoli e WCA dalla base. "
                "Nessun dataset: solo corse con i soli prior.",
    }

    Path(new_priors).write_text(json.dumps(priors, indent=4) + "\n")
    shutil.copyfile(base_rb, new_rb)
    roles = {}
    for b in contacts:
        key = f"{b.get('type')}/{b.get('role', '?')}"
        roles[key] = roles.get(key, 0) + 1
    print(f"[INFO] base {base_priors}: {len(kept)} legami tenuti, {removed} contatti sostituiti")
    print(f"[INFO] da {new_topo}: {len(contacts)} contatti {roles}"
          f"{', DH' if dh else ''}")
    print(f"[DONE] {new_priors}, {new_rb}")
    print(f"       Solo corse con i soli prior: CLASSICAL=1 DISABLE_ML=1 PRIOR_SET={args.set}")


if __name__ == "__main__":
    main()
