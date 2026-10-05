"""Configurazione di sistema per gli script di analisi e di prior (G-quadruplex CG).

PERCHE'
    Gli script 46-52, add_state_contacts.py e tune_state_contacts.py sono nati
    sul TEL26 con sequenza, tetradi, numero di residui, gruppi dei loop e
    coppie di contatto scritti nel codice.  Qui tutto quello che dipende dal
    sistema sta in un file JSON (system.json), cosi' gli stessi script girano
    su un altro quadruplex senza modifiche.

DOVE SI CERCA IL FILE (il primo che esiste)
    1. variabile d'ambiente MLCG_SYSTEM (percorso del JSON);
    2. system.json nella cartella corrente (i job fanno 'cd' nella cartella
       del sistema);
    3. system.json nella cartella dello script che lo chiede.
    Nessun valore di ripiego nel codice: un sistema senza system.json si
    ferma con un errore chiaro invece di usare in silenzio i numeri del TEL26.

CAMPI (vedi tutorials/tel26/system.json per un esempio completo)
    name, sequence (1 lettera per residuo, una copia), copies,
    sites {lettera: [nomi dei siti nell'ordine del dataset]},
    tetrads [[residui 1-based], ...] dal basso verso l'alto della pila,
    outer_tetrads (default: prima e ultima), hoogsteen_site (es. "B3"),
    loop_groups [[nome, [residui]], ...], contact_pairs "i-j,...",
    file_prefix, reference_dataset, aa {frame_dt_ps, ns_day},
    md {dt_ps, gamma, kT}, production {model, prior_set}.
"""
from __future__ import annotations

import json
import os
import pathlib

_CACHE: dict[str, "System"] = {}


class System:
    def __init__(self, data: dict, path: pathlib.Path):
        self.path = path
        self.raw = data
        self.name = str(data["name"])
        self.sequence = str(data["sequence"]).upper()
        self.nuc = len(self.sequence)
        if "residues_per_copy" in data and int(data["residues_per_copy"]) != self.nuc:
            raise ValueError(f"{path}: residues_per_copy {data['residues_per_copy']} != len(sequence) {self.nuc}")
        self.copies = int(data.get("copies", 1))
        self.sites = {k.upper(): list(v) for k, v in data["sites"].items()}
        self.tetrads = [tuple(int(r) for r in t) for t in data["tetrads"]]
        outer = data.get("outer_tetrads") or [self.tetrads[0], self.tetrads[-1]]
        self.outer_tetrads = [tuple(int(r) for r in t) for t in outer]
        self.hoogsteen_site = str(data.get("hoogsteen_site", "B3"))
        self.loop_groups = [(str(n), tuple(int(r) for r in res)) for n, res in data.get("loop_groups", [])]
        self.contact_pairs = str(data.get("contact_pairs", ""))
        self.file_prefix = str(data.get("file_prefix", self.name))
        self.reference_dataset = data.get("reference_dataset")
        aa = data.get("aa", {})
        self.aa_frame_dt_ps = float(aa.get("frame_dt_ps", 20.0))
        self.aa_ns_day = aa.get("ns_day")
        md = data.get("md", {})
        self.dt_ps = float(md.get("dt_ps", 0.004))
        self.gamma = float(md.get("gamma", 2.0))
        self.kT = float(md.get("kT", 2.49))
        self.production = data.get("production", {})
        self._validate()

    # ── controlli ──
    def _validate(self):
        for r in (x for t in self.tetrads for x in t):
            if not 1 <= r <= self.nuc or self.sequence[r - 1] != "G":
                raise ValueError(f"{self.path}: il residuo {r} delle tetradi non e' una G della sequenza")
        for t in self.outer_tetrads:
            if t not in self.tetrads:
                raise ValueError(f"{self.path}: outer_tetrads {t} non e' fra le tetradi")
        for name, res in self.loop_groups:
            for r in res:
                if not 1 <= r <= self.nuc:
                    raise ValueError(f"{self.path}: gruppo '{name}' con residuo {r} fuori da 1..{self.nuc}")
        for letter in set(self.sequence):
            if letter not in self.sites:
                raise ValueError(f"{self.path}: manca la lista dei siti per i residui '{letter}'")
        if self.hoogsteen_site not in self.sites["G"]:
            raise ValueError(f"{self.path}: hoogsteen_site {self.hoogsteen_site} non e' un sito di G")

    # ── utilita' ──
    def site_index(self, letter: str, site: str) -> int:
        return self.sites[letter.upper()].index(site)

    @property
    def hoogsteen_index(self) -> int:
        return self.site_index("G", self.hoogsteen_site)

    @property
    def max_sites(self) -> int:
        return max(len(v) for v in self.sites.values())

    def is_loop(self, r1: int) -> bool:
        """Residuo 1-based fuori dalle tetradi e non G."""
        return self.sequence[r1 - 1] != "G"

    def label(self, r1: int) -> str:
        return f"{self.sequence[r1 - 1]}{r1}"

    def __repr__(self):
        return f"System({self.name}: {self.nuc} residui x {self.copies} copie, {len(self.tetrads)} tetradi, {self.path})"


def find_system_file(script_dir=None) -> pathlib.Path:
    cands = []
    if os.environ.get("MLCG_SYSTEM"):
        cands.append(pathlib.Path(os.environ["MLCG_SYSTEM"]))
    cands.append(pathlib.Path.cwd() / "system.json")
    if script_dir is not None:
        cands.append(pathlib.Path(script_dir) / "system.json")
    for c in cands:
        if c.is_file():
            return c.resolve()
    raise SystemExit("[ERROR] system.json non trovato (MLCG_SYSTEM, cartella corrente, cartella dello script): "
                     + ", ".join(str(c) for c in cands))


def load_system(script_dir=None, path=None) -> System:
    p = pathlib.Path(path).resolve() if path else find_system_file(script_dir)
    key = str(p)
    if key not in _CACHE:
        _CACHE[key] = System(json.loads(p.read_text()), p)
    return _CACHE[key]
