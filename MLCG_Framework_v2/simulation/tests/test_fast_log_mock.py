"""Log vettoriale di run_cg_md.py contro quello per particella, senza ESPResSo.

Estrae da run_cg_md.py le sole funzioni del log e le esegue su un sistema
finto che riproduce l'interfaccia di ESPResSo, compresa la stranezza delle
slice (pos/type/q restituite in ordine di id).  Uso:  python3 test_fast_log_mock.py
"""
import ast, os, sys, types, math, time
import numpy as np
from scipy.spatial.distance import pdist, squareform
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM)
src = open(os.path.join(SIM, "run_cg_md.py")).read()
tree = ast.parse(src)
want = {"_minimum_image_box", "_minimum_image", "_minimum_image_distance_matrix", "log_diagnostics", "measure_energies", "stringify_pair", "record_structured_sample",
        "record_state_sample", "_OrderedSlice", "_build_fast_log_cache", "_fast_diagnostics",
        "_fast_kinetic", "_fast_max_torque", "_fast_vcf_text", "_legacy_max_torque",
        "_verify_fast_log", "_fast_log_cache"}
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in want]
assert {n.name for n in nodes} == want, want - {n.name for n in nodes}
mod = ast.Module(body=nodes, type_ignores=[])
code = compile(mod, "fast_log_subset", "exec")

rng = np.random.default_rng(5)
class P:
    def __init__(s, **k): s.__dict__.update(k)
    def is_virtual(s): return s.virtual
parts = {}
pid = 100
mol_com_parts, mol_vs_parts = {}, {}
nmol, nsite, nspec = 12, 5, 4
for m in range(nmol):
    c = rng.uniform(0, 6, 3)
    parts[pid] = P(id=pid, type=nspec + 2, mol_id=m, mass=300.0, rinertia=rng.uniform(1, 5, 3),
                   v=rng.normal(size=3), omega_body=rng.normal(size=3), f=rng.normal(size=3),
                   torque_lab=rng.normal(size=3), pos=c, quat=rng.normal(size=4), virtual=False)
    mol_com_parts[m] = pid; pid += 7
    for s in range(nsite):
        parts[pid] = P(id=pid, type=int(rng.integers(0, nspec + 1)), mol_id=m, mass=1e-6,
                       rinertia=np.ones(3), v=np.zeros(3), omega_body=np.zeros(3),
                       f=rng.normal(size=3) * 50, torque_lab=np.zeros(3),
                       pos=c + rng.normal(size=3) * 0.3, quat=np.r_[1, 0, 0, 0.0], virtual=True)
        mol_vs_parts[(m, s)] = pid; pid += 3
# a marker (non virtual, massless)
parts[5] = P(id=5, type=nspec + 3, mol_id=999, mass=1e-6, rinertia=np.ones(3), v=np.zeros(3),
             omega_body=np.zeros(3), f=np.zeros(3), torque_lab=np.zeros(3), pos=np.zeros(3),
             quat=np.r_[1, 0, 0, 0.0], virtual=False)
OPT = {"pos", "type", "q"}
class Slice:
    def __init__(s, ids): s.ids = list(ids)
    def __getattr__(s, name):
        ids = sorted(s.ids) if name in OPT else s.ids   # ESPResSo quirk
        return np.stack([np.asarray(getattr(parts[i], name), dtype=float) for i in ids])
class Part:
    def __iter__(s): return iter([parts[i] for i in sorted(parts)])
    def by_id(s, i): return parts[i]
    def by_ids(s, ids): return Slice(ids)
class Analysis:
    def energy(s): return {"total": 1.0, "kinetic": 2.0, "bonded": 0.5, "non_bonded": 0.25}
# contatto attraverso il bordo: due siti di molecole diverse a 0.12 nm
# tramite l'immagine x+L (posizioni non ripiegate, una fuori dalla box)
BOX = np.array([6.0, 6.5, 7.0])
vs_all = sorted(i for i in parts if parts[i].virtual)
a, b = vs_all[2], vs_all[-3]
assert parts[a].mol_id != parts[b].mol_id
parts[a].type, parts[b].type = 1, 2
parts[a].pos = np.array([-0.05, 3.0, 3.0])
parts[b].pos = np.array([BOX[0] + 6.0 - 0.17, 3.0 + 6.5, 3.0])   # due immagini piu' in la'
system = types.SimpleNamespace(part=Part(), analysis=Analysis(), box_l=BOX,
                               periodicity=[True, True, True])
def writevcf(system, fp):
    idx = {pid: k for k, pid in enumerate(p.id for p in system.part)}
    fp.write("\ntimestep indexed\n")
    for pid, vid in idx.items():
        fp.write(f"{vid} {' '.join(map(str, system.part.by_id(pid).pos))}\n")
espressomd = types.SimpleNamespace(
    io=types.SimpleNamespace(writer=types.SimpleNamespace(vtf=types.SimpleNamespace(writevcf=writevcf))),
    painn=types.SimpleNamespace(get_painn_energy=lambda: -3.0))
from framework_utils import particle_is_virtual, mask_excluded_particle_distances
vs_sorted = sorted(i for i in parts if parts[i].virtual)
excl = {(vs_sorted[0], vs_sorted[7]), (vs_sorted[3], vs_sorted[20])}
ns = dict(np=np, math=math, time=time, pdist=pdist, squareform=squareform, system=system,
          espressomd=espressomd, particle_is_virtual=particle_is_virtual,
          mask_excluded_particle_distances=mask_excluded_particle_distances,
          diagnostic_nonbonded_excluded_pid_pairs=excl, nn_config={"num_species": nspec},
          ml_active=True, mol_com_parts=mol_com_parts, mol_vs_parts=mol_vs_parts,
          num_molecules=nmol, sample_site_keys=sorted(mol_vs_parts),
          state_sample_particle_ids=sorted(i for i in parts if parts[i].mass > 1e-4),
          args=types.SimpleNamespace(legacy_diagnostics=False, sample_npz="x", sample_start_step=0,
                                     state_sample_npz="y"),
          sample_steps=[], sample_com=[], sample_sites=[], state_sample_steps=[],
          state_sample_positions=[], state_sample_velocities=[], state_sample_quaternions=[],
          state_sample_omegas=[])
exec(code, ns)
ns["_fast_log_state"] = {"cache": None, "disabled": False}
cache = ns["_fast_log_cache"](0)
assert cache is not None, "fallback!"
fd_, ld_ = ns["_fast_diagnostics"](cache), ns["log_diagnostics"](0)
print("fast diag:", fd_)
print("legacy   :", ld_)
assert abs(fd_[0] - 0.12) < 1e-9 and set(map(int, fd_[2])) == {a, b}, "contatto attraverso il bordo non visto"
system.periodicity = [False, True, True]   # asse x aperto: il contatto sparisce
_f, _l = ns["_fast_diagnostics"](cache)[0], ns["log_diagnostics"](0)[0]
assert _f > 0.5 and abs(_f - _l) < 1e-12, (_f, _l)
system.periodicity = [True, True, True]
ns["record_structured_sample"](0, cache); ns["record_structured_sample"](0, None)
assert np.array_equal(ns["sample_com"][0], ns["sample_com"][1]) and np.array_equal(ns["sample_sites"][0], ns["sample_sites"][1])
ns["record_state_sample"](0, cache); ns["record_state_sample"](0, None)
for k in ("state_sample_positions", "state_sample_velocities", "state_sample_quaternions", "state_sample_omegas"):
    assert np.array_equal(ns[k][0], ns[k][1]), k
print("kinetic fast/legacy:", ns["measure_energies"](cache)[2:4], ns["measure_energies"]()[2:4])
# negative control: corrupt a cached type -> must fall back
ns["_fast_log_state"] = {"cache": None, "disabled": False}
orig = ns["_build_fast_log_cache"]
def broken():
    c = orig(); c["pair_i"] = c["pair_i"][1:]; c["pair_j"] = c["pair_j"][1:]; c["vs_types"] = np.zeros_like(c["vs_types"]); return c
ns["_build_fast_log_cache"] = broken
assert ns["_fast_log_cache"](0) is None, "il controllo negativo non ha fatto scattare il ritorno"
print("negative control -> fallback ok")
print("ALL OK")
