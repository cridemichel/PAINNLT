import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError
import numpy as np

from geometry_utils import diagonalize_inertia_tensor, minimum_image_distance_matrix
from prior_kernels import (
    load_tabulated_prior,
    switched_morse_radial_force_array,
    tabulated_angle_forces,
    tabulated_dihedral_forces,
    tabulated_distance_forces,
)
import struct
import json
import copy
import argparse
import sys
import os
from scipy.ndimage import gaussian_filter1d


DEFAULT_MORSE_SWITCH_FRACTION = 0.75


def normalize_morse_type_pairs(entries, known_site_types):
    """Validate broad non-bonded Morse priors selected by CG site type."""
    known = {int(x) for x in known_site_types}
    result = []
    seen = {}
    for index, raw in enumerate(entries or []):
        if not isinstance(raw, dict):
            raise ValueError(f"morse_type_pairs[{index}] must be an object")
        type_i = int(raw["type_i"])
        type_j = int(raw["type_j"])
        if type_i not in known or type_j not in known:
            raise ValueError(
                f"morse_type_pairs[{index}] uses unknown CG site types "
                f"{type_i}, {type_j}; known types are {sorted(known)}"
            )
        if "r_cut" not in raw:
            raise ValueError(
                f"morse_type_pairs[{index}] requires an explicit r_cut because "
                "type-pair Morse contributes to the regular neighbor-search cutoff"
            )
        pair = tuple(sorted((type_i, type_j)))
        if pair in seen:
            raise ValueError(
                f"Duplicate Morse type pair {pair[0]} <-> {pair[1]} in "
                f"morse_type_pairs[{seen[pair]}] and morse_type_pairs[{index}]"
            )
        seen[pair] = index
        D = float(raw["D"])
        a = float(raw["a"])
        r0 = float(raw["r0"])
        r_cut = float(raw["r_cut"])
        if D < 0.0 or a <= 0.0 or r0 < 0.0 or r_cut <= r0:
            raise ValueError(
                f"Invalid morse_type_pairs[{index}] parameters: "
                f"D={D}, a={a}, r0={r0}, r_cut={r_cut}"
            )
        r_switch = float(
            raw.get("r_switch", r0 + DEFAULT_MORSE_SWITCH_FRACTION * (r_cut - r0))
        )
        if not (r0 < r_switch < r_cut):
            raise ValueError(
                f"morse_type_pairs[{index}] requires r0 < r_switch < r_cut; "
                f"got r0={r0}, r_switch={r_switch}, r_cut={r_cut}"
            )
        item = copy.deepcopy(raw)
        item.update({
            "type_i": pair[0],
            "type_j": pair[1],
            "D": D,
            "a": a,
            "r0": r0,
            "r_switch": r_switch,
            "r_cut": r_cut,
        })
        result.append(item)
    return result




# =====================================================================
# 1. PARSING ARGOMENTI E CARICAMENTO TOPOLOGIA
# =====================================================================
parser = argparse.ArgumentParser(description="Coarse-Graining Dataset Builder with Direct Boltzmann Inversion")
parser.add_argument("-c", "--topology", type=str, required=True, help="File di topologia (es. .tpr o .gro)")
parser.add_argument("-f", "--trajectory", type=str, required=True, help="File di traiettoria (es. .trr o .xtc)")
parser.add_argument("-j", "--config", type=str, default="topology_config.json", help="File JSON con topologia CG e mapping")

parser.add_argument("-p", "--priors", type=str, default=None, help="File JSON con prior pre-esistenti (salta calcolo statistico DBI e carica questo)")
parser.add_argument("-o", "--output", type=str, default="cg_dataset.bin", help="Nome del file binario di output")
parser.add_argument("--priors-output", type=str, default=None, help="Output JSON dei prior; default: cg_priors.json accanto al dataset")
parser.add_argument("--rb-info-output", type=str, default=None, help="Output JSON dei rigid body; default: rigid_bodies_info.json accanto al dataset")
parser.add_argument("--clip_forces", type=float, default=None, help="Valore massimo per il modulo delle forze residue. Se non specificato, nessun clip viene applicato (raccomandato per priors analitici dolci).")
args = parser.parse_args()

output_path = os.path.abspath(args.output)
output_dir = os.path.dirname(output_path) or os.getcwd()
os.makedirs(output_dir, exist_ok=True)
args.output = output_path
priors_output_path = os.path.abspath(args.priors_output) if args.priors_output else os.path.join(output_dir, "cg_priors.json")
rb_info_output_path = os.path.abspath(args.rb_info_output) if args.rb_info_output else os.path.join(output_dir, "rigid_bodies_info.json")

try:
    with open(args.config, "r") as mf:
        config_data = json.load(mf)
except FileNotFoundError:
    print(f"[ERRORE] File di configurazione '{args.config}' non trovato!")
    sys.exit(1)

TEMPERATURE = config_data.get("temperature", 300.0)
KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184
# Convert kJ/mol/K
# 1 kcal = 4.184 kJ.
# R = N_A * k_B = 8.314 J / (mol K) = 0.008314 kJ / (mol K)
R_KJ_MOL_K = 0.008314462618
BETA = 1.0 / (R_KJ_MOL_K * TEMPERATURE)

MAPPING_DATA = config_data.get("mapping", {})
MAPPING_METHOD = MAPPING_DATA.get("mapping_method", "COM")
mapping_by_resname = MAPPING_DATA.get("residues", {})
site_types = MAPPING_DATA.get("site_types", {})

BONDS = copy.deepcopy(config_data.get("bonds", []))
MORSE_TYPE_PAIRS = normalize_morse_type_pairs(
    copy.deepcopy(config_data.get("morse_type_pairs", [])), site_types.values()
)
if MORSE_TYPE_PAIRS and any(
    isinstance(b, dict) and str(b.get("type", "")).lower() == "morse"
    for b in BONDS
):
    print(
        "[WARNING] Topology enables both pair-specific Morse contacts and "
        "site type-pair Morse interactions. The priors are additive; verify that "
        "this is intentional and not double counting."
    )
ANGLES = copy.deepcopy(config_data.get("angles", []))
DIHEDRALS = copy.deepcopy(config_data.get("dihedrals", []))

# Optional geometry defaults for bonded priors.  They are applied before both
# Boltzmann inversion and WCA 1-3 classification, and are copied into the
# generated cg_priors.json.  Omitting the setting preserves COM-based angles.
PRIOR_GEOMETRY = config_data.get("prior_geometry", {})
_DEFAULT_ANGLE_SITE = PRIOR_GEOMETRY.get("default_angle_site", None)
if _DEFAULT_ANGLE_SITE is not None:
    _DEFAULT_ANGLE_SITE = int(_DEFAULT_ANGLE_SITE)
    for _angle in ANGLES:
        if not isinstance(_angle, dict):
            continue
        _angle.setdefault("site_i", _DEFAULT_ANGLE_SITE)
        _angle.setdefault("site_j", _DEFAULT_ANGLE_SITE)
        _angle.setdefault("site_k", _DEFAULT_ANGLE_SITE)

# WCA exclusion membership is a topological property, not a property of the
# analytic energy form.  ``exclude_wca`` is therefore carried explicitly on
# generated priors.  For legacy configs, distance restraints of type Morse are
# non-topological by default; ordinary bond potentials retain the old default.
for _bond in BONDS:
    if isinstance(_bond, dict):
        _bond.setdefault("exclude_wca", str(_bond.get("type", "harmonic")).lower() != "morse")
for _angle in ANGLES:
    if isinstance(_angle, dict):
        _angle.setdefault("exclude_wca", True)

# When --priors is supplied, WCA topology exclusions must be derived from the
# exact bonded topology that will later be subtracted and used at runtime.
_preloaded_priors = None
if args.priors:
    with open(args.priors, "r") as _pf:
        _preloaded_priors = json.load(_pf)
WCA_TOPOLOGY_BONDS = (
    _preloaded_priors.get("bonds", BONDS) if _preloaded_priors is not None else BONDS
)
WCA_TOPOLOGY_ANGLES = (
    _preloaded_priors.get("angles", ANGLES) if _preloaded_priors is not None else ANGLES
)
WCA_SIGMA = config_data.get("wca_sigma", 0.0)
WCA_EPSILON = config_data.get("wca_epsilon", 0.0)
WCA_OVERRIDES = config_data.get("wca_overrides", {})

# Pair-specific WCA guard parameters.  The cutoff is inferred from a low
# percentile of the physical distance distribution, while epsilon controls
# how rapidly the repulsion grows only after entering the short-range core.
WCA_QUANTILE_PERCENT = float(config_data.get("wca_quantile_percent", 0.1))
WCA_GUARD_FRACTION = float(config_data.get("wca_guard_fraction", 0.80))
WCA_GUARD_KBT = float(config_data.get("wca_guard_kbt", 10.0))
# Keep the WCA deep-core guard strictly below the shortest physical
# nonbonded distance observed for each type pair.  A value below 1.0 leaves
# a small numerical/statistical buffer instead of placing the guard exactly
# on the most compressed training configuration.
WCA_PHYSICAL_GUARD_MARGIN = float(config_data.get("wca_physical_guard_margin", 0.98))
# Legacy OOD decoys write an all-zero residual target for the entire frame,
# although only one local contact is intentionally displaced.  Without a
# per-molecule loss mask this creates artificial labels away from the OOD
# contact, so the safe default is no decoys.  The legacy path is retained only
# behind an explicit opt-in for reproducibility of old experiments.
DECOY_TARGET_FRACTION = float(config_data.get("decoy_target_fraction", 0.0))
DECOY_RANDOM_SEED = int(config_data.get("decoy_random_seed", 20260808))
ALLOW_UNMASKED_ZERO_TARGET_DECOYS = bool(
    config_data.get("allow_unmasked_zero_target_decoys", False)
)

if not (0.0 < WCA_QUANTILE_PERCENT < 50.0):
    raise ValueError("wca_quantile_percent must be in (0, 50)")
if not (0.0 < WCA_GUARD_FRACTION < 1.0):
    raise ValueError("wca_guard_fraction must be in (0, 1)")
if WCA_GUARD_KBT <= 0.0:
    raise ValueError("wca_guard_kbt must be > 0")
if not (0.0 < WCA_PHYSICAL_GUARD_MARGIN <= 1.0):
    raise ValueError("wca_physical_guard_margin must be in (0, 1]")
if not (0.0 <= DECOY_TARGET_FRACTION < 1.0):
    raise ValueError("decoy_target_fraction must be in [0, 1)")
if DECOY_TARGET_FRACTION > 0.0 and not ALLOW_UNMASKED_ZERO_TARGET_DECOYS:
    raise ValueError(
        "decoy_target_fraction > 0 requests legacy whole-frame zero-target OOD decoys, "
        "but the current binary schema has no per-molecule loss mask. Set "
        "decoy_target_fraction=0 (recommended) or explicitly set "
        "allow_unmasked_zero_target_decoys=true only to reproduce a legacy ablation."
    )
RIGID_BODIES_CONFIG = config_data.get("rigid_bodies", {})

print(f"[INFO] Caricamento MDAnalysis: {args.topology}, {args.trajectory}...")
u = mda.Universe(args.topology, args.trajectory)

# Mapping dictionary for atomic masses
ATOMIC_MASSES = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974, 'S': 32.065, 'K': 39.098}
def get_mass(atom_name):
    alpha_chars = ''.join([c for c in atom_name if c.isalpha()]).upper()
    return ATOMIC_MASSES.get(alpha_chars[0], 12.0) if alpha_chars else 12.0



def _bond_excludes_wca(bond):
    """Return whether a bonded prior defines a topological 1-2 WCA exclusion."""
    if isinstance(bond, dict):
        return bool(bond.get("exclude_wca", str(bond.get("type", "harmonic")).lower() != "morse"))
    # Legacy list/tuple bonds are ordinary topological bonds.
    return isinstance(bond, (list, tuple)) and len(bond) >= 2


def _angle_excludes_wca(angle):
    """Return whether an angle prior defines an explicit 1-3 WCA exclusion."""
    return isinstance(angle, dict) and bool(angle.get("exclude_wca", True))


def build_wca_topology_exclusions(bonds, angles, num_molecules):
    """Return molecule-level topological 1-2 and explicit-angle 1-3 exclusions.

    WCA exclusions are determined by explicit topology semantics, not merely by
    the presence of an analytic restraint between two molecules. Distance
    restraints such as Morse terms therefore need not suppress the short-range
    WCA core. 1-2 molecule pairs are retained for topology classification,
    while only explicitly bonded virtual-site pairs are excluded from WCA;
    explicit 1-3 angle endpoints remain all-sites exclusions.
    """
    direct_pairs = set()
    for bond in bonds:
        if not _bond_excludes_wca(bond):
            continue
        if isinstance(bond, dict):
            mi, mj = int(bond["mol_i"]), int(bond["mol_j"])
        elif isinstance(bond, (list, tuple)) and len(bond) >= 2:
            mi, mj = int(bond[0]), int(bond[1])
        else:
            continue
        if 0 <= mi < num_molecules and 0 <= mj < num_molecules and mi != mj:
            direct_pairs.add((min(mi, mj), max(mi, mj)))

    one_three_pairs = set()
    for angle in angles:
        if not _angle_excludes_wca(angle):
            continue
        mi, mk = int(angle["mol_i"]), int(angle["mol_k"])
        if 0 <= mi < num_molecules and 0 <= mk < num_molecules and mi != mk:
            key = (min(mi, mk), max(mi, mk))
            if key not in direct_pairs:
                one_three_pairs.add(key)

    return direct_pairs, one_three_pairs


def build_wca_direct_site_exclusions(bonds, num_molecules):
    """Return normalized (mol_i, mol_j, site_i, site_j) 1-2 exclusions.

    Only the virtual-site pair explicitly referenced by a bonded prior is
    excluded.  COM-level bonds still define a topological 1-2 relation but do
    not suppress WCA between any virtual-site cross pair.
    """
    result = set()
    for bond in bonds:
        if not _bond_excludes_wca(bond):
            continue
        if isinstance(bond, dict):
            mi, mj = int(bond["mol_i"]), int(bond["mol_j"])
            si, sj = int(bond.get("site_i", -1)), int(bond.get("site_j", -1))
        elif isinstance(bond, (list, tuple)) and len(bond) >= 2:
            mi, mj = int(bond[0]), int(bond[1])
            si = int(bond[2]) if len(bond) > 2 else -1
            sj = int(bond[3]) if len(bond) > 3 else -1
        else:
            continue
        if not (0 <= mi < num_molecules and 0 <= mj < num_molecules and mi != mj):
            continue
        if si < 0 or sj < 0:
            continue
        if mi < mj:
            result.add((mi, mj, si, sj))
        else:
            result.add((mj, mi, sj, si))
    return result


def _direct_site_excluded_mask(
    pair_i, pair_j, flat_mols, flat_sites, direct_site_exclusions, num_molecules
):
    """Vectorized membership mask for explicit bonded virtual-site pairs."""
    if not direct_site_exclusions or len(pair_i) == 0:
        return np.zeros(len(pair_i), dtype=bool)
    site_stride = int(np.max(flat_sites)) + 1
    site_codes = flat_mols.astype(np.int64) * site_stride + flat_sites.astype(np.int64)
    pair_base = int(num_molecules) * site_stride + 1
    pair_codes = site_codes[pair_i] * pair_base + site_codes[pair_j]
    excluded_codes = np.asarray([
        (mi * site_stride + si) * pair_base + (mj * site_stride + sj)
        for mi, mj, si, sj in sorted(direct_site_exclusions)
    ], dtype=np.int64)
    return np.isin(pair_codes, excluded_codes, assume_unique=False)

def get_unwrapped_positions(positions, box_dim):
    unwrapped = np.copy(positions)
    ref = unwrapped[0]
    for i in range(1, len(unwrapped)):
        dvec = unwrapped[i] - ref
        dvec -= box_dim * np.round(dvec / box_dim)
        unwrapped[i] = ref + dvec
    return unwrapped

def compute_com(positions, masses):
    total_mass = np.sum(masses)
    return np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass

def compute_inertia_tensor(positions, masses, center):
    rel_pos = positions - center
    I = np.zeros((3, 3))
    for r, m in zip(rel_pos, masses):
        I += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return I


def kabsch_align(P, Q):
    """Finds rotation matrix R that aligns P to Q using SVD"""
    H = np.dot(P.T, Q)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R


def reconstruct_rigid_sites_for_frame(
    raw_frame_sites, frame_centers, mol_resnames, rigid_bodies_info
):
    """Rebuild mapped sites on the ideal rigid-body geometry for one frame.

    This is the single geometry path used both by the WCA fit and by Pass 2.
    The ideal body-frame offsets are rotated onto the instantaneous mapped
    orientation with Kabsch, exactly as they are during prior subtraction.
    """
    frame_sites = [list(sites) for sites in raw_frame_sites]

    for mol_idx, r_name in enumerate(mol_resnames):
        rb_info = rigid_bodies_info.get(r_name)
        if not rb_info or "sites" not in rb_info:
            continue

        site_names = list(rb_info["sites"].keys())
        if len(site_names) < 2:
            continue

        ideal_rel = []
        inst_rel = []
        site_indices = []
        for s_idx, (_site_type, site_pos) in enumerate(frame_sites[mol_idx]):
            if s_idx >= len(site_names):
                continue
            site_name = site_names[s_idx]
            ideal_rel.append(rb_info["sites"][site_name]["relative_pos_nm"])
            inst_rel.append(np.asarray(site_pos, dtype=float) - frame_centers[mol_idx])
            site_indices.append(s_idx)

        if len(ideal_rel) < 2:
            continue

        ideal_rel = np.asarray(ideal_rel, dtype=float)
        inst_rel = np.asarray(inst_rel, dtype=float)
        rotation = kabsch_align(ideal_rel, inst_rel)
        ideal_rel_rotated = (rotation @ ideal_rel.T).T

        for i_local, s_idx in enumerate(site_indices):
            new_pos = np.asarray(frame_centers[mol_idx], dtype=float) + ideal_rel_rotated[i_local]
            frame_sites[mol_idx][s_idx] = (
                frame_sites[mol_idx][s_idx][0],
                new_pos,
            )

    return frame_sites


def accumulate_wca_distance_samples(
    frame_sites,
    box_dim,
    wca_direct_site_exclusions,
    wca_one_three_mol_matrix,
    all_pairwise_distances,
    exact_pair_min,
    sample_limit=1000,
):
    """Accumulate WCA statistics for one already-rigidified physical frame.

    The exact minimum is updated before bounded statistical subsampling so the
    physical-support guard cannot miss rare short contacts.
    """
    flat_pos = []
    flat_types = []
    flat_mols = []
    flat_sites = []
    for m_idx, mol_sites in enumerate(frame_sites):
        for site_idx, (site_type, site_pos) in enumerate(mol_sites):
            flat_pos.append(site_pos)
            flat_types.append(int(site_type))
            flat_mols.append(m_idx)
            flat_sites.append(site_idx)

    flat_pos = np.asarray(flat_pos, dtype=float)
    flat_types = np.asarray(flat_types, dtype=np.int32)
    flat_mols = np.asarray(flat_mols, dtype=np.int32)
    flat_sites = np.asarray(flat_sites, dtype=np.int32)

    dist_matrix = minimum_image_distance_matrix(flat_pos, box_dim)
    same_mol_mask = flat_mols[:, None] == flat_mols[None, :]
    dist_matrix[same_mol_mask] = np.inf

    i_idx, j_idx = np.triu_indices(len(flat_pos), k=1)
    valid_dist = dist_matrix[i_idx, j_idx]
    topology_excluded = wca_one_three_mol_matrix[
        flat_mols[i_idx], flat_mols[j_idx]
    ]
    topology_excluded |= _direct_site_excluded_mask(
        i_idx, j_idx, flat_mols, flat_sites, wca_direct_site_exclusions, len(frame_sites)
    )

    types_i = flat_types[i_idx]
    types_j = flat_types[j_idx]
    t1 = np.minimum(types_i, types_j)
    t2 = np.maximum(types_i, types_j)

    close_mask = (valid_dist < 1.5) & (~topology_excluded)
    if not np.any(close_mask):
        return

    valid_dist = valid_dist[close_mask]
    t1 = t1[close_mask]
    t2 = t2[close_mask]
    pair_ids = t1 * 10000 + t2

    for pid in np.unique(pair_ids):
        dists_full = valid_dist[pair_ids == pid]
        pair = (str(int(pid // 10000)), str(int(pid % 10000)))
        all_pairwise_distances.setdefault(pair, [])

        frame_pair_min = float(np.min(dists_full))
        exact_pair_min[pair] = min(
            exact_pair_min.get(pair, np.inf), frame_pair_min
        )

        dists = dists_full
        if len(dists) > sample_limit:
            dists = np.random.choice(dists, sample_limit, replace=False)
        all_pairwise_distances[pair].extend(dists)


def mic_vector(pos1, pos2, box_dim):
    """Vettore da pos1 a pos2 con Minimum Image Convention"""
    dvec = pos2 - pos1
    return dvec - box_dim * np.round(dvec / box_dim)

def resolve_site_position(frame_centers, frame_sites, mol_idx, site_idx):
    """Resolve a prior site reference using the runtime convention.

    site_idx == -1 addresses the rigid-body COM; non-negative values are
    virtual-site indices, not site-type identifiers.
    """
    if site_idx == -1:
        return frame_centers[mol_idx]
    if site_idx < 0 or site_idx >= len(frame_sites[mol_idx]):
        raise IndexError(
            f"Invalid site index {site_idx} for molecule {mol_idx}; "
            f"available sites: {len(frame_sites[mol_idx])}"
        )
    return frame_sites[mol_idx][site_idx][1]

def get_angle(pos_i, pos_j, pos_k, box_dim):
    r_ji = mic_vector(pos_j, pos_i, box_dim)
    r_jk = mic_vector(pos_j, pos_k, box_dim)
    d_ji = np.linalg.norm(r_ji)
    d_jk = np.linalg.norm(r_jk)
    if d_ji < 1e-6 or d_jk < 1e-6: return 0.0
    cos_theta = np.clip(np.dot(r_ji, r_jk) / (d_ji * d_jk), -1.0, 1.0)
    return np.arccos(cos_theta)

def angle_forces(pos_i, pos_j, pos_k, box_dim, k, theta0):
    r_ji = mic_vector(pos_j, pos_i, box_dim)
    r_jk = mic_vector(pos_j, pos_k, box_dim)
    d_ji = np.linalg.norm(r_ji)
    d_jk = np.linalg.norm(r_jk)
    if d_ji < 1e-6 or d_jk < 1e-6: return np.zeros(3), np.zeros(3), np.zeros(3)
    cos_theta = np.clip(np.dot(r_ji, r_jk) / (d_ji * d_jk), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    if sin_theta < 1e-6: return np.zeros(3), np.zeros(3), np.zeros(3)
    dV_dtheta = k * (theta - theta0)
    grad_i_cos = r_jk / (d_ji * d_jk) - cos_theta * r_ji / (d_ji**2)
    grad_k_cos = r_ji / (d_ji * d_jk) - cos_theta * r_jk / (d_jk**2)
    force_i = (dV_dtheta / sin_theta) * grad_i_cos
    force_k = (dV_dtheta / sin_theta) * grad_k_cos
    return force_i, -(force_i + force_k), force_k

def get_dihedral(pos_i, pos_j, pos_k, pos_l, box_dim):
    b1 = mic_vector(pos_i, pos_j, box_dim)
    b2 = mic_vector(pos_j, pos_k, box_dim)
    b3 = mic_vector(pos_k, pos_l, box_dim)
    m1 = np.cross(b1, b2)
    m2 = np.cross(b2, b3)
    m1_sq = np.dot(m1, m1)
    m2_sq = np.dot(m2, m2)
    if m1_sq < 1e-12 or m2_sq < 1e-12: return 0.0
    b2_norm = np.linalg.norm(b2)
    cos_phi = np.clip(np.dot(m1, m2) / np.sqrt(m1_sq * m2_sq), -1.0, 1.0)
    sin_phi = np.dot(b2, np.cross(m1, m2)) / (b2_norm * np.sqrt(m1_sq * m2_sq))
    return np.arctan2(sin_phi, cos_phi)

def dihedral_energy(pos_i, pos_j, pos_k, pos_l, box_dim, K, n, phi0):
    phi = get_dihedral(pos_i, pos_j, pos_k, pos_l, box_dim)
    return K * (1.0 - np.cos(n * phi - phi0))


def dihedral_forces(pos_i, pos_j, pos_k, pos_l, box_dim, K, n, phi0):
    """Conservative reference forces for the cosine dihedral.

    A central finite difference is deliberately used here.  Preprocessing is
    offline, and this avoids silently subtracting an analytic expression whose
    sign/index convention differs from the ESPResSo dihedral definition.
    """
    positions = np.array([pos_i, pos_j, pos_k, pos_l], dtype=float)
    forces = np.zeros_like(positions)
    eps = 1.0e-6

    def energy(coords):
        return dihedral_energy(*coords, box_dim, K, n, phi0)

    for atom in range(4):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += eps
            minus[atom, axis] -= eps
            forces[atom, axis] = -(energy(plus) - energy(minus)) / (2.0 * eps)

    # Remove round-off drift while preserving internal forces.
    forces -= forces.mean(axis=0, keepdims=True)
    return tuple(forces)


def get_atom_forces_kjmol_nm(atoms, ts):
    """Return atom forces in kJ mol^-1 nm^-1 or fail loudly."""
    if getattr(ts, "has_forces", None) is False:
        raise RuntimeError(
            "The trajectory frame does not contain forces. Use a force-bearing "
            "TRR for force matching (check the GROMACS force-output settings)."
        )

    try:
        atom_forces = np.asarray(atoms.forces, dtype=np.float64)
    except (NoDataError, AttributeError):
        try:
            ts_forces = np.asarray(ts.forces, dtype=np.float64)
            atom_forces = ts_forces[np.asarray(atoms.indices, dtype=np.int64)]
        except (NoDataError, AttributeError, TypeError, IndexError) as exc:
            raise RuntimeError(
                "Reference forces are unavailable from the trajectory; "
                "cannot build force-matching targets."
            ) from exc

    if atom_forces.shape != (len(atoms), 3):
        raise RuntimeError(
            f"Unexpected atomic-force shape {atom_forces.shape}; "
            f"expected ({len(atoms)}, 3)."
        )
    if not np.all(np.isfinite(atom_forces)):
        raise RuntimeError("Non-finite reference forces found in trajectory.")
    return atom_forces * 10.0


# =====================================================================
# 2. PASS 1: DIRECT BOLTZMANN INVERSION
# =====================================================================
print(f"[INFO] Pass 1: Direct Boltzmann Inversion su {len(u.trajectory)} frame a T={TEMPERATURE} K...")

bond_distances = {tuple(b): [] for b in BONDS if isinstance(b, list)}
for b_idx, b in enumerate(BONDS):
    if isinstance(b, dict): bond_distances[f"dict_{b_idx}"] = []
angle_values = {f"dict_{idx}": [] for idx in range(len(ANGLES))}
dihedral_values = {f"dict_{idx}": [] for idx in range(len(DIHEDRALS))}
rigid_bodies_info = {}
principal_axes_lab_by_resname = {}

all_pairwise_distances = {} # Per statistical WCA: (type_i, type_j) -> sampled distances
# Streaming exact minimum over every physical nonbonded distance seen in Pass 1.
# Keep this separate from all_pairwise_distances because the latter is deliberately
# subsampled (<=1000 distances per pair per frame) to keep quantile/histogram memory
# bounded.  The physical-support guard must never use that sampled minimum.
exact_pair_min = {}

# Helper to map entire trajectory frames into memory for CG centers
# Assuming each molecule is a CG site for the non-bonded/bonded priors as per convert_gro2bin
# Wait, if mapping is per molecule (rigid body):
cg_centers_history = []
cg_forces_history = []
cg_torques_history = []
box_dim_history = []
sites_data_history = []
mol_site_indices = {}
wca_direct_mol_pairs = set()
wca_direct_site_pairs = set()
wca_one_three_mol_pairs = set()
wca_direct_mol_matrix = None
wca_one_three_mol_matrix = None

for ts_idx, ts in enumerate(u.trajectory):
    if ts_idx % 100 == 0:
        print(f"\r[INFO] Inversione Boltzmann: Frame {ts_idx}/{len(u.trajectory)}", end="")
    
    box_dim = ts.dimensions[:3] / 10.0 # nm
    box_dim_history.append(box_dim)
    
    frame_centers = []
    frame_forces = []
    frame_torques = []
    frame_sites = []
    
    valid_residues = [res for res in u.residues if res.resname in mapping_by_resname]
    if ts_idx == 0:
        mol_resnames = [res.resname for res in valid_residues]
        wca_direct_mol_pairs, wca_one_three_mol_pairs = build_wca_topology_exclusions(
            WCA_TOPOLOGY_BONDS, WCA_TOPOLOGY_ANGLES, len(mol_resnames)
        )
        wca_direct_site_pairs = build_wca_direct_site_exclusions(
            WCA_TOPOLOGY_BONDS, len(mol_resnames)
        )
        wca_direct_mol_matrix = np.zeros(
            (len(mol_resnames), len(mol_resnames)), dtype=bool
        )
        wca_one_three_mol_matrix = np.zeros_like(wca_direct_mol_matrix)
        for mi, mj in wca_direct_mol_pairs:
            wca_direct_mol_matrix[mi, mj] = True
            wca_direct_mol_matrix[mj, mi] = True
        for mi, mj in wca_one_three_mol_pairs:
            wca_one_three_mol_matrix[mi, mj] = True
            wca_one_three_mol_matrix[mj, mi] = True
        print(
            f"\n[INFO] WCA topology policy v3: {len(wca_direct_mol_pairs)} 1-2 molecule pairs "
            f"with {len(wca_direct_site_pairs)} explicitly bonded site-pair exclusions; "
            f"{len(wca_one_three_mol_pairs)} 1-3 all-sites exclusions."
        )
        
    for mol_id, residue in enumerate(valid_residues):
        resname = residue.resname
        current_mapping = mapping_by_resname[resname]
        
        atoms = residue.atoms
        positions_nm = atoms.positions / 10.0
        forces_nm = get_atom_forces_kjmol_nm(atoms, ts)
            
        try:
            masses = atoms.masses
        except:
            masses = np.array([get_mass(name) for name in atoms.names])
            
        unwrapped_pos = get_unwrapped_positions(positions_nm, box_dim)
        center = compute_com(unwrapped_pos, masses)
        
        total_force = np.sum(forces_nm, axis=0)
        r_vec = unwrapped_pos - center
        total_torque = np.sum(np.cross(r_vec, forces_nm), axis=0)
        
        if ts_idx == 0 and resname not in rigid_bodies_info:
            total_mass = float(np.sum(masses))
            I_tensor = compute_inertia_tensor(unwrapped_pos, masses, center)
            eigvals, principal_axes = diagonalize_inertia_tensor(I_tensor)
            principal_axes_lab_by_resname[resname] = principal_axes
            rb_config = RIGID_BODIES_CONFIG.get(resname, {})
            rigid_bodies_info[resname] = {
                "schema_version": 2,
                "body_frame": "principal_axes",
                "auto_align_sites": bool(rb_config.get("auto_align_sites", True)),
                "mass_amu": float(total_mass),
                "inertia_amu_nm2": [float(v) for v in eigvals],
                "sites": {}
            }
        
        frame_centers.append(center)
        frame_forces.append(total_force)
        frame_torques.append(total_torque)
        
        if ts_idx == 0:
            mol_site_indices[mol_id] = {}
            for site_name, atom_names in current_mapping.items():
                if atom_names == ["*"]:
                    mol_site_indices[mol_id][site_name] = list(range(len(atoms)))
                else:
                    sel = "name " + " ".join(atom_names)
                    site_atoms = atoms.select_atoms(sel)
                    local_by_global = {
                        int(global_index): local_index
                        for local_index, global_index in enumerate(atoms.indices)
                    }
                    mol_site_indices[mol_id][site_name] = [
                        local_by_global[int(global_index)]
                        for global_index in site_atoms.indices
                    ]
        
        sites_for_mol = []
        for site_name, atom_names in current_mapping.items():
            indices = mol_site_indices[mol_id][site_name]
            if len(indices) == 0: continue
            
            if MAPPING_METHOD == "COM":
                site_pos = compute_com(unwrapped_pos[indices], masses[indices])
            elif MAPPING_METHOD == "COG":
                site_pos = np.mean(unwrapped_pos[indices], axis=0)
            elif MAPPING_METHOD == "ATOM":
                site_pos = unwrapped_pos[indices[0]]
            
            site_type = site_types[site_name]
            if ts_idx == 0:
                if resname in rigid_bodies_info and site_name not in rigid_bodies_info[resname].get("sites", {}):
                    relative_pos_nm = site_pos - center
                    rigid_bodies_info[resname]["sites"][site_name] = {
                        "type": int(site_type),
                        "relative_pos_nm": [float(v) for v in relative_pos_nm],
                    }
            sites_for_mol.append((site_type, site_pos))
            
        frame_sites.append(sites_for_mol)
        
    cg_centers_history.append(frame_centers)
    cg_forces_history.append(frame_forces)
    cg_torques_history.append(frame_torques)
    sites_data_history.append(frame_sites)
    
    # WCA distance statistics are intentionally NOT collected here.
    # At this stage multi-site bodies still contain instantaneous internal distortions.
    # The WCA prior used in Pass 2/runtime acts on the averaged rigid geometry,
    # so WCA statistics are collected only after the Kabsch rigid-body average.

    # Bond/angle/dihedral statistics are deliberately NOT collected here.
    # Multi-site bodies still contain instantaneous internal distortions at
    # this point.  They are collected after the ideal rigid geometry has been
    # finalized, using exactly the same reconstructed sites as Pass 2/runtime.

# Refuse to continue if force mapping produced an all-zero trajectory.
_reference_force_max = max(
    (float(np.max(np.abs(np.asarray(frame, dtype=float)))) for frame in cg_forces_history),
    default=0.0,
)
if _reference_force_max <= 1.0e-12:
    raise RuntimeError(
        "All mapped reference forces are zero. The trajectory likely does not "
        "contain usable force records; refusing to build F_ref - F_prior targets."
    )

# A one-site CG molecule has no rotational degree of freedom in the current
# ESPResSo runtime.  Therefore its sole interaction site must coincide with
# the translational reference particle.  A non-zero offset would be frozen in
# the laboratory frame at runtime and would not match the mapped trajectory.
for _resname, _info in rigid_bodies_info.items():
    _sites = _info.get("sites", {})
    if len(_sites) == 1:
        _site_name, _site_info = next(iter(_sites.items()))
        _offset = np.asarray(_site_info["relative_pos_nm"], dtype=float)
        _offset_norm = float(np.linalg.norm(_offset))
        if _offset_norm > 1.0e-6:
            raise ValueError(
                f"Single-site CG body {_resname}/{_site_name} has a non-zero "
                f"COM offset ({_offset_norm:.6g} nm), but one-site bodies have no "
                "rotational DOF at runtime. Map the site to the residue COM "
                "(for COM mapping, use ['*']) or generalize the runtime DOFs."
            )

print("\n[INFO] Esecuzione allineamento Kabsch per mediare le geometrie dei corpi rigidi...")

for resname, info in rigid_bodies_info.items():
    if "sites" not in info or len(info["sites"]) < 2:
        continue
        
    site_names = list(info["sites"].keys())
    N_sites = len(site_names)
    
    # Raccogli tutti gli snapshot delle coordinate relative
    snapshots = []
    for frame_idx, frame_sites in enumerate(sites_data_history):
        frame_centers = cg_centers_history[frame_idx]
        for mol_id, rname in enumerate(mol_resnames):
            if rname == resname:
                center = frame_centers[mol_id]
                site_positions = [site[1] for site in frame_sites[mol_id]]
                if len(site_positions) == N_sites:
                    rel_pos = np.array(site_positions) - center
                    snapshots.append(rel_pos)
                    
    snapshots = np.array(snapshots)
    if len(snapshots) == 0: continue
    
    if info.get("auto_align_sites", True):
        # Iterative Kabsch average, anchored to the first observed orientation.
        ref = snapshots[0].copy()
        for iteration in range(3):
            aligned_snapshots = []
            for snap in snapshots:
                R = kabsch_align(snap, ref)
                aligned_snapshots.append((R @ snap.T).T)
            ref = np.mean(aligned_snapshots, axis=0)
        source_description = f"mediati {len(snapshots)} snapshot"
    else:
        rb_config = RIGID_BODIES_CONFIG.get(resname, {})
        configured_sites = rb_config.get("sites", {})
        missing = [name for name in site_names if name not in configured_sites]
        if missing:
            raise ValueError(
                f"Rigid body {resname} has auto_align_sites=False but is missing manual sites: {missing}"
            )
        manual = np.asarray(
            [configured_sites[name]["relative_pos_nm"] for name in site_names], dtype=float
        )
        if manual.shape != snapshots[0].shape:
            raise ValueError(f"Invalid manual rigid-body geometry for {resname}: {manual.shape}")
        manual_to_lab = kabsch_align(manual, snapshots[0])
        ref = (manual_to_lab @ manual.T).T
        source_description = "usata geometria manuale"
        
    # Express the ideal geometry in the principal-axis body frame.
    # ESPResSo stores rinertia as three principal moments, so the virtual-site
    # offsets must use the same body-fixed axes.
    principal_axes = principal_axes_lab_by_resname.get(resname)
    if principal_axes is None:
        raise RuntimeError(f"Missing principal axes for rigid body {resname}")
    ref_body = (principal_axes.T @ ref.T).T
    for i, sname in enumerate(site_names):
        info["sites"][sname]["relative_pos_nm"] = [float(v) for v in ref_body[i]]
    print(
        f"  -> {resname}: {source_description}; "
        "siti salvati nel frame degli assi principali."
    )

# Validate site-level 1-2 exclusions against the actual mapped CG layout before
# collecting WCA statistics.  A bad site index must never silently turn into an
# active WCA pair.
for _mi, _mj, _si, _sj in sorted(wca_direct_site_pairs):
    if _si >= len(sites_data_history[0][_mi]) or _sj >= len(sites_data_history[0][_mj]):
        raise RuntimeError(
            "WCA bonded-site exclusion references an out-of-range CG site: "
            f"{_mi}:{_si} <-> {_mj}:{_sj}"
        )

# Collect every geometry-derived prior statistic on the same effective CG
# geometry used by Pass 2/runtime.  This is essential for rigid multi-site
# bodies: fitting on raw mapped sites and applying on ideal rigid sites would
# define different Hamiltonians.
print("\n[INFO] Raccolta statistiche bonded sulla geometria CG effettiva usata nel Pass 2...")
if WCA_SIGMA == "auto":
    print("[INFO] Raccolta simultanea delle distanze WCA nonbonded...")
    all_pairwise_distances = {}
    exact_pair_min = {}

for ts_idx, raw_frame_sites in enumerate(sites_data_history):
    if ts_idx % 100 == 0:
        print(
            f"\r[INFO] Prior geometry: Frame {ts_idx}/{len(sites_data_history)}",
            end="",
        )

    frame_centers = cg_centers_history[ts_idx]
    box_dim = box_dim_history[ts_idx]
    rigid_frame_sites = reconstruct_rigid_sites_for_frame(
        raw_frame_sites,
        frame_centers,
        mol_resnames,
        rigid_bodies_info,
    )

    for b_idx, b in enumerate(BONDS):
        if isinstance(b, list) and len(b) >= 2:
            i, j = b[0], b[1]
            site_i = b[2] if len(b) > 2 else -1
            site_j = b[3] if len(b) > 3 else -1
            b_key = tuple(b)
        elif isinstance(b, dict) and b.get("r0") == "auto":
            i, j = b["mol_i"], b["mol_j"]
            site_i = b.get("site_i", -1)
            site_j = b.get("site_j", -1)
            b_key = f"dict_{b_idx}"
        else:
            continue

        if i < len(frame_centers) and j < len(frame_centers):
            pos_i = resolve_site_position(frame_centers, rigid_frame_sites, i, site_i)
            pos_j = resolve_site_position(frame_centers, rigid_frame_sites, j, site_j)
            r = np.linalg.norm(mic_vector(pos_i, pos_j, box_dim))
            bond_distances[b_key].append(r)

    for idx, a in enumerate(ANGLES):
        i, j, k = a["mol_i"], a["mol_j"], a["mol_k"]
        if i < len(frame_centers) and j < len(frame_centers) and k < len(frame_centers):
            pos_i = resolve_site_position(frame_centers, rigid_frame_sites, i, a.get("site_i", -1))
            pos_j = resolve_site_position(frame_centers, rigid_frame_sites, j, a.get("site_j", -1))
            pos_k = resolve_site_position(frame_centers, rigid_frame_sites, k, a.get("site_k", -1))
            angle_values[f"dict_{idx}"].append(get_angle(pos_i, pos_j, pos_k, box_dim))

    for idx, d in enumerate(DIHEDRALS):
        i, j, k, l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
        if i < len(frame_centers) and j < len(frame_centers) and k < len(frame_centers) and l < len(frame_centers):
            pos_i = resolve_site_position(frame_centers, rigid_frame_sites, i, d.get("site_i", -1))
            pos_j = resolve_site_position(frame_centers, rigid_frame_sites, j, d.get("site_j", -1))
            pos_k = resolve_site_position(frame_centers, rigid_frame_sites, k, d.get("site_k", -1))
            pos_l = resolve_site_position(frame_centers, rigid_frame_sites, l, d.get("site_l", -1))
            dihedral_values[f"dict_{idx}"].append(get_dihedral(pos_i, pos_j, pos_k, pos_l, box_dim))

    if WCA_SIGMA == "auto":
        accumulate_wca_distance_samples(
            rigid_frame_sites,
            box_dim,
            wca_direct_site_pairs,
            wca_one_three_mol_matrix,
            all_pairwise_distances,
            exact_pair_min,
        )

print()
if WCA_SIGMA == "auto" and not all_pairwise_distances:
    raise RuntimeError(
        "No nonbonded WCA distance samples were collected on the effective CG geometry."
    )

if WCA_SIGMA == "auto":
    print("\n[INFO] Calcolo parametri WCA geometrici con regolarizzazione gerarchica...")
    import scipy.optimize
    from scipy.ndimage import gaussian_filter1d

    # Trova tutti i tipi unici
    all_types = set()
    for (t1, t2) in all_pairwise_distances.keys():
        all_types.add(t1)
        all_types.add(t2)
    all_types = sorted(list(all_types))
    n_types = len(all_types)
    type_to_idx = {t: i for i, t in enumerate(all_types)}

    empirical_QLOW = {}
    empirical_min = {}
    pair_counts = {}

    for pair, dists in all_pairwise_distances.items():
        if len(dists) > 0:
            empirical_QLOW[pair] = np.percentile(dists, WCA_QUANTILE_PERCENT)
            # The low quantile is intentionally estimated from the bounded sample,
            # but the hard support limit must use the exact streaming minimum.
            if pair not in exact_pair_min:
                raise RuntimeError(
                    f"Missing exact streaming minimum for WCA type pair {pair}."
                )
            empirical_min[pair] = float(exact_pair_min[pair])
            pair_counts[pair] = len(dists)
        
    # Ottimizzazione globale dei raggi di base R_i
    def cost_func_R(R):
        loss = 0.0
        N0 = 1000.0
        for (t1, t2), q_low in empirical_QLOW.items():
            N = pair_counts[(t1, t2)]
            weight = N / (N + N0)
            r_pred = R[type_to_idx[t1]] + R[type_to_idx[t2]]
            loss += weight * (r_pred - q_low)**2
        return loss
    
    R_init = np.ones(n_types) * 0.15
    bounds_R = [(0.05, 0.5) for _ in range(n_types)]
    res_R = scipy.optimize.minimize(cost_func_R, R_init, bounds=bounds_R)
    R_opt = res_R.x

    print("  Raggi base R_i ottimizzati:")
    for t, r in zip(all_types, R_opt):
        print(f"    Tipo {t}: {r:.4f} nm")
    
    KB_T = 2.494 # kJ/mol at 300K
    wca_prior_dict = {}

    for (t1, t2), dists in all_pairwise_distances.items():
        r_base = R_opt[type_to_idx[t1]] + R_opt[type_to_idx[t2]]

        # Bug 10: Use gaussian_filter1d for robust extraction
        if len(dists) > 0:
            hist, bin_edges = np.histogram(dists, bins=50)
            smoothed_hist = gaussian_filter1d(hist, sigma=1.0)
            valid_idx = np.where(smoothed_hist > 0)[0]
            if len(valid_idx) > 0:
                r_emp_min = bin_edges[valid_idx[0]]
            else:
                r_emp_min = np.min(dists)
        else:
            r_emp_min = R_opt[type_to_idx[t1]] + R_opt[type_to_idx[t2]]
        
        q_low = empirical_QLOW.get((t1, t2), r_base)
        N_samples = len(dists)
        N0 = 1000.0
        alpha = N_samples / (N_samples + N0)
        
        r_base = R_opt[type_to_idx[t1]] + R_opt[type_to_idx[t2]]
        r_c = alpha * q_low + (1.0 - alpha) * r_base
        
        # Do not let the WCA onset invade more of the physical distribution
        # than the selected low percentile configured by wca_quantile_percent.
        r_c = min(r_c, q_low)

        # Hard physical-support constraint.  The deep-core guard must lie
        # strictly below the shortest *physical nonbonded* distance observed
        # for this type pair.  Otherwise the analytic WCA prior can generate
        # forces far larger than F_reference on configurations that actually
        # belong to the training distribution.
        r_true_min = float(empirical_min.get((t1, t2), np.min(dists)))
        r_c_before_physical_guard = float(r_c)
        r_c_physical_limit = (
            WCA_PHYSICAL_GUARD_MARGIN * r_true_min / WCA_GUARD_FRACTION
        )
        r_c = min(r_c, r_c_physical_limit)

        # Keep the onset at r_c but make the wall gradual near the cutoff.
        # epsilon is chosen so that U_WCA(guard_fraction * r_c) = guard_kbt*kBT.
        r_guard = WCA_GUARD_FRACTION * r_c
        sigma = r_c / (2.0**(1.0/6.0))
        term = (sigma / r_guard)**6
        u_factor = 4.0 * (term**2 - term) + 1.0
        kT = R_KJ_MOL_K * TEMPERATURE
        epsilon = WCA_GUARD_KBT * kT / u_factor
        
        wca_prior_dict[f"{t1}_{t2}"] = {
            "type_i": int(t1),
            "type_j": int(t2),
            "sigma_nm": float(sigma),
            "epsilon_kjmol": float(epsilon),
            "cutoff_nm": float(r_c),
            # True minimum over the physical rigid nonbonded samples.  Keep the old
            # histogram-derived estimate only as diagnostic metadata.
            "empirical_min": float(r_true_min),
            "empirical_min_source": "exact_streaming_rigid_nonbonded",
            "histogram_min_nm": float(r_emp_min),
            "quantile_percent": float(WCA_QUANTILE_PERCENT),
            "q_low_nm": float(q_low),
            "r_guard_nm": float(r_guard),
            "guard_fraction": float(WCA_GUARD_FRACTION),
            "guard_kbt": float(WCA_GUARD_KBT),
            "physical_guard_margin": float(WCA_PHYSICAL_GUARD_MARGIN),
            "cutoff_before_physical_guard_nm": float(r_c_before_physical_guard),
            "physical_cutoff_limit_nm": float(r_c_physical_limit)
        }
        
    print(f"[INFO] Elaborazione parametri WCA completata ({len(wca_prior_dict)} coppie)")
    
    # Print WCA invasion table
    print("\n[INFO] Tabella di Invasione WCA (statistiche sui dati fisici):")
    print(
        f"{'pair':<8} {'r_c':<8} {'% < r_c':<15} {'% < r_guard':<15} "
        f"{'r_guard/r_min':<15}"
    )
    physical_guard_violations = []
    for pair_key, wca_info in wca_prior_dict.items():
        t1 = wca_info["type_i"]
        t2 = wca_info["type_j"]
        r_c = wca_info["cutoff_nm"]
        r_guard = WCA_GUARD_FRACTION * r_c
        
        # Retrieve sampled distances for invasion percentages.  The hard safety
        # comparison below uses the exact streaming minimum saved in the prior.
        # all_pairwise_distances keys are string tuples: (str(t1), str(t2))
        dists = all_pairwise_distances.get((str(t1), str(t2)), [])
        if not dists:
            # try symmetric key
            dists = all_pairwise_distances.get((str(t2), str(t1)), [])
            
        if len(dists) > 0:
            dists_arr = np.array(dists)
            physical_min = float(wca_info["empirical_min"])
            frac_rc = 100.0 * np.sum(dists_arr < r_c) / len(dists_arr)
            frac_guard = 100.0 * np.sum(dists_arr < r_guard) / len(dists_arr)
            guard_ratio = r_guard / physical_min
            print(
                f"{t1}-{t2:<6} {r_c:<8.3f} {frac_rc:<15.2f} "
                f"{frac_guard:<15.2f} {guard_ratio:<15.4f}"
            )
            if r_guard > physical_min + 1.0e-12:
                physical_guard_violations.append(
                    (t1, t2, r_guard, physical_min, guard_ratio)
                )
        else:
            print(
                f"{t1}-{t2:<6} {r_c:<8.3f} {'N/A':<15} {'N/A':<15} {'N/A':<15}"
            )

    if physical_guard_violations:
        details = "; ".join(
            f"{t1}-{t2}: r_guard={rg:.6f} > r_min={rmin:.6f} "
            f"(ratio={ratio:.6f})"
            for t1, t2, rg, rmin, ratio in physical_guard_violations
        )
        raise RuntimeError(
            "WCA physical-support safety check failed: at least one deep-core "
            f"guard invades the physical nonbonded trajectory. {details}"
        )
    print(
        "[INFO] WCA physical-support safety check passed: "
        f"r_guard <= {WCA_PHYSICAL_GUARD_MARGIN:.3f} * rigid r_min for every type pair."
    )
            
    print(f"[INFO] Elaborazione parametri WCA completata ({len(wca_prior_dict)} coppie)")

print("\n[INFO] Pass 1 completato. Calcolo parametri Harmonic Priors...")

if args.priors:
    print(f"[INFO] Trovato flag --priors. Salto inferenza statistica e carico i prior esatti da: {args.priors}")
    derived_priors = _preloaded_priors
    derived_priors["morse_type_pairs"] = normalize_morse_type_pairs(
        copy.deepcopy(derived_priors.get("morse_type_pairs", [])), site_types.values()
    )
    exclusion_meta = derived_priors.get("wca_exclusions", {})
    if not (
        exclusion_meta.get("policy_version") == 3
        and exclusion_meta.get("exclude_12") is True
        and exclusion_meta.get("exclude_13") is True
        and exclusion_meta.get("direct_scope") == "bonded_site_pairs_only"
        and exclusion_meta.get("one_three_scope") == "molecule_pair_all_sites"
        and exclusion_meta.get("pair_source") == "explicit_topology_pairs_v3"
        and isinstance(exclusion_meta.get("direct_pairs"), list)
        and isinstance(exclusion_meta.get("direct_site_pairs"), list)
        and isinstance(exclusion_meta.get("one_three_pairs"), list)
    ):
        raise RuntimeError(
            "The supplied priors do not use WCA policy v3 "
            "(1-2 bonded-site-only, 1-3 all-sites). Rebuild cg_priors.json from "
            "the physical trajectory before reusing --priors."
        )
    _meta_direct = {tuple(sorted(map(int, pair))) for pair in exclusion_meta["direct_pairs"]}
    _meta_one_three = {tuple(sorted(map(int, pair))) for pair in exclusion_meta["one_three_pairs"]}
    _meta_direct_sites = set()
    for _raw in exclusion_meta["direct_site_pairs"]:
        if not isinstance(_raw, (list, tuple)) or len(_raw) != 4:
            raise RuntimeError(f"Invalid direct_site_pairs entry in supplied priors: {_raw!r}")
        _mi, _mj, _si, _sj = map(int, _raw)
        _meta_direct_sites.add(
            (_mi, _mj, _si, _sj) if _mi < _mj else (_mj, _mi, _sj, _si)
        )
    if (
        _meta_direct != wca_direct_mol_pairs
        or _meta_one_three != wca_one_three_mol_pairs
        or _meta_direct_sites != wca_direct_site_pairs
    ):
        raise RuntimeError(
            "The supplied wca_exclusions metadata disagrees with its bonded topology. "
            "Rebuild cg_priors.json instead of reusing inconsistent priors."
        )
    wca_info = derived_priors.get("wca", {})
    WCA_SIGMA_VAL = float(wca_info.get("sigma", 0.0))
    # Override solo se epsilon è specificato nei priors, altrimenti tieni args
    if "epsilon" in wca_info:
        WCA_EPSILON = float(wca_info["epsilon"])
    WCA_OVERRIDES = wca_info.get("overrides", {})
else:
    derived_priors = None

if derived_priors is None:
    # Calcolo k e r0 statistico (Boltzmann Inversion)
    if WCA_SIGMA == "auto":
        WCA_SIGMA_VAL = 0.25 # Valore di fallback sicuro
    else:
        WCA_SIGMA_VAL = float(WCA_SIGMA)

    derived_priors = {
        "bonds": [],
        "morse_type_pairs": copy.deepcopy(MORSE_TYPE_PAIRS),
        "wca": {
            "sigma": WCA_SIGMA_VAL,
            "epsilon": WCA_EPSILON if WCA_EPSILON > 0 else 1000.0,
            "overrides": WCA_OVERRIDES,
        },
    }
    
    for b, dists in bond_distances.items():
        if isinstance(b, str): continue
        if len(dists) == 0: continue
        r_arr = np.array(dists)
        mean_r = np.mean(r_arr)
        var_r = np.var(r_arr)
        
        k_stat = 1.0 / (BETA * var_r) if var_r > 1e-12 else 0.0
        
        derived_priors["bonds"].append({
            "mol_i": b[0],
            "mol_j": b[1],
            "site_i": b[2] if len(b) > 2 else -1,
            "site_j": b[3] if len(b) > 3 else -1,
            "type": "harmonic",
            "k": float(k_stat),
            "r0": float(mean_r)
        })
        print(f"  -> Legame (Boltzmann) mol {b[0]} - mol {b[1]}: r0 = {mean_r:.4f} nm, k = {k_stat:.2f} kJ/mol/nm^2")

    # Pool statistics for named groups
    shared_bond_distances = {}
    for b_idx, b in enumerate(BONDS):
        if isinstance(b, dict) and "name" in b:
            b_name = b["name"]
            if b_name not in shared_bond_distances: shared_bond_distances[b_name] = []
            shared_bond_distances[b_name].extend(bond_distances[f"dict_{b_idx}"])

    # Aggiungi eventuali legami espliciti (FENE, Morse, o Harmonic custom) forniti come dizionari
    for b_idx, b in enumerate(BONDS):
        if isinstance(b, dict):
            b_type = b.get("type", "unknown")
        if b.get("r0") == "auto":
            vals = shared_bond_distances[b["name"]] if "name" in b else bond_distances[f"dict_{b_idx}"]
            if len(vals) > 0:
                b["r0"] = float(np.mean(vals))
                if b.get("k") == "auto":
                    var_r = np.var(vals)
                    b["k"] = float(1.0 / (BETA * var_r)) if var_r > 1e-12 else 0.0
                print(f"  -> Auto-calcolato r0 per legame esplicito ({b_type}) {b['mol_i']}-{b['mol_j']}: r0 = {b['r0']:.4f} nm")
        
        derived_priors["bonds"].append(b)
        print(f"  -> Legame esplicito ({b.get('type', b_type)}) aggiunto: {b['mol_i']}-{b['mol_j']}")

    shared_angle_values = {}
    for idx, a in enumerate(ANGLES):
        if "name" in a:
            a_name = a["name"]
            if a_name not in shared_angle_values: shared_angle_values[a_name] = []
            shared_angle_values[a_name].extend(angle_values[f"dict_{idx}"])

    derived_priors["angles"] = []
    for idx, a in enumerate(ANGLES):
        vals = shared_angle_values[a["name"]] if "name" in a else angle_values[f"dict_{idx}"]
        a_type = a.get("type", "harmonic")
            
        if len(vals) > 0:
            if a.get("theta0") == "auto":
                a["theta0"] = float(np.mean(vals))
            if a.get("k") == "auto":
                var_theta = np.var(vals)
                a["k"] = float(1.0 / (BETA * var_theta)) if var_theta > 1e-12 else 0.0
        
        derived_priors["angles"].append(a)
        print(f"  -> Angle aggiunto: {a['mol_i']}-{a['mol_j']}-{a['mol_k']} (theta0={a.get('theta0', 0):.4f} rad, k={a.get('k', 0):.2f})")

    shared_dihedral_values = {}
    for idx, d in enumerate(DIHEDRALS):
        if "name" in d:
            d_name = d["name"]
            if d_name not in shared_dihedral_values: shared_dihedral_values[d_name] = []
            shared_dihedral_values[d_name].extend(dihedral_values[f"dict_{idx}"])

    derived_priors["dihedrals"] = []
    for idx, d in enumerate(DIHEDRALS):
        vals = shared_dihedral_values[d["name"]] if "name" in d else dihedral_values[f"dict_{idx}"]
        d_type = d.get("type", "cosine")
            
        if len(vals) > 0:
            if d.get("phi0") == "auto":
                sin_vals = np.sin(vals)
                cos_vals = np.cos(vals)
                d["phi0"] = float(np.arctan2(np.mean(sin_vals), np.mean(cos_vals)))
            if d.get("k") == "auto":
                var_phi = np.var(vals)
                d["k"] = float(1.0 / (BETA * var_phi)) if var_phi > 1e-12 else 0.0
        
        derived_priors["dihedrals"].append(d)
        print(f"  -> Dihedral aggiunto: {d['mol_i']}-{d['mol_j']}-{d['mol_k']}-{d['mol_l']} (phi0={d.get('phi0', 0):.4f} rad, k={d.get('k', 0):.2f})")

    if WCA_SIGMA == "auto":
        derived_priors["wca_pairs"] = wca_prior_dict

    derived_priors["wca_exclusions"] = {
        "policy_version": 3,
        "exclude_12": True,
        "exclude_13": True,
        "direct_scope": "bonded_site_pairs_only",
        "one_three_scope": "molecule_pair_all_sites",
        "pair_source": "explicit_topology_pairs_v3",
        "direct_pairs": [list(pair) for pair in sorted(wca_direct_mol_pairs)],
        "direct_site_pairs": [list(record) for record in sorted(wca_direct_site_pairs)],
        "one_three_pairs": [list(pair) for pair in sorted(wca_one_three_mol_pairs)],
        "direct_pair_count": len(wca_direct_mol_pairs),
        "direct_site_pair_count": len(wca_direct_site_pairs),
        "one_three_pair_count": len(wca_one_three_mol_pairs),
    }
        
    # Save priors for simulation
    with open(priors_output_path, "w") as pf:
        json.dump(derived_priors, pf, indent=4)
    print(f"[INFO] Salvati prior CG in {priors_output_path}")

# I WCA utilizzati nel Pass 2 vengono SEMPRE dai priors finali,
# sia che siano stati appena calcolati sia che siano stati caricati
# tramite --priors.
wca_prior_dict = derived_priors.get("wca_pairs", {})

if not wca_prior_dict:
    raise RuntimeError("No pair-specific WCA priors found in derived_priors['wca_pairs']")

_final_direct, _final_one_three = build_wca_topology_exclusions(
    derived_priors.get("bonds", []), derived_priors.get("angles", []), len(mol_resnames)
)
_final_direct_sites = build_wca_direct_site_exclusions(
    derived_priors.get("bonds", []), len(mol_resnames)
)
if (
    _final_direct != wca_direct_mol_pairs
    or _final_direct_sites != wca_direct_site_pairs
    or _final_one_three != wca_one_three_mol_pairs
):
    raise RuntimeError(
        "WCA topology exclusions derived in Pass 1 do not match the final bonded priors."
    )


# =====================================================================
# 3. PASS 2: SOTTRAZIONE PRIOR E SCRITTURA BINARIO
# =====================================================================
print(f"[INFO] Pass 2: Sottrazione forze prior e generazione dataset {args.output}...")

out_dir = os.path.dirname(args.output)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

# Load explicit bonded tables once. Relative paths are interpreted relative to
# the priors JSON, matching the runtime contract. When a topology config itself
# contains a tabulated entry, use that config as the reference path.
table_reference_path = args.priors if args.priors else args.config
tabulated_bond_tables = {
    idx: load_tabulated_prior(b, kind="bond", priors_path=table_reference_path)
    for idx, b in enumerate(derived_priors.get("bonds", []))
    if str(b.get("type", "harmonic")).lower() == "tabulated"
}
tabulated_angle_tables = {
    idx: load_tabulated_prior(a, kind="angle", priors_path=table_reference_path)
    for idx, a in enumerate(derived_priors.get("angles", []))
    if str(a.get("type", "harmonic")).lower() == "tabulated"
}
tabulated_dihedral_tables = {
    idx: load_tabulated_prior(d, kind="dihedral", priors_path=table_reference_path)
    for idx, d in enumerate(derived_priors.get("dihedrals", []))
    if str(d.get("type", "cosine")).lower() == "tabulated"
}
if tabulated_bond_tables or tabulated_angle_tables or tabulated_dihedral_tables:
    print(
        "[INFO] Tabulated bonded subtraction: "
        f"{len(tabulated_bond_tables)} bonds, {len(tabulated_angle_tables)} angles, "
        f"{len(tabulated_dihedral_tables)} dihedrals"
    )

# Diagnostics on physical frames only.  These are intentionally collected
# before any optional --clip_forces operation so that pathological prior
# subtraction cannot be hidden by clipping.
reference_force_norms = []
residual_force_norms = []
wca_force_norms = []
wca_force_norms_by_pair = {key: [] for key in wca_prior_dict}
wca_active_distances_by_pair = {key: [] for key in wca_prior_dict}
wca_active_classes_by_pair = {key: [] for key in wca_prior_dict}

# Precompute all topology- and type-dependent WCA data once.  Site ordering is
# fixed by the CG mapping, so the same upper-triangular inter-molecular pair
# list can be reused for every frame.  This removes the O(N_sites^2) Python
# loops that previously rebuilt sigma/epsilon/cutoff matrices at each frame.
if wca_prior_dict:
    flat_mol_template = []
    flat_site_template = []
    flat_type_template = []
    for m_idx, sites in enumerate(sites_data_history[0]):
        for site_idx, (s_type, _) in enumerate(sites):
            flat_mol_template.append(m_idx)
            flat_site_template.append(site_idx)
            flat_type_template.append(int(s_type))

    flat_mol_template = np.asarray(flat_mol_template, dtype=np.int32)
    flat_site_template = np.asarray(flat_site_template, dtype=np.int32)
    flat_type_template = np.asarray(flat_type_template, dtype=np.int32)

    max_prior_type = max(
        max(int(w["type_i"]), int(w["type_j"]))
        for w in wca_prior_dict.values()
    )
    n_wca_types = max(int(np.max(flat_type_template)), max_prior_type) + 1

    wca_sigma_matrix = np.zeros((n_wca_types, n_wca_types), dtype=np.float64)
    wca_epsilon_matrix = np.zeros_like(wca_sigma_matrix)
    wca_cutoff_sq_matrix = np.zeros_like(wca_sigma_matrix)

    for pair_key, w in wca_prior_dict.items():
        ti = int(w["type_i"])
        tj = int(w["type_j"])
        sigma = float(w["sigma_nm"])
        epsilon = float(w["epsilon_kjmol"])
        cutoff_sq = float(w["cutoff_nm"]) ** 2
        wca_sigma_matrix[ti, tj] = wca_sigma_matrix[tj, ti] = sigma
        wca_epsilon_matrix[ti, tj] = wca_epsilon_matrix[tj, ti] = epsilon
        wca_cutoff_sq_matrix[ti, tj] = wca_cutoff_sq_matrix[tj, ti] = cutoff_sq

    pair_i_all, pair_j_all = np.triu_indices(flat_type_template.size, k=1)
    pair_mol_i = flat_mol_template[pair_i_all]
    pair_mol_j = flat_mol_template[pair_j_all]
    inter_mol = pair_mol_i != pair_mol_j
    topology_excluded = wca_one_three_mol_matrix[pair_mol_i, pair_mol_j].copy()
    topology_excluded |= _direct_site_excluded_mask(
        pair_i_all, pair_j_all, flat_mol_template, flat_site_template,
        wca_direct_site_pairs, len(mol_resnames)
    )
    wca_candidate = inter_mol & (~topology_excluded)
    pair_i_all = pair_i_all[wca_candidate]
    pair_j_all = pair_j_all[wca_candidate]

    pair_type_i = flat_type_template[pair_i_all]
    pair_type_j = flat_type_template[pair_j_all]
    pair_sigma_all = wca_sigma_matrix[pair_type_i, pair_type_j]
    pair_epsilon_all = wca_epsilon_matrix[pair_type_i, pair_type_j]
    pair_cutoff_sq_all = wca_cutoff_sq_matrix[pair_type_i, pair_type_j]

    configured = pair_cutoff_sq_all > 0.0
    pair_i_all = pair_i_all[configured]
    pair_j_all = pair_j_all[configured]
    pair_type_i = pair_type_i[configured]
    pair_type_j = pair_type_j[configured]
    pair_sigma_all = pair_sigma_all[configured]
    pair_epsilon_all = pair_epsilon_all[configured]
    pair_cutoff_sq_all = pair_cutoff_sq_all[configured]

    pair_type_min = np.minimum(pair_type_i, pair_type_j)
    pair_type_max = np.maximum(pair_type_i, pair_type_j)
    pair_code_all = pair_type_min * n_wca_types + pair_type_max
    pair_code_to_key = {
        int(ti) * n_wca_types + int(tj): f"{int(ti)}_{int(tj)}"
        for ti in range(n_wca_types)
        for tj in range(ti, n_wca_types)
        if wca_cutoff_sq_matrix[ti, tj] > 0.0
    }
    # Allowed 1-2 site pairs now reach the WCA kernel; 1-3 pairs must never do so.
    pair_mol_i_all = flat_mol_template[pair_i_all]
    pair_mol_j_all = flat_mol_template[pair_j_all]
    pair_topology_class_all = np.where(
        wca_direct_mol_matrix[pair_mol_i_all, pair_mol_j_all], 0, 2
    ).astype(np.int8)
    if np.any(wca_one_three_mol_matrix[pair_mol_i_all, pair_mol_j_all]):
        raise RuntimeError("A 1-3 pair leaked into the WCA candidate list.")
    if np.any(_direct_site_excluded_mask(
        pair_i_all, pair_j_all, flat_mol_template, flat_site_template,
        wca_direct_site_pairs, len(mol_resnames)
    )):
        raise RuntimeError("An explicitly bonded 1-2 site pair leaked into the WCA candidate list.")

    wca_global_cutoff_sq = float(np.max(pair_cutoff_sq_all))

    print(
        f"[INFO] WCA vectorized kernel: {flat_type_template.size} sites, "
        f"{pair_i_all.size} WCA candidate pairs (nonbonded + allowed 1-2), "
        f"global cutoff={np.sqrt(wca_global_cutoff_sq):.4f} nm"
    )

# Precompute the ordinary ESPResSo type-pair Morse selection on the same
# non-bonded site-pair mask used at runtime. Particle exclusions suppress all
# non-bonded potentials, not only WCA, so preprocessing must mirror intra-body,
# explicit bonded-site 1-2, and 1-3 exclusions for Morse type pairs as well.
morse_type_pair_priors = derived_priors.get("morse_type_pairs", [])
if morse_type_pair_priors:
    max_morse_type = max(
        max(int(item["type_i"]), int(item["type_j"]))
        for item in morse_type_pair_priors
    )
    n_morse_types = max(int(np.max(flat_type_template)), max_morse_type) + 1
    shape = (n_morse_types, n_morse_types)
    morse_D_matrix = np.zeros(shape, dtype=np.float64)
    morse_a_matrix = np.zeros(shape, dtype=np.float64)
    morse_r0_matrix = np.zeros(shape, dtype=np.float64)
    morse_switch_matrix = np.zeros(shape, dtype=np.float64)
    morse_cut_matrix = np.zeros(shape, dtype=np.float64)
    for item in morse_type_pair_priors:
        ti, tj = int(item["type_i"]), int(item["type_j"])
        for matrix, value in (
            (morse_D_matrix, float(item["D"])),
            (morse_a_matrix, float(item["a"])),
            (morse_r0_matrix, float(item["r0"])),
            (morse_switch_matrix, float(item["r_switch"])),
            (morse_cut_matrix, float(item["r_cut"])),
        ):
            matrix[ti, tj] = matrix[tj, ti] = value

    morse_pair_i, morse_pair_j = np.triu_indices(flat_type_template.size, k=1)
    morse_mol_i = flat_mol_template[morse_pair_i]
    morse_mol_j = flat_mol_template[morse_pair_j]
    morse_excluded = wca_one_three_mol_matrix[morse_mol_i, morse_mol_j].copy()
    morse_excluded |= _direct_site_excluded_mask(
        morse_pair_i, morse_pair_j, flat_mol_template, flat_site_template,
        wca_direct_site_pairs, len(mol_resnames)
    )
    morse_candidate = (morse_mol_i != morse_mol_j) & (~morse_excluded)
    morse_pair_i = morse_pair_i[morse_candidate]
    morse_pair_j = morse_pair_j[morse_candidate]
    morse_type_i = flat_type_template[morse_pair_i]
    morse_type_j = flat_type_template[morse_pair_j]
    morse_cut_all = morse_cut_matrix[morse_type_i, morse_type_j]
    configured = morse_cut_all > 0.0
    morse_pair_i = morse_pair_i[configured]
    morse_pair_j = morse_pair_j[configured]
    morse_type_i = morse_type_i[configured]
    morse_type_j = morse_type_j[configured]
    morse_D_all = morse_D_matrix[morse_type_i, morse_type_j]
    morse_a_all = morse_a_matrix[morse_type_i, morse_type_j]
    morse_r0_all = morse_r0_matrix[morse_type_i, morse_type_j]
    morse_switch_all = morse_switch_matrix[morse_type_i, morse_type_j]
    morse_cut_all = morse_cut_matrix[morse_type_i, morse_type_j]
    print(
        f"[INFO] Morse type-pair kernel: {morse_pair_i.size} selected non-bonded "
        f"site pairs, max cutoff={np.max(morse_cut_all):.4f} nm"
    )
else:
    morse_pair_i = np.empty(0, dtype=np.int64)
    morse_pair_j = np.empty(0, dtype=np.int64)
    morse_D_all = morse_a_all = morse_r0_all = np.empty(0, dtype=np.float64)
    morse_switch_all = morse_cut_all = np.empty(0, dtype=np.float64)

with open(args.output, "wb") as f:
    num_frames = len(cg_centers_history)
    f.write(struct.pack("i", num_frames))

    for ts_idx in range(num_frames):
        if ts_idx % 100 == 0:
            print(f"\r[INFO] Scrittura binario: Frame {ts_idx}/{num_frames}", end="")
    
        box_dim = box_dim_history[ts_idx]
        frame_centers = cg_centers_history[ts_idx]
        frame_forces = cg_forces_history[ts_idx]
        frame_torques = cg_torques_history[ts_idx]
        # Reconstruct exactly the same ideal rigid-site geometry used to fit WCA.
        frame_sites = reconstruct_rigid_sites_for_frame(
            sites_data_history[ts_idx],
            frame_centers,
            mol_resnames,
            rigid_bodies_info,
        )
    
        num_molecules = len(frame_centers)
        num_total_sites = sum(len(s) for s in frame_sites)
    
        f.write(struct.pack("i", num_molecules))
        f.write(struct.pack("i", num_total_sites))
        f.write(struct.pack("3f", *box_dim))
    
        # Copia indipendente per la sottrazione
        res_forces = np.copy(frame_forces)
        res_torques = np.copy(frame_torques)
        reference_force_norms.extend(np.linalg.norm(frame_forces, axis=1).tolist())
    
        # 3.1 Sottrazione WCA sui siti virtuali.
        # The pair topology and all type-dependent parameters are precomputed
        # once above.  Per frame we only evaluate MIC distances for the unique
        # inter-molecular upper-triangular pairs and vectorize the WCA kernel.
        if wca_prior_dict:
            flat_pos = np.asarray(
                [s_pos for sites in frame_sites for _, s_pos in sites],
                dtype=np.float64,
            )
            frame_centers_arr = np.asarray(frame_centers, dtype=np.float64)

            if flat_pos.shape[0] != flat_type_template.size:
                raise RuntimeError(
                    "CG site count changed between frames; cannot reuse the "
                    "precomputed WCA pair topology"
                )

            pair_diff = flat_pos[pair_i_all] - flat_pos[pair_j_all]
            pair_diff -= box_dim * np.round(pair_diff / box_dim)
            pair_dist_sq = np.einsum("ij,ij->i", pair_diff, pair_diff)

            # Cheap global cutoff first, then the exact pair-specific cutoff.
            active = (pair_dist_sq > 1.0e-6) & (pair_dist_sq < wca_global_cutoff_sq)
            active_idx = np.flatnonzero(active)
            if active_idx.size:
                active_idx = active_idx[
                    pair_dist_sq[active_idx] < pair_cutoff_sq_all[active_idx]
                ]

            if active_idx.size:
                i_idx = pair_i_all[active_idx]
                j_idx = pair_j_all[active_idx]
                mol_i = flat_mol_template[i_idx]
                mol_j = flat_mol_template[j_idx]

                r_sq = pair_dist_sq[active_idx]
                r = np.sqrt(r_sq)
                r_hat = pair_diff[active_idx] / r[:, None]
                sigma = pair_sigma_all[active_idx]
                epsilon = pair_epsilon_all[active_idx]

                sr6 = (sigma / r) ** 6
                f_scalar = 24.0 * epsilon * (2.0 * sr6 * sr6 - sr6) / r
                f_vec = f_scalar[:, None] * r_hat

                # Accumulate all site-site contributions on rigid-body COM forces.
                np.add.at(res_forces, mol_i, -f_vec)
                np.add.at(res_forces, mol_j, +f_vec)

                # Preserve the original torque convention exactly: lever arms
                # are site position minus the corresponding molecular COM.
                lever_i = flat_pos[i_idx] - frame_centers_arr[mol_i]
                lever_j = flat_pos[j_idx] - frame_centers_arr[mol_j]
                np.add.at(res_torques, mol_i, -np.cross(lever_i, f_vec))
                np.add.at(res_torques, mol_j, +np.cross(lever_j, f_vec))

                # Diagnostics are collected only for actually active WCA pairs.
                f_norm = np.linalg.norm(f_vec, axis=1)
                wca_force_norms.extend(f_norm.tolist())

                active_codes = pair_code_all[active_idx]
                active_classes = pair_topology_class_all[active_idx]
                for code in np.unique(active_codes):
                    pair_key_diag = pair_code_to_key[int(code)]
                    code_mask = active_codes == code
                    vals = f_norm[code_mask]
                    wca_force_norms_by_pair[pair_key_diag].extend(vals.tolist())
                    wca_active_distances_by_pair[pair_key_diag].extend(r[code_mask].tolist())
                    wca_active_classes_by_pair[pair_key_diag].extend(active_classes[code_mask].tolist())

        # 3.1.1 Sottrazione Morse type-pair non-bonded sui siti virtuali.
        # This uses exactly the same particle-exclusion semantics as ESPResSo.
        if morse_pair_i.size:
            morse_diff = flat_pos[morse_pair_i] - flat_pos[morse_pair_j]
            morse_diff -= box_dim * np.round(morse_diff / box_dim)
            morse_dist_sq = np.einsum("ij,ij->i", morse_diff, morse_diff)
            active = (morse_dist_sq > 1.0e-12) & (morse_dist_sq < morse_cut_all**2)
            active_idx = np.flatnonzero(active)
            if active_idx.size:
                i_idx = morse_pair_i[active_idx]
                j_idx = morse_pair_j[active_idx]
                mol_i = flat_mol_template[i_idx]
                mol_j = flat_mol_template[j_idx]
                r = np.sqrt(morse_dist_sq[active_idx])
                r_hat = morse_diff[active_idx] / r[:, None]
                f_scalar = switched_morse_radial_force_array(
                    r,
                    morse_D_all[active_idx],
                    morse_a_all[active_idx],
                    morse_r0_all[active_idx],
                    morse_switch_all[active_idx],
                    morse_cut_all[active_idx],
                )
                # morse_diff points from j to i, so signed radial force times
                # r_hat is the force on i. Subtract that explicit prior from
                # the mapped reference force and its rigid-body torque.
                f_vec = f_scalar[:, None] * r_hat
                np.add.at(res_forces, mol_i, -f_vec)
                np.add.at(res_forces, mol_j, +f_vec)
                lever_i = flat_pos[i_idx] - frame_centers_arr[mol_i]
                lever_j = flat_pos[j_idx] - frame_centers_arr[mol_j]
                np.add.at(res_torques, mol_i, -np.cross(lever_i, f_vec))
                np.add.at(res_torques, mol_j, +np.cross(lever_j, f_vec))

        # 3.2 Sottrazione Legami (con supporto per siti specifici e momento torcente)
        for b_idx, b in enumerate(derived_priors["bonds"]):
            i, j = b["mol_i"], b["mol_j"]
            site_i = b.get("site_i", -1)
            site_j = b.get("site_j", -1)

            if i >= num_molecules or j >= num_molecules: continue

            pos_i = resolve_site_position(frame_centers, frame_sites, i, site_i)
            pos_j = resolve_site_position(frame_centers, frame_sites, j, site_j)

            r_vec = mic_vector(pos_i, pos_j, box_dim)
            r = np.linalg.norm(r_vec)
            if r < 1e-6: continue

            r_hat = r_vec / r
            f_scalar = 0.0
            b_type = str(b.get("type", "harmonic")).lower()

            if b_type == "tabulated":
                f_i, f_j = tabulated_distance_forces(
                    pos_i, pos_j, box_dim, tabulated_bond_tables[b_idx]
                )
                res_forces[i] -= f_i
                res_forces[j] -= f_j
                if site_i != -1:
                    r_rel_i = mic_vector(frame_centers[i], pos_i, box_dim)
                    res_torques[i] -= np.cross(r_rel_i, f_i)
                if site_j != -1:
                    r_rel_j = mic_vector(frame_centers[j], pos_j, box_dim)
                    res_torques[j] -= np.cross(r_rel_j, f_j)
                continue

            if b_type == "harmonic":
                k, r0 = b["k"], b["r0"]
                f_scalar = - k * (r - r0)

            elif b_type == "fene":
                k, r0, r_max = b["k"], b["r0"], b["r_max"]
                diff = r - r0
                if abs(diff) >= r_max:
                    raise ValueError(
                        f"FENE bond {i}-{j} is outside its domain: "
                        f"|r-r0|={abs(diff):.6g} >= r_max={r_max:.6g}"
                    )
                f_scalar = - k * diff / (1.0 - (diff/r_max)**2)

            elif b_type == "morse":
                # Keep the force prior exactly consistent with the reversible
                # switched Morse used by the ESPResSo runtime.  The underlying
                # Morse force is unchanged up to r_switch; only the far tail is
                # smoothly brought to zero before r_cut.
                D, a, r0 = float(b["D"]), float(b["a"]), float(b["r0"])
                r_cut = float(b.get("r_cut", 15.0))
                r_switch = float(
                    b.get("r_switch", r0 + 0.75 * (r_cut - r0))
                )
                if not (r0 < r_switch < r_cut):
                    raise ValueError(
                        f"Morse bond {i}-{j} requires r0 < r_switch < r_cut; "
                        f"got r0={r0}, r_switch={r_switch}, r_cut={r_cut}"
                    )
                if r >= r_cut:
                    f_scalar = 0.0
                else:
                    exp_term = np.exp(-a * (r - r0))
                    base_energy = D * (exp_term * exp_term - 2.0 * exp_term)
                    base_force = 2.0 * D * a * exp_term * (exp_term - 1.0)
                    if r <= r_switch:
                        f_scalar = base_force
                    else:
                        width = r_cut - r_switch
                        t = (r - r_switch) / width
                        switch = 1.0 - 10.0*t**3 + 15.0*t**4 - 6.0*t**5
                        d_switch_dr = -30.0*t*t*(1.0-t)*(1.0-t) / width
                        f_scalar = switch * base_force - base_energy * d_switch_dr


            f_vec = - f_scalar * r_hat

            res_forces[i] -= f_vec
            res_forces[j] += f_vec

            if site_i != -1:
                r_rel_i = mic_vector(frame_centers[i], pos_i, box_dim)
                res_torques[i] -= np.cross(r_rel_i, f_vec)

            if site_j != -1:
                r_rel_j = mic_vector(frame_centers[j], pos_j, box_dim)
                res_torques[j] += np.cross(r_rel_j, f_vec)
            
        # 3.2.1 Sottrazione Angoli
        for a_idx, a in enumerate(derived_priors.get("angles", [])):
            a_type = str(a.get("type", "harmonic")).lower()
            if a_type in ["ibi", "dbi"]:
                continue
        
            i, j, k_idx = a["mol_i"], a["mol_j"], a["mol_k"]
            if i >= num_molecules or j >= num_molecules or k_idx >= num_molecules: continue
        
            site_i, site_j, site_k = a.get("site_i", -1), a.get("site_j", -1), a.get("site_k", -1)
            pos_i = resolve_site_position(frame_centers, frame_sites, i, site_i)
            pos_j = resolve_site_position(frame_centers, frame_sites, j, site_j)
            pos_k = resolve_site_position(frame_centers, frame_sites, k_idx, site_k)

            if a_type == "tabulated":
                f_i, f_j, f_k = tabulated_angle_forces(
                    pos_i, pos_j, pos_k, box_dim, tabulated_angle_tables[a_idx]
                )
                res_forces[i] -= f_i
                res_forces[j] -= f_j
                res_forces[k_idx] -= f_k
                if site_i != -1:
                    res_torques[i] -= np.cross(mic_vector(frame_centers[i], pos_i, box_dim), f_i)
                if site_j != -1:
                    res_torques[j] -= np.cross(mic_vector(frame_centers[j], pos_j, box_dim), f_j)
                if site_k != -1:
                    res_torques[k_idx] -= np.cross(mic_vector(frame_centers[k_idx], pos_k, box_dim), f_k)
                continue
        
            r_ji = mic_vector(pos_j, pos_i, box_dim)
            r_jk = mic_vector(pos_j, pos_k, box_dim)
            d_ji = np.linalg.norm(r_ji)
            d_jk = np.linalg.norm(r_jk)
            if d_ji < 1e-6 or d_jk < 1e-6: continue
        
            cos_theta = np.clip(np.dot(r_ji, r_jk) / (d_ji * d_jk), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            if sin_theta < 1e-6: continue
        
            if a_type == "harmonic":
                dV_dtheta = a["k"] * (theta - a["theta0"])

            else:
                dV_dtheta = 0.0
            
            grad_i_cos = r_jk / (d_ji * d_jk) - cos_theta * r_ji / (d_ji**2)
            grad_k_cos = r_ji / (d_ji * d_jk) - cos_theta * r_jk / (d_jk**2)
        
            scalar_force = dV_dtheta / sin_theta
            f_i = scalar_force * grad_i_cos
            f_k = scalar_force * grad_k_cos
            f_j = -(f_i + f_k)
        
            res_forces[i] -= f_i
            res_forces[j] -= f_j
            res_forces[k_idx] -= f_k
        
            if site_i != -1: res_torques[i] -= np.cross(mic_vector(frame_centers[i], pos_i, box_dim), f_i)
            if site_j != -1: res_torques[j] -= np.cross(mic_vector(frame_centers[j], pos_j, box_dim), f_j)
            if site_k != -1: res_torques[k_idx] -= np.cross(mic_vector(frame_centers[k_idx], pos_k, box_dim), f_k)
        
        # 3.2.2 Sottrazione Diedri
        for d_idx, d in enumerate(derived_priors.get("dihedrals", [])):
            d_type = str(d.get("type", "cosine")).lower()
            if d_type in ["ibi", "dbi"]:
                continue
            
            i, j, k_idx, l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
            if i >= num_molecules or j >= num_molecules or k_idx >= num_molecules or l >= num_molecules: continue
        
            site_i, site_j, site_k, site_l = d.get("site_i", -1), d.get("site_j", -1), d.get("site_k", -1), d.get("site_l", -1)
            pos_i = resolve_site_position(frame_centers, frame_sites, i, site_i)
            pos_j = resolve_site_position(frame_centers, frame_sites, j, site_j)
            pos_k = resolve_site_position(frame_centers, frame_sites, k_idx, site_k)
            pos_l = resolve_site_position(frame_centers, frame_sites, l, site_l)

            if d_type == "tabulated":
                f_i, f_j, f_k, f_l = tabulated_dihedral_forces(
                    pos_i, pos_j, pos_k, pos_l, box_dim,
                    tabulated_dihedral_tables[d_idx],
                )
            elif d_type == "cosine":
                f_i, f_j, f_k, f_l = dihedral_forces(
                    pos_i,
                    pos_j,
                    pos_k,
                    pos_l,
                    box_dim,
                    d["k"],
                    d.get("n", 1),
                    d["phi0"],
                )
            else:
                continue

            res_forces[i] -= f_i
            res_forces[j] -= f_j
            res_forces[k_idx] -= f_k
            res_forces[l] -= f_l
        
            if site_i != -1: res_torques[i] -= np.cross(mic_vector(frame_centers[i], pos_i, box_dim), f_i)
            if site_j != -1: res_torques[j] -= np.cross(mic_vector(frame_centers[j], pos_j, box_dim), f_j)
            if site_k != -1: res_torques[k_idx] -= np.cross(mic_vector(frame_centers[k_idx], pos_k, box_dim), f_k)
            if site_l != -1: res_torques[l] -= np.cross(mic_vector(frame_centers[l], pos_l, box_dim), f_l)
        
        
        # 3.2.2 Sottrazione Diedri
        # (Omitted lines in between handled properly)
    
        # Record the true residual target distribution before optional clipping.
        residual_force_norms.extend(np.linalg.norm(res_forces, axis=1).tolist())

        # 3.3 Optional residual-force clipping and binary output.
        # Clipping is a training-data policy, not part of the IBI definition;
        # leave it disabled unless the chosen force-matching workflow requires it.
        if args.clip_forces is not None:
            res_forces = np.clip(res_forces, -args.clip_forces, args.clip_forces)
            res_torques = np.clip(res_torques, -args.clip_forces, args.clip_forces)
    
        for mol_id in range(num_molecules):
            num_sites = len(frame_sites[mol_id])
            f.write(struct.pack("i", mol_id))
            f.write(struct.pack("i", num_sites))
            f.write(struct.pack("3f", *frame_centers[mol_id]))
            f.write(struct.pack("3f", *res_forces[mol_id]))
            f.write(struct.pack("3f", *res_torques[mol_id]))
            for site_type, site_pos in frame_sites[mol_id]:
                f.write(struct.pack("i", site_type))
                f.write(struct.pack("3f", *site_pos))



def _print_force_percentiles(label, values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        print(f"  {label}: no samples")
        return
    qs = [50.0, 90.0, 95.0, 99.0, 99.9]
    vals = np.percentile(arr, qs)
    print(f"  {label}:")
    for q, value in zip(qs, vals):
        print(f"    P{q:g}: {value:.6g}")
    print(f"    MAX: {np.max(arr):.6g}")

print("\n[INFO] Diagnostica forze sui frame fisici (prima di eventuale clipping):")
_print_force_percentiles("|F_reference|", reference_force_norms)
_print_force_percentiles("|F_WCA pair contribution|", wca_force_norms)
_print_force_percentiles("|F_ML,target residual|", residual_force_norms)

print("\n[INFO] Coda delle forze WCA per type-pair:")
print(f"{'pair':<8} {'N_WCA':>10} {'P99.9 |F|':>16} {'MAX |F|':>16}")
for pair_key in sorted(wca_force_norms_by_pair):
    vals = np.asarray(wca_force_norms_by_pair[pair_key], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        print(f"{pair_key:<8} {0:>10d} {'-':>16} {'-':>16}")
    else:
        print(
            f"{pair_key:<8} {vals.size:>10d} "
            f"{np.percentile(vals, 99.9):>16.6g} {np.max(vals):>16.6g}"
        )

print("\n[INFO] Diagnostica geometrica short-range per type-pair:")
print(
    f"{'pair':<8} {'N_total':>12} {'N_WCA':>10} {'r_min[nm]':>11} "
    f"{'r_min/rc':>10} {'Q0.01%/rc':>12} {'min_source':>12} "
    f"{'1-2':>8} {'1-3':>8} {'NB':>8}"
)
_class_names = {0: "1-2", 1: "1-3", 2: "nonbonded"}
pass2_fit_min_violations = []
pass2_guard_violations = []
for pair_key in sorted(wca_prior_dict):
    w = wca_prior_dict[pair_key]
    ti, tj = int(w["type_i"]), int(w["type_j"])
    code = min(ti, tj) * n_wca_types + max(ti, tj)
    n_per_frame = int(np.count_nonzero(pair_code_all == code))
    n_total = n_per_frame * num_frames
    rc = float(w["cutoff_nm"])

    dists = np.asarray(wca_active_distances_by_pair[pair_key], dtype=float)
    classes = np.asarray(wca_active_classes_by_pair[pair_key], dtype=np.int8)
    if dists.size:
        order = np.argsort(dists)
        dists = dists[order]
        classes = classes[order]
        r_min = float(dists[0])
        fit_min = float(w["empirical_min"])
        r_guard = float(w.get("r_guard_nm", WCA_GUARD_FRACTION * rc))
        min_source = _class_names[int(classes[0])]
        rank = max(0, int(np.ceil(0.0001 * n_total)) - 1)
        q001_text = f"{dists[rank] / rc:.4f}" if rank < dists.size else ">1.0000"
        rmin_text = f"{r_min:.4f}"
        rminrc_text = f"{r_min / rc:.4f}"

        # The WCA fit and Pass 2 must see the same rigid geometry.  A Pass-2
        # minimum below the fit minimum means the support guard was calibrated
        # on a different coordinate representation and is unsafe.
        if r_min < fit_min - 1.0e-9:
            pass2_fit_min_violations.append((pair_key, r_min, fit_min))
        if r_min < r_guard - 1.0e-9:
            pass2_guard_violations.append((pair_key, r_min, r_guard))
    else:
        min_source = "none<rc"
        rmin_text = ">rc"
        rminrc_text = ">1.0000"
        q001_text = ">1.0000"

    counts = [int(np.count_nonzero(classes == cls)) for cls in (0, 1, 2)]
    print(
        f"{pair_key:<8} {n_total:>12d} {len(classes):>10d} "
        f"{rmin_text:>11} {rminrc_text:>10} {q001_text:>12} "
        f"{min_source:>12} {counts[0]:>8d} {counts[1]:>8d} {counts[2]:>8d}"
    )

if pass2_fit_min_violations:
    details = "; ".join(
        f"{pair}: pass2_min={rmin:.6f} < fit_min={fitmin:.6f}"
        for pair, rmin, fitmin in pass2_fit_min_violations
    )
    raise RuntimeError(
        "WCA rigid-geometry consistency check failed: Pass 2 observed shorter "
        f"physical distances than the WCA fit. {details}"
    )
if pass2_guard_violations:
    details = "; ".join(
        f"{pair}: pass2_min={rmin:.6f} < r_guard={rguard:.6f}"
        for pair, rmin, rguard in pass2_guard_violations
    )
    raise RuntimeError(
        "WCA physical-support check failed on the actual Pass-2 rigid geometry. "
        + details
    )
print(
    "[INFO] WCA rigid-geometry consistency check passed: Pass 2 does not "
    "enter the fitted deep-core guard."
)

# --- DECOY GENERATION ---
print(f"\\n[INFO] Generazione Decoy OOD (Deep Core) in corso...")
decoy_frames = []
import copy
import random

rng = random.Random(DECOY_RANDOM_SEED)
n_physical_frames = len(cg_centers_history)
if DECOY_TARGET_FRACTION > 0.0:
    target_decoys = int(round(
        DECOY_TARGET_FRACTION * n_physical_frames / (1.0 - DECOY_TARGET_FRACTION)
    ))
else:
    target_decoys = 0

valid_decoy_specs = []
for pair_key, wca_info in wca_prior_dict.items():
    t1, t2 = int(wca_info["type_i"]), int(wca_info["type_j"])
    r_c = float(wca_info["cutoff_nm"])
    r_emp_min = float(wca_info["empirical_min"])
    r_ood_max = min(0.95 * r_c, r_emp_min - 0.01)
    r_ood_min = 0.70 * r_c
    if r_ood_max <= r_ood_min:
        print(
            f"    [Decoy] Skipping pair {t1}-{t2} "
            f"(rc={r_c:.3f}, r_min={r_emp_min:.3f}) - no valid OOD interval."
        )
        continue
    valid_decoy_specs.append((pair_key, t1, t2, r_ood_min, r_ood_max))

if target_decoys and not valid_decoy_specs:
    raise RuntimeError("Decoy target is non-zero but no type-pair has a valid OOD interval.")

quotas = {}
if valid_decoy_specs:
    base, remainder = divmod(target_decoys, len(valid_decoy_specs))
    for idx, spec in enumerate(valid_decoy_specs):
        quotas[spec[0]] = base + (1 if idx < remainder else 0)

for pair_key, t1, t2, r_ood_min, r_ood_max in valid_decoy_specs:
    quota = quotas[pair_key]
    generated = 0
    attempts = 0
    max_attempts = max(50, quota * 50)
    while generated < quota and attempts < max_attempts:
        attempts += 1
        frame_idx = rng.randrange(len(sites_data_history))
        source_sites = reconstruct_rigid_sites_for_frame(
            sites_data_history[frame_idx],
            cg_centers_history[frame_idx],
            mol_resnames,
            rigid_bodies_info,
        )
        mol_ids_t1 = []
        mol_ids_t2 = []
        for m_idx, sites in enumerate(source_sites):
            for s_type, s_pos in sites:
                if int(s_type) == t1:
                    mol_ids_t1.append((m_idx, s_pos))
                if int(s_type) == t2:
                    mol_ids_t2.append((m_idx, s_pos))
        if not mol_ids_t1 or not mol_ids_t2:
            continue

        m1_idx, pos1 = rng.choice(mol_ids_t1)
        m2_idx, pos2 = rng.choice(mol_ids_t2)
        if m1_idx == m2_idx:
            continue
        if wca_excluded_mol_matrix[m1_idx, m2_idx]:
            continue

        target_r = rng.uniform(r_ood_min, r_ood_max)
        box = np.asarray(box_dim_history[frame_idx])
        dvec = mic_vector(pos2, pos1, box)
        dist = np.linalg.norm(dvec)
        uvec = np.array([1.0, 0.0, 0.0]) if dist < 1e-8 else dvec / dist
        translation = dvec - target_r * uvec

        # Start from the same ideal rigid geometry used for the physical frames.
        # Translating the whole selected molecule preserves all rigid-body offsets.
        decoy_sites = copy.deepcopy(source_sites)
        decoy_centers = np.copy(cg_centers_history[frame_idx])
        decoy_forces = np.zeros_like(cg_forces_history[frame_idx])
        for site_idx in range(len(decoy_sites[m2_idx])):
            s_type, s_pos = decoy_sites[m2_idx][site_idx]
            decoy_sites[m2_idx][site_idx] = (s_type, s_pos + translation)
        decoy_centers[m2_idx] += translation

        decoy_frames.append((decoy_sites, decoy_centers, decoy_forces, box_dim_history[frame_idx]))
        generated += 1

    if generated < quota:
        print(
            f"    [WARN] Pair {t1}-{t2}: generated {generated}/{quota} decoys "
            f"after {attempts} attempts."
        )

print(
    f"[INFO] Generati {len(decoy_frames)} decoy frames "
    f"(target={target_decoys}, target fraction={DECOY_TARGET_FRACTION:.2%}, "
    f"seed={DECOY_RANDOM_SEED})."
)

# Write decoys to dataset
print("[INFO] Scrittura decoy nel binario...")
with open(args.output, "ab") as f:
    for d_idx, (d_sites, d_centers, d_forces, d_box) in enumerate(decoy_frames):
        num_molecules = len(d_sites)
        num_total_sites = sum(len(sites) for sites in d_sites)
        
        f.write(struct.pack("i", num_molecules))
        f.write(struct.pack("i", num_total_sites))
        f.write(struct.pack("3f", float(d_box[0]), float(d_box[1]), float(d_box[2])))

        for mol_id in range(num_molecules):
            num_sites = len(d_sites[mol_id])
            f.write(struct.pack("i", mol_id))
            f.write(struct.pack("i", num_sites))
            f.write(struct.pack("3f", *d_centers[mol_id]))
            f.write(struct.pack("3f", *d_forces[mol_id]))
            f.write(struct.pack("3f", 0.0, 0.0, 0.0)) # torques are zero
            for site_type, site_pos in d_sites[mol_id]:
                f.write(struct.pack("i", int(site_type)))
                f.write(struct.pack("3f", *site_pos))

print(f"[INFO] Scritti {len(decoy_frames)} decoy nel binario.")
with open(args.output, "r+b") as f:
    f.seek(0)
    total_frames = len(cg_centers_history) + len(decoy_frames)
    f.write(struct.pack("i", total_frames))

f_decoy = len(decoy_frames) / total_frames if total_frames > 0 else 0
print(f"[INFO] Aggiornato il contatore dei frame totali: {total_frames} (Frazione decoy: {f_decoy:.2%})")


print("[INFO] Conversione completata e forze residue salvate con successo nel dataset!")


with open(rb_info_output_path, "w") as jf:
    json.dump(rigid_bodies_info, jf, indent=4)
print(f"[INFO] Masse, inerzie e geometrie rigid-body salvate in {rb_info_output_path}")
