"""Factories and helpers for ESPResSo interactions used by the MLCG runtime."""

from __future__ import annotations

from typing import Any

import math


DEFAULT_MORSE_R_CUT_NM = 15.0
DEFAULT_MORSE_SWITCH_FRACTION = 0.75


def switched_morse_energy_radial_force(
    distance: float, *, D: float, a: float, r0: float, r_switch: float, r_cut: float
) -> tuple[float, float]:
    """Return switched Morse energy and signed radial force.

    The unswitched gauge is ``U(r0)=-D`` and ``U(infinity)=0``.  A quintic
    smoothstep multiplies the Morse tail between ``r_switch`` and ``r_cut``;
    both energy and force are exactly zero at and beyond ``r_cut``.
    """
    r = float(distance)
    if r >= float(r_cut):
        return 0.0, 0.0
    y = math.exp(-float(a) * (r - float(r0)))
    base_energy = float(D) * (y * y - 2.0 * y)
    base_force = 2.0 * float(D) * float(a) * y * (y - 1.0)
    if r <= float(r_switch):
        return base_energy, base_force

    width = float(r_cut) - float(r_switch)
    t = (r - float(r_switch)) / width
    switch = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
    d_switch_dr = -30.0 * t * t * (1.0 - t) * (1.0 - t) / width
    return switch * base_energy, switch * base_force - base_energy * d_switch_dr


def make_analytic_morse_bond(interactions: Any, prior: dict[str, Any]):
    """Construct the legacy conservative MLCG Morse bonded interaction.

    This helper is intentionally retained for the breakage regression diagnostic.
    Production/equilibration runtimes use :func:`prepare_pair_specific_morse`
    and :func:`configure_pair_specific_morse` instead, because a bonded Morse
    returning ``nullopt`` at ``r_cut`` is treated by ESPResSo as a fatal broken
    bond rather than as reversible dissociation.
    """
    cls = getattr(interactions, "MorseBond", None)
    if cls is None:
        raise RuntimeError(
            "This simulation contains a legacy Morse bonded diagnostic, but the current "
            "ESPResSo build has no espressomd.interactions.MorseBond. Re-run "
            "simulation/espresso_plugin/copy_plugin_files.sh and rebuild ESPResSo."
        )

    D = float(prior["D"])
    a = float(prior["a"])
    r0 = float(prior["r0"])
    r_cut = float(prior.get("r_cut", DEFAULT_MORSE_R_CUT_NM))
    if D < 0.0:
        raise ValueError("Morse D must be >= 0")
    if a <= 0.0:
        raise ValueError("Morse a must be > 0")
    if r0 < 0.0:
        raise ValueError("Morse r0 must be >= 0")
    if r_cut <= 0.0:
        raise ValueError("Morse r_cut must be > 0")
    return cls(D=D, a=a, r_0=r0, r_cut=r_cut)


def _normalize_switched_morse_parameters(
    prior: dict[str, Any], *, label: str
) -> dict[str, float]:
    """Validate and normalize one conservative switched Morse parameter set."""
    D = float(prior["D"])
    a = float(prior["a"])
    r0 = float(prior["r0"])
    r_cut = float(prior.get("r_cut", DEFAULT_MORSE_R_CUT_NM))
    if D < 0.0:
        raise ValueError(f"{label} D must be >= 0")
    if a <= 0.0:
        raise ValueError(f"{label} a must be > 0")
    if r0 < 0.0:
        raise ValueError(f"{label} r0 must be >= 0")
    if r_cut <= r0:
        raise ValueError(
            f"{label} requires r_cut > r0, got r0={r0} r_cut={r_cut}"
        )

    if "r_switch" in prior:
        r_switch = float(prior["r_switch"])
    else:
        r_switch = r0 + DEFAULT_MORSE_SWITCH_FRACTION * (r_cut - r0)
    if not (r0 < r_switch < r_cut):
        raise ValueError(
            f"{label} requires r0 < r_switch < r_cut, got "
            f"r0={r0} r_switch={r_switch} r_cut={r_cut}"
        )
    return {
        "D": D,
        "a": a,
        "r0": r0,
        "r_cut": r_cut,
        "r_switch": r_switch,
    }


def _normalize_morse_contact(prior: dict[str, Any], index: int) -> dict[str, Any]:
    """Normalize one explicit pair-specific Morse contact.

    ``site=-1`` addresses the rigid-body COM; non-negative indices address the
    corresponding CG site.  Site existence is validated later against the
    particles created from the dataset, where the number of sites is known.
    """
    mol_i = int(prior["mol_i"])
    mol_j = int(prior["mol_j"])
    site_i = int(prior.get("site_i", -1))
    site_j = int(prior.get("site_j", -1))
    if site_i < -1 or site_j < -1:
        raise ValueError(
            f"Morse bond[{index}] uses invalid site indices {site_i}, {site_j}; "
            "use -1 for COM or a non-negative CG-site index"
        )
    if mol_i < 0 or mol_j < 0 or mol_i == mol_j:
        raise ValueError(
            f"Morse bond[{index}] has invalid molecule pair {mol_i} <-> {mol_j}"
        )

    params = _normalize_switched_morse_parameters(
        prior, label=f"Morse bond[{index}]"
    )
    return {
        "index": index,
        "mol_i": mol_i,
        "site_i": site_i,
        "mol_j": mol_j,
        "site_j": site_j,
        **params,
    }


def prepare_type_pair_morse(
    priors: dict[str, Any], num_species: int
) -> list[dict[str, Any]]:
    """Normalize broad non-bonded Morse interactions selected by CG site type.

    ``morse_type_pairs`` uses the physical CG site types 0..num_species-1.
    Unlike pair-specific explicit contacts, one entry applies to every non-excluded
    site pair carrying the selected types, exactly following ESPResSo's normal
    ``non_bonded_inter[type_i, type_j]`` semantics.
    """
    n_species = int(num_species)
    raw_entries = priors.get("morse_type_pairs", [])
    if not raw_entries:
        return []
    if n_species <= 0:
        raise ValueError("num_species must be positive for Morse type-pair interactions")

    normalized: list[dict[str, Any]] = []
    seen: dict[tuple[int, int], int] = {}
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ValueError(f"morse_type_pairs[{index}] must be an object")
        type_i = int(raw["type_i"])
        type_j = int(raw["type_j"])
        if "r_cut" not in raw:
            raise ValueError(
                f"morse_type_pairs[{index}] requires an explicit r_cut because "
                "type-pair Morse contributes to the regular neighbor-search cutoff"
            )
        if not (0 <= type_i < n_species and 0 <= type_j < n_species):
            raise ValueError(
                f"morse_type_pairs[{index}] uses site types {type_i}, {type_j}, "
                f"but valid CG site types are 0..{n_species - 1}"
            )
        pair = tuple(sorted((type_i, type_j)))
        if pair in seen:
            raise ValueError(
                f"Duplicate Morse type pair {pair[0]} <-> {pair[1]} in "
                f"morse_type_pairs[{seen[pair]}] and morse_type_pairs[{index}]"
            )
        seen[pair] = index
        params = _normalize_switched_morse_parameters(
            raw, label=f"morse_type_pairs[{index}]"
        )
        normalized.append({
            "index": index,
            "type_i": pair[0],
            "type_j": pair[1],
            **params,
        })
    return normalized


def configure_type_pair_morse(
    system: Any, interactions: list[dict[str, Any]]
) -> None:
    """Configure ordinary ESPResSo type-pair non-bonded switched Morse terms."""
    for item in interactions:
        try:
            system.non_bonded_inter[item["type_i"], item["type_j"]].morse.set_params(
                eps=item["D"],
                alpha=item["a"],
                rmin=item["r0"],
                cutoff=item["r_cut"],
                switch_start=item["r_switch"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to configure a Morse type-pair interaction. The ESPResSo "
                "build must expose the MLCG switched non-bonded Morse extension; run "
                "simulation/espresso_plugin/copy_plugin_files.sh, re-run CMake, and "
                f"rebuild ESPResSo. Original error: {exc}"
            ) from exc


def max_type_pair_morse_cutoff(interactions: list[dict[str, Any]]) -> float:
    """Return the largest regular-decomposition cutoff used by type-pair Morse."""
    return max((float(item["r_cut"]) for item in interactions), default=0.0)


def prepare_pair_specific_morse(
    priors: dict[str, Any], num_species: int
) -> tuple[dict[tuple[int, int], int], list[dict[str, Any]]]:
    """Plan reversible pair-specific Morse interactions on explicit endpoints.

    Each endpoint is addressed as ``(mol_id, site_id)`` where ``site_id=-1``
    denotes the rigid-body COM and a non-negative value denotes a CG site.

    ESPResSo non-bonded interactions are selected by particle type, but changing
    the physical CG-site type would also change the PaiNN species.  The runtime
    therefore creates one coincident *virtual marker* for every endpoint used by
    pair-specific Morse.  Marker types are technical, unique per endpoint, and
    live above the physical ML-species range.  Forces on a marker are transferred
    by ESPResSo to the parent rigid body at the marker position, preserving the
    correct translational force and torque while leaving PaiNN/WCA/type-pair
    interactions on the original CG site untouched.
    """
    if int(num_species) < 0:
        raise ValueError("num_species must be non-negative")

    contacts: list[dict[str, Any]] = []
    seen_pairs: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    endpoints: set[tuple[int, int]] = set()
    for index, prior in enumerate(priors.get("bonds", [])):
        if str(prior.get("type", "harmonic")).lower() != "morse":
            continue
        contact = _normalize_morse_contact(prior, index)
        endpoint_i = (contact["mol_i"], contact["site_i"])
        endpoint_j = (contact["mol_j"], contact["site_j"])
        pair = tuple(sorted((endpoint_i, endpoint_j)))
        if pair in seen_pairs:
            raise ValueError(
                "Duplicate pair-specific Morse endpoint pair "
                f"{pair[0][0]}:{pair[0][1]} <-> {pair[1][0]}:{pair[1][1]} in "
                f"bond[{seen_pairs[pair]}] and bond[{index}]"
            )
        seen_pairs[pair] = index
        endpoints.update((endpoint_i, endpoint_j))
        contacts.append(contact)

    # ``num_species + 1`` is the shared COM type used by the runtime. Marker
    # types start one slot above it and are deterministic across runs.
    first_marker_type = int(num_species) + 2
    marker_types = {
        endpoint: first_marker_type + offset
        for offset, endpoint in enumerate(sorted(endpoints))
    }
    return marker_types, contacts


def create_pair_specific_morse_markers(
    system: Any,
    marker_types: dict[tuple[int, int], int],
    mol_com_parts: dict[int, int],
    mol_vs_parts: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    """Create coincident virtual marker particles for explicit Morse endpoints.

    The markers are deliberately *not* inserted into ``mol_vs_parts``: they are
    technical interaction carriers, not physical CG sites.  Consequently they
    are ignored by PaiNN and by the physical WCA/type-pair parameter tables.
    """
    marker_parts: dict[tuple[int, int], int] = {}
    for endpoint, runtime_type in sorted(marker_types.items()):
        mol_id, site_id = endpoint
        if mol_id not in mol_com_parts:
            raise ValueError(
                f"Pair-specific Morse endpoint references missing molecule {mol_id}"
            )
        parent_pid = mol_com_parts[mol_id]
        if site_id == -1:
            target_pid = parent_pid
        else:
            target_pid = mol_vs_parts.get((mol_id, site_id))
            if target_pid is None:
                available = sorted(
                    site for (mol, site) in mol_vs_parts if mol == mol_id
                )
                raise ValueError(
                    "Pair-specific Morse endpoint references missing CG site "
                    f"{mol_id}:{site_id}; available site indices are {available}"
                )

        target_pos = list(system.part.by_id(target_pid).pos)
        marker = system.part.add(
            pos=target_pos,
            type=int(runtime_type),
            mass=1.0e-5,
            rinertia=[1.0e-5, 1.0e-5, 1.0e-5],
            mol_id=int(mol_id),
        )
        marker.virtual = True
        marker.vs_auto_relate_to(parent_pid)
        marker.gamma = 0.0
        marker.gamma_rot = 0.0
        marker_parts[endpoint] = marker.id
    return marker_parts


def configure_pair_specific_morse(
    system: Any,
    contacts: list[dict[str, Any]],
    marker_types: dict[tuple[int, int], int],
) -> None:
    """Activate switched Morse terms between technical endpoint marker types."""
    for contact in contacts:
        endpoint_i = (contact["mol_i"], contact["site_i"])
        endpoint_j = (contact["mol_j"], contact["site_j"])
        type_i = marker_types[endpoint_i]
        type_j = marker_types[endpoint_j]
        try:
            system.non_bonded_inter[type_i, type_j].morse.set_params(
                eps=contact["D"],
                alpha=contact["a"],
                rmin=contact["r0"],
                cutoff=contact["r_cut"],
                switch_start=contact["r_switch"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to configure reversible switched Morse. The ESPResSo source "
                "must contain the MLCG switched non-bonded Morse extension; run "
                "simulation/espresso_plugin/copy_plugin_files.sh and rebuild ESPResSo. "
                f"Original error: {exc}"
            ) from exc
