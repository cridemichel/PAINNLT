"""Numerically testable geometry helpers used by CG preprocessing."""

from __future__ import annotations

import numpy as np


def diagonalize_inertia_tensor(inertia_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted principal moments and a right-handed body-to-space basis.

    The columns of ``principal_axes`` are the principal axes expressed in the
    current space-fixed frame. Therefore ``principal_axes.T @ r_space`` gives
    body-frame coordinates.
    """
    inertia_tensor = np.asarray(inertia_tensor, dtype=float)
    if inertia_tensor.shape != (3, 3):
        raise ValueError("inertia_tensor must have shape (3, 3)")
    if not np.allclose(inertia_tensor, inertia_tensor.T, atol=1.0e-12, rtol=0.0):
        raise ValueError("inertia_tensor must be symmetric")

    eigvals, principal_axes = np.linalg.eigh(inertia_tensor)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    principal_axes = principal_axes[:, order]
    if np.linalg.det(principal_axes) < 0.0:
        principal_axes[:, -1] *= -1.0
    diagonal = principal_axes.T @ inertia_tensor @ principal_axes
    if not np.allclose(diagonal, np.diag(eigvals), atol=1.0e-8, rtol=1.0e-8):
        raise RuntimeError("Failed to diagonalize the rigid-body inertia tensor")
    return eigvals, principal_axes


def minimum_image_displacements(positions: np.ndarray, box_l: np.ndarray) -> np.ndarray:
    """Return all pair displacements using an orthorhombic minimum image."""
    positions = np.asarray(positions, dtype=float)
    box_l = np.asarray(box_l, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if box_l.shape != (3,) or np.any(box_l <= 0.0):
        raise ValueError("box_l must contain three positive lengths")
    displacement = positions[:, None, :] - positions[None, :, :]
    displacement -= box_l * np.round(displacement / box_l)
    return displacement


def minimum_image_distance_matrix(positions: np.ndarray, box_l: np.ndarray) -> np.ndarray:
    """Return the pairwise MIC distance matrix for an orthorhombic box."""
    return np.linalg.norm(minimum_image_displacements(positions, box_l), axis=-1)
