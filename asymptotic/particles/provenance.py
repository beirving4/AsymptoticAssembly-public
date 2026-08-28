"""Import-safe scientific core for the initial-conditions -> asymptotic-state
particle-provenance pipeline (Remote Work Task 2 / Chapter 5 capstone).

This module holds ONLY pure scientific primitives. It must NOT parse CLI
arguments, decide filesystem locations, or import the heavy `Cosmology`/`yt`
stack. The staged driver (`postprocessing/save_asymptotic_provenance.py`) does
all I/O, snapshot loading, selection, and routing; it calls these functions.

Reference set (the "fixed final splashback set"):

    S_ref^sp_i ≡ S_sp,i(a_ref = 100)   (finite a=100 reference, NOT infinity)

i.e. the exact ParticleID set inside host i's final spherical splashback
boundary at snapshot 074. We trace those *same* IDs to earlier snapshots and
measure the cloud's geometry/kinematics.

Conventions (audited against the existing pipeline — see remote_work maps):
  * ID -> row mapping is argsort + searchsorted (the snapshot is SUBFIND-
    reordered, so `ParticleID - 1` is WRONG except in the raw IC snapshot,
    where it is validated as a contiguous lattice). See `locate_rows`.
  * Inertia-tensor shapes match `asymptotic/particles/shapes.py`:
    eigenvalues sqrt-sorted descending (major,inter,minor); q=b/a, s=c/a,
    triaxiality=(1-q^2)/(1-s^2). Reduced tensor is single-pass, weighted by
    the ordinary axis lengths.
  * beta = 1 - sigma_t^2 / (2 sigma_r^2), computed the *particle-property* way
    (scalar radial variance + (N,3) tangential variance) so it does not trip
    the standalone `anisotropy.radial_anisotropy` (N,3)-only shape check.
  * The Hubble flow uses the SAME closed form as the retention pipeline
    `_kinematics`: v_H[km/s] = 100*sqrt(Om/a^3 + OL) * r_physical[Mpc/h]
    (little_h cancels; r must be the *physical* Mpc/h displacement).
  * Lagrangian sizes (r50/r90) are measured in COMOVING coordinates; the
    Hubble term in beta_actual uses PHYSICAL distance. The driver supplies the
    right frame to each function.

GADGET-4 IC ParticleID lattice convention (validated before use):
    ID = 1 + z + y*N + x*N^2     (0 <= x,y,z < N)
so row0 = ID-1, x = row0 // N^2, y = (row0 % N^2) // N, z = row0 % N.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =============================================================================
# Schema / provenance constants
# =============================================================================

SCHEMA_VERSION = 1
#: Velocity-anisotropy convention recorded in output metadata.
BETA_CONVENTION = "actual_relative_minus_shellmean"
#: ParticleID lattice convention recorded in output metadata.
ID_CONVENTION = "gadget4_lattice: ID = 1 + z + y*N + x*N^2"
#: Reference-set semantics recorded in output metadata.
REFERENCE_SEMANTICS = (
    "fixed final spherical splashback membership at a_ref=100 (snap 074); "
    "finite reference, not literal infinity"
)


# =============================================================================
# Ragged reference-membership representation
# =============================================================================

@dataclass
class RaggedMembership:
    """CSR-style ragged container for per-halo fixed reference ID sets.

    halo_ids[i] owns particle ids `particle_ids[offsets[i]:offsets[i+1]]`.
    Membership sets MAY overlap across halos (final splashback spheres can
    overlap); ids are never silently de-duplicated across halos.
    """

    halo_ids: np.ndarray            # (H,) int64
    offsets: np.ndarray             # (H+1,) int64, cumulative
    particle_ids: np.ndarray        # (sum N_i,) uint64, per-halo sorted-unique

    def __post_init__(self) -> None:
        self.halo_ids = np.asarray(self.halo_ids, dtype=np.int64)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        self.particle_ids = np.asarray(self.particle_ids, dtype=np.uint64)
        if self.offsets.shape[0] != self.halo_ids.shape[0] + 1:
            raise ValueError("offsets must have length n_halos + 1")
        if self.offsets[0] != 0 or self.offsets[-1] != self.particle_ids.shape[0]:
            raise ValueError("offsets must start at 0 and end at len(particle_ids)")
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("offsets must be non-decreasing")

    @property
    def n_halos(self) -> int:
        return int(self.halo_ids.shape[0])

    def ids_for(self, i: int) -> np.ndarray:
        return self.particle_ids[self.offsets[i]:self.offsets[i + 1]]

    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    @classmethod
    def from_id_lists(cls, halo_ids, id_lists) -> "RaggedMembership":
        halo_ids = np.asarray(halo_ids, dtype=np.int64)
        cleaned = [np.unique(np.asarray(ids, dtype=np.uint64)) for ids in id_lists]
        counts = np.array([c.shape[0] for c in cleaned], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        flat = (np.concatenate(cleaned) if cleaned
                else np.empty(0, dtype=np.uint64))
        return cls(halo_ids=halo_ids, offsets=offsets, particle_ids=flat)


# =============================================================================
# ID -> row mapping
# =============================================================================

def id_to_row_sorter(box_ids: np.ndarray) -> np.ndarray:
    """argsort of box ParticleIDs (stable).

    The snapshot is SUBFIND-reordered, so `box_ids` is a permutation of the id
    set (NOT identity); `searchsorted` with this sorter maps query ids -> rows.
    """
    return np.argsort(np.asarray(box_ids), kind="stable")


def locate_rows(query_ids: np.ndarray, box_ids: np.ndarray,
                sorter: Optional[np.ndarray] = None, *,
                allow_missing: bool = False):
    """Rows in the box arrays for each query id.

    Returns `rows` (int64). If `allow_missing` is False (default) every query
    id must be present or a RuntimeError is raised (mirrors the retention
    pipeline's all-present-or-raise contract). If True, returns
    `(rows, found_mask)` and rows for missing ids are -1.
    """
    box_ids = np.asarray(box_ids)
    q = np.asarray(query_ids, dtype=box_ids.dtype)
    if sorter is None:
        sorter = id_to_row_sorter(box_ids)
    pos = np.searchsorted(box_ids, q, sorter=sorter)
    pos = np.clip(pos, 0, len(sorter) - 1)
    rows = sorter[pos].astype(np.int64)
    found = box_ids[rows] == q
    if not allow_missing:
        if not np.all(found):
            bad = int(np.count_nonzero(~found))
            raise RuntimeError(
                f"locate_rows: {bad}/{len(q)} ids not found in box (id mismatch).")
        return rows
    rows = np.where(found, rows, -1)
    return rows, found


def ic_lattice_is_valid(box_ids: np.ndarray, n_grid: int) -> bool:
    """True iff `box_ids` is the identity ordering [1, 2, ..., N^3].

    Gate for the IC fast path (`row = ParticleID - 1`), which requires the
    snapshot to store particles in lattice/ID order. A SUBFIND-reordered
    snapshot contains the SAME id *set* but as a permutation, so a set-only
    check would wrongly green-light the fast path there — hence the strict
    identity-order check. If this fails, callers fall back to `locate_rows`.
    """
    box_ids = np.asarray(box_ids)
    n_expected = int(n_grid) ** 3
    if box_ids.shape[0] != n_expected:
        return False
    if int(box_ids[0]) != 1 or int(box_ids[-1]) != n_expected:
        return False  # cheap reject before the full O(N) compare
    return bool(np.array_equal(
        box_ids, np.arange(1, n_expected + 1, dtype=box_ids.dtype)))


def ic_fast_rows(query_ids: np.ndarray) -> np.ndarray:
    """row = ParticleID - 1 (IC fast path). Caller must have validated the
    lattice with `ic_lattice_is_valid` first."""
    return (np.asarray(query_ids, dtype=np.int64) - 1)


# =============================================================================
# GADGET-4 IC lattice decode + connected components
# =============================================================================

def ids_to_lattice(ids: np.ndarray, n_grid: int):
    """Decode GADGET-4 IC ParticleIDs to integer lattice coords (x, y, z).

    ID = 1 + z + y*N + x*N^2  =>  row0 = ID-1,  x = row0 // N^2,
    y = (row0 % N^2)//N,  z = row0 % N.
    """
    n = int(n_grid)
    row0 = np.asarray(ids, dtype=np.int64) - 1
    x = row0 // (n * n)
    rem = row0 % (n * n)
    y = rem // n
    z = rem % n
    return x.astype(np.int64), y.astype(np.int64), z.astype(np.int64)


# 6- and 26-neighbour offsets on a periodic cubic lattice.
def _neighbour_offsets(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                         (0, 0, 1), (0, 0, -1)], dtype=np.int64)
    if connectivity == 26:
        offs = [(dx, dy, dz)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                if not (dx == 0 and dy == 0 and dz == 0)]
        return np.array(offs, dtype=np.int64)
    raise ValueError("connectivity must be 6 or 26")


def lattice_components(ids: np.ndarray, n_grid: int,
                       connectivity: int = 6) -> np.ndarray:
    """Connected-component labels for a set of IC particles on the periodic
    cubic lattice (deterministic; no metric linking length).

    Returns `labels` (int64, length len(ids)) where particles in the same
    periodic adjacency component share a label. Labels are 0..K-1 ordered by
    descending component size (label 0 = largest).

    Vectorised: neighbour edges are built with `searchsorted` (the cell of each
    particle is unique) and the components are found with
    `scipy.sparse.csgraph.connected_components` (C-fast), so this scales to
    millions of particles.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ids = np.asarray(ids, dtype=np.int64)
    n = int(n_grid)
    N = ids.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)
    if N == 1:
        return np.zeros(1, dtype=np.int64)

    x, y, z = ids_to_lattice(ids, n)
    cell = (x.astype(np.int64) * n + y) * n + z          # unique per particle
    order = np.argsort(cell, kind="stable")
    cell_sorted = cell[order]

    src_list, dst_list = [], []
    for dx, dy, dz in _neighbour_offsets(connectivity):
        nx = (x + dx) % n
        ny = (y + dy) % n
        nz = (z + dz) % n
        ncell = (nx.astype(np.int64) * n + ny) * n + nz
        pos = np.clip(np.searchsorted(cell_sorted, ncell), 0, N - 1)
        hit = cell_sorted[pos] == ncell
        if hit.any():
            src_list.append(np.nonzero(hit)[0])           # original-order index
            dst_list.append(order[pos[hit]])              # neighbour original index
    if src_list:
        r = np.concatenate(src_list)
        c = np.concatenate(dst_list)
        adj = coo_matrix((np.ones(r.shape[0], dtype=np.int8), (r, c)),
                         shape=(N, N))
    else:
        adj = coo_matrix((N, N), dtype=np.int8)
    _, labels = connected_components(adj, directed=False)

    uniq, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order_by_size = np.argsort(-counts, kind="stable")
    remap = np.empty(uniq.shape[0], dtype=np.int64)
    remap[order_by_size] = np.arange(uniq.shape[0], dtype=np.int64)
    return remap[inv].astype(np.int64)


@dataclass
class ComponentDiagnostics:
    n_components: int
    largest_fraction: float
    mass_fractions: np.ndarray          # descending, fraction of particles
    sep_two_largest: float              # distance between the two largest centers
    labels: np.ndarray = field(repr=False)


def component_diagnostics(labels: np.ndarray,
                          centered_coords: Optional[np.ndarray] = None,
                          box_size: Optional[float] = None) -> ComponentDiagnostics:
    """Summary stats for a component labelling (label 0 = largest)."""
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.shape[0]
    if n == 0:
        return ComponentDiagnostics(0, np.nan, np.empty(0), np.nan, labels)
    uniq, counts = np.unique(labels, return_counts=True)   # robust to gaps
    order = np.argsort(-counts, kind="stable")
    uniq_s, counts_s = uniq[order], counts[order]
    fracs = counts_s / float(n)
    sep = np.nan
    if centered_coords is not None and uniq_s.shape[0] >= 2:
        c0 = centered_coords[labels == uniq_s[0]].mean(axis=0)
        c1 = centered_coords[labels == uniq_s[1]].mean(axis=0)
        d = c0 - c1
        if box_size is not None:
            d -= box_size * np.round(d / box_size)
        sep = float(np.linalg.norm(d))
    return ComponentDiagnostics(
        n_components=int(uniq_s.shape[0]),
        largest_fraction=float(fracs[0]),
        mass_fractions=fracs,
        sep_two_largest=sep,
        labels=labels,
    )


def _component_center(centered_coords, mask, box_size):
    sub = centered_coords[mask]
    if box_size is not None:
        # already centered/min-imaged in caller's frame; plain mean is fine
        return sub.mean(axis=0)
    return sub.mean(axis=0)


# =============================================================================
# Periodic unwrapping + centers
# =============================================================================

def min_image(disp: np.ndarray, box_size: float) -> np.ndarray:
    """Minimum-image wrap of a displacement array into (-L/2, L/2]."""
    disp = np.asarray(disp, dtype=np.float64)
    return disp - box_size * np.round(disp / box_size)


def recenter(coords: np.ndarray, center: np.ndarray,
             box_size: float) -> np.ndarray:
    """Centered, minimum-image displacement of `coords` about `center`."""
    return min_image(np.asarray(coords, dtype=np.float64)
                     - np.asarray(center, dtype=np.float64), box_size)


def periodic_center_of_mass(coords: np.ndarray, box_size: float,
                            anchor: Optional[np.ndarray] = None,
                            masses: Optional[np.ndarray] = None,
                            n_iter: int = 2) -> np.ndarray:
    """Periodic-unwrapped center of mass.

    Unwraps about an anchor (default: first particle), takes the (mass-)mean of
    the unwrapped displacements, and iterates a couple of times so the result
    is robust to the anchor choice. Returns a center wrapped into [0, L).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return np.full(3, np.nan)
    w = None if masses is None else np.asarray(masses, dtype=np.float64)
    center = (np.asarray(anchor, dtype=np.float64)
              if anchor is not None else coords[0].copy())
    for _ in range(max(1, n_iter)):
        disp = min_image(coords - center, box_size)
        shift = (np.average(disp, axis=0, weights=w) if w is not None
                 else disp.mean(axis=0))
        center = center + shift
    return np.mod(center, box_size)


def potential_min_center(coords: np.ndarray,
                         potentials: np.ndarray) -> np.ndarray:
    """Position of the (traced) particle with the minimum gravitational
    potential — the paper-matched inertia-tensor origin."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0 or potentials is None:
        return np.full(3, np.nan)
    return coords[int(np.nanargmin(np.asarray(potentials)))].copy()


# =============================================================================
# Geometry: percentile radii, compactness
# =============================================================================

def percentile_radii(centered_coords: np.ndarray,
                     percentiles=(50.0, 90.0)) -> dict:
    """Percentile radii of centered (min-imaged) coordinates."""
    centered = np.asarray(centered_coords, dtype=np.float64)
    r = np.sqrt(np.einsum("ij,ij->i", centered, centered))
    vals = np.percentile(r, list(percentiles))
    return {float(p): float(v) for p, v in zip(percentiles, vals)}


def compactness(r50: float, r90: float) -> float:
    """Cloud compactness / Lagrangian concentration c = R90/R50 (NOT NFW)."""
    if r50 is None or r50 == 0 or not np.isfinite(r50):
        return np.nan
    return float(r90) / float(r50)


def extent_stats(centered_coords: np.ndarray) -> dict:
    centered = np.asarray(centered_coords, dtype=np.float64)
    r = np.sqrt(np.einsum("ij,ij->i", centered, centered))
    return {"rms_extent": float(np.sqrt(np.mean(r ** 2))),
            "max_extent": float(np.max(r)) if r.size else np.nan}


# =============================================================================
# Shape: ordinary + reduced inertia tensor  (matches asymptotic/particles/shapes.py)
# =============================================================================

def inertia_tensor(centered_coords: np.ndarray) -> np.ndarray:
    c = np.asarray(centered_coords, dtype=np.float64)
    return np.einsum("ki,kj->ij", c, c)


def axis_lengths(tensor: np.ndarray) -> np.ndarray:
    """sqrt of eigenvalues, sorted DESCENDING -> (major a, inter b, minor c)."""
    eig = np.linalg.eigvalsh(np.asarray(tensor, dtype=np.float64))
    eig = np.clip(eig, 0.0, None)
    return np.sqrt(np.sort(eig)[::-1])


def reduced_inertia_tensor(centered_coords: np.ndarray,
                           ax_lengths: np.ndarray) -> np.ndarray:
    """Single-pass reduced inertia tensor, weighted by 1/r_ell where r_ell uses
    the ORDINARY axis lengths (matches shapes.reduced_inertia_tensor)."""
    c = np.asarray(centered_coords, dtype=np.float64)
    a, b, cc = float(ax_lengths[0]), float(ax_lengths[1]), float(ax_lengths[2])
    s, q = (cc / a if a else np.nan), (b / a if a else np.nan)
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    r_ell = np.sqrt(x ** 2 + (y / q) ** 2 + (z / s) ** 2)
    mask = r_ell > 0
    cm = c[mask]
    weighted = cm / r_ell[mask][:, None]
    return np.einsum("ki,kj->ij", weighted, weighted)


@dataclass
class ShapeResult:
    q: float            # b/a
    s: float            # c/a
    ellipticity: float  # (1 - s^2) / (1 + s^2 + q^2)
    triaxiality: float  # (1 - q^2) / (1 - s^2)
    reduced_q: float
    reduced_s: float
    reduced_triaxiality: float


def _ratios(ax):
    a, b, c = float(ax[0]), float(ax[1]), float(ax[2])
    q = b / a if a else np.nan
    s = c / a if a else np.nan
    return q, s


def shape_parameters(centered_coords: np.ndarray) -> ShapeResult:
    """Ordinary + (single-pass) reduced shape, project conventions."""
    c = np.asarray(centered_coords, dtype=np.float64)
    I = inertia_tensor(c)
    ax = axis_lengths(I)
    q, s = _ratios(ax)
    S = reduced_inertia_tensor(c, ax)
    rax = axis_lengths(S)
    rq, rs = _ratios(rax)
    def _tri(qq, ss):
        denom = (1.0 - ss ** 2)
        return (1.0 - qq ** 2) / denom if denom != 0 else np.nan
    ell_denom = (1.0 + s ** 2 + q ** 2)
    ellipticity = (1.0 - s ** 2) / ell_denom if ell_denom != 0 else np.nan
    return ShapeResult(
        q=q, s=s, ellipticity=ellipticity, triaxiality=_tri(q, s),
        reduced_q=rq, reduced_s=rs, reduced_triaxiality=_tri(rq, rs),
    )


# =============================================================================
# Kinematics: radial flow, anisotropy, angular momentum
# =============================================================================

def hubble_flow_kms(r_physical_mpch: np.ndarray, a: float,
                    omega_m: float, omega_l: float) -> np.ndarray:
    """Hubble-flow speed at physical radius r, matching the retention pipeline.

    v_H[km/s] = (H(a)/h) * r  with H(a) = 100 h sqrt(Om/a^3 + OL)
              = 100 * sqrt(Om/a^3 + OL) * r_physical[Mpc/h]   (h cancels).
    `r_physical_mpch` is the *physical* displacement magnitude in Mpc/h
    (= comoving Mpc/h * a).
    """
    hub = 100.0 * np.sqrt(omega_m / (a ** 3) + omega_l)
    return hub * np.asarray(r_physical_mpch, dtype=np.float64)


@dataclass
class KinematicsResult:
    mean_vr_pec: float
    median_vr_pec: float
    sigma_r: float           # std of peculiar radial velocity
    sigma_v_3d_pec: float    # 3D peculiar velocity dispersion (sqrt of summed var)
    beta_actual: float
    beta_pec: float          # QA / numerical-equivalence check
    j_vector: np.ndarray     # mean specific angular momentum (km^2/s)
    j_magnitude: float


def radial_tangential_split(disp_physical: np.ndarray,
                            vel_rel: np.ndarray):
    """Decompose relative velocities into (scalar radial, (N,3) tangential).

    `disp_physical` is the centered, min-imaged PHYSICAL displacement; `vel_rel`
    is the bulk-subtracted peculiar velocity (km/s). Returns
    (rhat, r, vr_pec, vt_vec). r==0 particles get rhat=0 and vr_pec=0.
    """
    d = np.asarray(disp_physical, dtype=np.float64)
    v = np.asarray(vel_rel, dtype=np.float64)
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    r_safe = np.where(r > 0, r, 1.0)
    rhat = d / r_safe[:, None]
    rhat[r == 0] = 0.0
    vr_pec = np.einsum("ij,ij->i", v, rhat)
    vt_vec = v - vr_pec[:, None] * rhat          # tangential (Hubble is radial -> cancels)
    return rhat, r, vr_pec, vt_vec


def anisotropy_beta(vr: np.ndarray, vt_vec: np.ndarray) -> float:
    """beta = 1 - sigma_t^2 / (2 sigma_r^2), computed the particle-property way
    (scalar radial variance + (N,3) tangential variance) to avoid the standalone
    helper's (N,3)-only radial shape check."""
    sigma_r2 = float(np.nanvar(np.asarray(vr, dtype=np.float64), axis=0))
    sigma_t2 = float(np.nanvar(np.asarray(vt_vec, dtype=np.float64), axis=0).sum())
    if sigma_r2 == 0 or not np.isfinite(sigma_r2):
        return np.nan
    return 1.0 - 0.5 * (sigma_t2 / sigma_r2)


def specific_angular_momentum(disp_physical_mpch: np.ndarray,
                              vel_rel_kms: np.ndarray) -> np.ndarray:
    """Mean specific angular momentum j = mean(r x v) (km^2/s).

    r is converted Mpc -> km (1 Mpc = 3.085677581e19 km); v is km/s. Equal
    particle mass (DM) => specific AM is the simple mean of r x v.
    """
    MPC_TO_KM = 3.085677581491367e19
    r = np.asarray(disp_physical_mpch, dtype=np.float64) * MPC_TO_KM
    v = np.asarray(vel_rel_kms, dtype=np.float64)
    cross = np.cross(r, v)
    return cross.mean(axis=0)


def compute_kinematics(disp_physical_mpch: np.ndarray, vel_rel_kms: np.ndarray,
                       a: float, omega_m: float, omega_l: float
                       ) -> KinematicsResult:
    """Full fixed-set kinematics from physical displacement + bulk-subtracted
    peculiar velocity.

    `disp_physical_mpch`: centered min-imaged PHYSICAL displacement (Mpc/h).
    `vel_rel_kms`:        v_peculiar - v_bulk (km/s).
    """
    rhat, r, vr_pec, vt_vec = radial_tangential_split(disp_physical_mpch, vel_rel_kms)
    v_h = hubble_flow_kms(r, a, omega_m, omega_l)
    vr_act = vr_pec + v_h                          # actual radial velocity
    beta_actual = anisotropy_beta(vr_act, vt_vec)
    beta_pec = anisotropy_beta(vr_pec, vt_vec)
    sigma_v_3d = float(np.sqrt(np.nanvar(vel_rel_kms, axis=0).sum()))
    j_vec = specific_angular_momentum(disp_physical_mpch, vel_rel_kms)
    return KinematicsResult(
        mean_vr_pec=float(np.nanmean(vr_pec)),
        median_vr_pec=float(np.nanmedian(vr_pec)),
        sigma_r=float(np.nanstd(vr_pec)),
        sigma_v_3d_pec=sigma_v_3d,
        beta_actual=float(beta_actual),
        beta_pec=float(beta_pec),
        j_vector=j_vec,
        j_magnitude=float(np.linalg.norm(j_vec)),
    )


# =============================================================================
# Cross-halo membership overlap (no forced de-duplication)
# =============================================================================

@dataclass
class OverlapResult:
    overlap_fraction: np.ndarray        # (H,) fraction of halo i's set shared with ANY other host
    strongest_neighbour: np.ndarray     # (H,) max pairwise shared fraction
    strongest_neighbour_id: np.ndarray  # (H,) halo id of the strongest overlap (or -1)


def overlap_diagnostics(membership: RaggedMembership) -> OverlapResult:
    """Per-halo overlap fractions across hosts. Particles may belong to more
    than one host's reference set; nothing is de-duplicated."""
    H = membership.n_halos
    sets = [membership.ids_for(i) for i in range(H)]
    sizes = np.array([s.shape[0] for s in sets], dtype=np.int64)
    overlap_any = np.zeros(H, dtype=np.float64)
    strongest = np.zeros(H, dtype=np.float64)
    strongest_id = np.full(H, -1, dtype=np.int64)
    # Build a global id -> count map to find shared particles cheaply.
    all_ids = membership.particle_ids
    uniq, counts = np.unique(all_ids, return_counts=True)
    shared = uniq[counts > 1]
    shared_set = set(shared.tolist())
    for i in range(H):
        if sizes[i] == 0:
            overlap_any[i] = np.nan
            continue
        si = sets[i]
        in_shared = np.fromiter((int(x) in shared_set for x in si.tolist()),
                                dtype=bool, count=si.shape[0])
        overlap_any[i] = float(in_shared.sum()) / float(sizes[i])
        # strongest pairwise neighbour (only meaningful if there is overlap)
        if in_shared.any():
            best, best_j = 0.0, -1
            shared_i = set(si[in_shared].tolist())
            for j in range(H):
                if j == i or sizes[j] == 0:
                    continue
                inter = len(shared_i.intersection(sets[j].tolist()))
                frac = inter / float(sizes[i])
                if frac > best:
                    best, best_j = frac, int(membership.halo_ids[j])
            strongest[i] = best
            strongest_id[i] = best_j
    return OverlapResult(overlap_any, strongest, strongest_id)


# =============================================================================
# Quality flags
# =============================================================================

@dataclass
class QualityFlags:
    n_recovered: int
    n_reference: int
    recovery_complete: bool
    min_particles_ok: bool
    unwrap_ok: bool             # cloud smaller than L/2 in every axis after unwrap
    finite_values_ok: bool
    center_sensitive: bool      # potential-min vs COM centers differ by > tol*R50
    fragmented: bool            # largest-component fraction below threshold


def quality_flags(n_recovered: int, n_reference: int, *,
                  min_particles: int = 1000,
                  centered_coords: Optional[np.ndarray] = None,
                  box_size: Optional[float] = None,
                  r50: Optional[float] = None,
                  center_offset: Optional[float] = None,
                  center_sens_tol: float = 0.5,
                  largest_fraction: Optional[float] = None,
                  fragment_tol: float = 0.5,
                  extra_finite: Optional[list] = None) -> QualityFlags:
    finite_ok = True
    unwrap_ok = True
    if centered_coords is not None:
        cc = np.asarray(centered_coords, dtype=np.float64)
        finite_ok = bool(np.all(np.isfinite(cc)))
        if box_size is not None and cc.size:
            unwrap_ok = bool(np.all(np.abs(cc) <= 0.5 * box_size + 1e-9))
    if extra_finite:
        finite_ok = finite_ok and all(
            bool(np.all(np.isfinite(np.asarray(v)))) for v in extra_finite)
    center_sensitive = False
    if center_offset is not None and r50 is not None and r50 > 0:
        center_sensitive = bool(center_offset > center_sens_tol * r50)
    fragmented = (bool(largest_fraction < fragment_tol)
                  if largest_fraction is not None else False)
    return QualityFlags(
        n_recovered=int(n_recovered),
        n_reference=int(n_reference),
        recovery_complete=bool(n_recovered == n_reference),
        min_particles_ok=bool(n_reference >= min_particles),
        unwrap_ok=unwrap_ok,
        finite_values_ok=finite_ok,
        center_sensitive=center_sensitive,
        fragmented=fragmented,
    )


# =============================================================================
# Convenience aggregator (driver-facing, still pure)
# =============================================================================

def measure_fixed_set(coords_comoving: np.ndarray,
                      vel_peculiar: np.ndarray,
                      center_comoving: np.ndarray,
                      bulk_velocity: np.ndarray,
                      box_size_comoving: float,
                      a: float, omega_m: float, omega_l: float) -> dict:
    """Compute the full fixed-set property block for one halo from already-
    sliced (per-halo) particle arrays.

    Geometry/shape use COMOVING coordinates (Lagrangian sizes); kinematics use
    the PHYSICAL displacement (for the Hubble term), computed here as
    `disp_comoving * a` on the small per-halo slice — so the driver never has to
    duplicate the full ~400 MB physical-coordinate array (memory-lean). The
    minimum-image wrap is preserved under the uniform scaling. Returns a flat
    dict of scalars/vectors.
    """
    disp_c = recenter(coords_comoving, center_comoving, box_size_comoving)
    disp_p = disp_c * a                      # physical displacement (Mpc/h)
    vel_rel = np.asarray(vel_peculiar, dtype=np.float64) - np.asarray(bulk_velocity)

    rad = percentile_radii(disp_c, (50.0, 90.0))
    r50, r90 = rad[50.0], rad[90.0]
    shape = shape_parameters(disp_c)
    kin = compute_kinematics(disp_p, vel_rel, a, omega_m, omega_l)
    ext = extent_stats(disp_c)

    out = {
        "r50_comoving": r50,
        "r90_comoving": r90,
        "c_compactness": compactness(r50, r90),
        "q": shape.q, "s": shape.s,
        "ellipticity": shape.ellipticity, "triaxiality": shape.triaxiality,
        "reduced_q": shape.reduced_q, "reduced_s": shape.reduced_s,
        "reduced_triaxiality": shape.reduced_triaxiality,
        "rms_extent_comoving": ext["rms_extent"],
        "max_extent_comoving": ext["max_extent"],
        "mean_vr_pec": kin.mean_vr_pec,
        "median_vr_pec": kin.median_vr_pec,
        "sigma_r": kin.sigma_r,
        "sigma_v_3d_pec": kin.sigma_v_3d_pec,
        "beta_actual": kin.beta_actual,
        "beta_pec": kin.beta_pec,
        "j_magnitude": kin.j_magnitude,
        "j_x": float(kin.j_vector[0]),
        "j_y": float(kin.j_vector[1]),
        "j_z": float(kin.j_vector[2]),
    }
    return out
