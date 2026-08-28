"""phusis-backed composite-halo construction for AsymptoticAssembly.

Drop-in replacement for `get_population_averaged_halo_particles` (collection.py)
that uses phusis C++ for tree construction and phase-space querying, then
applies a configurable weighting scheme and radial mapping (multiplicative by
default — fixes the r_min-dependent bug in scale_radial_distances).

Returns a `BoundedParticleCollection` built via `from_dictionary(...)` so
downstream AA methods (`get_mass_distribution`, `get_2D_phase_space_histogram`,
`get_anisotropy_profile`, `get_shape_profile`) work unchanged.

See phusis/specs/weighting-schemes/design.md for the theoretical motivation and
phusis/specs/codebase_overview.md §11.3 for the overcrowding diagnosis.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from AsymptoticAssembly.asymptotic.particles.collection import (
        BoundedParticleCollection,
        BoxParticleCollection,
    )

_VALID_WEIGHTINGS = (
    "equal_per_halo",
    "equal_per_halo_median",
    "number_weighted",
    "mass_weighted",
)

_VALID_MAPPINGS = ("affine", "multiplicative")


def _cache_path_for_tree(
    cache_dir: Path,
    box_particles: "BoxParticleCollection",
    store_velocities: bool,
) -> Path:
    """Build a deterministic cache filename keyed on box-particles identity.

    The hash includes: number of particles, box size, softening length,
    snapshot redshift (via moment), and the store_velocities flag. This is a
    cheap fingerprint that will detect most incompatibilities; full snapshot
    path isn't reliably available, so we fall back on these invariants.
    """
    mdata = (
        f"n={box_particles.properties.num_particles}|"
        f"box={box_particles.box_size:.9f}|"
        f"soft={box_particles.properties.softening_length:.9f}|"
        f"z={box_particles.moment.redshift:.6f}|"
        f"vel={'Y' if store_velocities else 'N'}"
    )
    h = hashlib.sha1(mdata.encode()).hexdigest()[:16]
    return cache_dir / f"phusis_tree_{h}.bin"


def _build_or_load_tree(
    box_particles: "BoxParticleCollection",
    tree_cfg,
    positions: np.ndarray,
    velocities: np.ndarray | None,
    ids: np.ndarray,
    store_velocities: bool,
    cache_dir: Path | None,
    verbose: bool = False,
):
    """Return a phusis KDTree, preferring mmap from cache if available."""
    import _phusis

    if cache_dir is None:
        if store_velocities:
            return _phusis.build_kdtree(positions, velocities, ids, tree_cfg)
        return _phusis.build_kdtree(positions, None, ids, tree_cfg)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path_for_tree(cache_dir, box_particles, store_velocities)

    if cache_file.exists():
        if verbose:
            sz = cache_file.stat().st_size / 1024 ** 2
            print(f"[phusis_adapter] mmap tree from cache: {cache_file} ({sz:.1f} MB)")
        try:
            return _phusis.mmap_binary(str(cache_file))
        except Exception as e:
            if verbose:
                print(f"[phusis_adapter] mmap failed ({e}); rebuilding.")

    if verbose:
        print(f"[phusis_adapter] Building tree (no cache found)...")
    if store_velocities:
        tree = _phusis.build_kdtree(positions, velocities, ids, tree_cfg)
    else:
        tree = _phusis.build_kdtree(positions, None, ids, tree_cfg)

    # Serialize atomically (write to .tmp first, then rename).
    tmp_file = cache_file.with_suffix(".bin.tmp")
    try:
        _phusis.write_binary(str(tmp_file), tree)
        os.replace(tmp_file, cache_file)
        if verbose:
            sz = cache_file.stat().st_size / 1024 ** 2
            print(f"[phusis_adapter] Tree cached: {cache_file} ({sz:.1f} MB)")
    except Exception as e:
        if verbose:
            print(f"[phusis_adapter] Tree serialization failed ({e}); "
                  f"continuing without cache.")
        if tmp_file.exists():
            tmp_file.unlink()

    return tree


def _aggregate(values: np.ndarray, use_median: bool) -> float:
    return float(np.nanmedian(values) if use_median else np.nanmean(values))


def _compute_remapped_radii(
    r_halo: np.ndarray,
    r_min_i: float,
    r_max_i: float,
    target_R: float,
    char_R_i: float,
    r_median_min: float,
    median_diff: float,
    mapping: str,
) -> np.ndarray:
    """Apply the selected radial mapping to one halo's radial distances."""
    if mapping == "multiplicative":
        return r_halo * (target_R / char_R_i)

    # affine — mirrors scale_radial_distances in collection.py:1745
    radial_range = r_max_i - r_min_i
    radial_weight = target_R / r_max_i
    if radial_range <= 0.0:
        return r_halo * radial_weight
    slopes = (r_halo - r_min_i) / radial_range
    return (r_median_min + slopes * median_diff) * radial_weight


def _compute_mass_weight_per_particle(
    scheme: str,
    n_halos: int,
    norm_count_i: int,     # M9: n_virial_i if catalog+virial_norm; else n_total_i
    n_filtered_i: int,
    target_mass_i: float,  # M9: catalog M_200m_i if catalog_masses given, else enclosed
    particle_mass: float,
    target_new_mass: float,
) -> float:
    """Per-particle mass weight (in Msun/h, pre-1E10 normalization)."""
    if scheme in ("equal_per_halo", "equal_per_halo_median"):
        # mass_w[i] = new_mass * query_weight[i] with query_weight = 1/(N*norm_count)
        # M9: norm_count = n_virial_i when catalog-virial mode is on (so that
        # composite's M at r=target_R equals new_mass exactly).
        return target_new_mass / (n_halos * norm_count_i)
    if scheme == "number_weighted":
        return particle_mass
    if scheme == "mass_weighted":
        return target_mass_i / n_filtered_i
    raise ValueError(f"Unsupported weighting scheme: {scheme}")


def get_population_averaged_halo_particles_phusis(
    box_particles: "BoxParticleCollection",
    coordinates: np.ndarray,
    group_velocities: np.ndarray,
    query_radii: np.ndarray,
    characteristic_radii: np.ndarray,
    hubble: float,
    little_h: float,
    use_comoving: bool = True,
    run_sort_by_radial_distance: bool = True,
    weighting: str = "equal_per_halo_median",
    radial_mapping: str = "multiplicative",
    min_particles: int = 50,
    num_threads: int = 4,
    store_velocities_in_tree: bool = True,
    use_f32_coords: bool = False,
    catalog_masses: np.ndarray | None = None,
    tree_cache_dir: str | Path | None = None,
    verbose: bool = False,
) -> "BoundedParticleCollection | None":
    """Build a composite BoundedParticleCollection via phusis.

    Arguments:
        box_particles: simulation-box ParticleCollection (with positions,
            velocities, ids, box_size, softening_length).
        coordinates: (N_halos, 3) halo centers.
        group_velocities: (N_halos, 3) halo bulk velocities.
        query_radii: (N_halos,) query radius per halo.
        characteristic_radii: (N_halos,) R_{mdef} per halo — used by
            multiplicative mapping as r_char_i.
        hubble: H0 in km/s/Mpc (phusis internally divides by little_h to
            compute the Mpc/h Hubble factor).
        little_h: h = H0/100.
        use_comoving: passed through to BoundedParticleProperties.
        run_sort_by_radial_distance: if True, collection sorts particles by
            radial distance post-construction.
        weighting: one of 'equal_per_halo', 'equal_per_halo_median',
            'number_weighted', 'mass_weighted'.
        radial_mapping: 'multiplicative' (default, resolution-independent) or
            'affine' (legacy CDF remap).
        min_particles: per-halo minimum; halos below this are skipped.
        num_threads: OpenMP threads for phusis query.

    Returns:
        BoundedParticleCollection, or None if no halo passed min_particles.

    Note:
        Unlike collection.py:1667 (which trims n-1 particles per halo), this
        adapter keeps all particles. Strict-equivalence testing should account
        for this +N_halos size difference.
    """
    if weighting not in _VALID_WEIGHTINGS:
        raise ValueError(
            f"weighting={weighting!r} not in {_VALID_WEIGHTINGS}"
        )
    if radial_mapping not in _VALID_MAPPINGS:
        raise ValueError(
            f"radial_mapping={radial_mapping!r} not in {_VALID_MAPPINGS}"
        )

    import _phusis
    from AsymptoticAssembly.asymptotic.particles.collection import (
        BoundedParticleCollection,
    )

    coordinates = np.ascontiguousarray(coordinates, dtype=np.float64)
    group_velocities = np.ascontiguousarray(group_velocities, dtype=np.float64)
    query_radii = np.ascontiguousarray(query_radii, dtype=np.float64)
    characteristic_radii = np.asarray(characteristic_radii, dtype=np.float64)

    n_halos_input = coordinates.shape[0]
    if (
        group_velocities.shape[0] != n_halos_input
        or query_radii.shape[0] != n_halos_input
        or characteristic_radii.shape[0] != n_halos_input
    ):
        raise ValueError(
            "coordinates, group_velocities, query_radii, and "
            "characteristic_radii must all have the same leading dimension"
        )

    # ------------------------------------------------------------------
    # 1. Build phusis KD-tree from box_particles
    # ------------------------------------------------------------------
    positions = np.ascontiguousarray(
        box_particles.properties.coordinates, dtype=np.float64
    )
    velocities = np.ascontiguousarray(
        box_particles.properties.peculiar.velocities, dtype=np.float64
    )
    ids = np.ascontiguousarray(box_particles.ids, dtype=np.uint64)

    # Defaults keep the f64 + stored-velocities tree (closest to the legacy AA
    # path). Opt-in flags:
    #   store_velocities_in_tree=False  — M8.1: saves ~84% tree RAM; query
    #                                      gathers velocities via tree.indices()
    #   use_f32_coords=True             — M8.2: f32 position storage, lossless
    #                                      for GADGET-4-style f32-origin data
    tree_cfg = _phusis.TreeConfig()
    tree_cfg.boxsize = float(box_particles.box_size)
    tree_cfg.store_velocities = store_velocities_in_tree
    tree_cfg.use_f32_coords = use_f32_coords

    # M8.3: if a cache dir is provided, mmap a previously serialized tree when
    # available; otherwise build and cache. Tree fingerprint is derived from
    # particle count / box_size / softening / redshift / store_velocities flag.
    cache_dir_path = Path(tree_cache_dir) if tree_cache_dir is not None else None
    tree = _build_or_load_tree(
        box_particles=box_particles,
        tree_cfg=tree_cfg,
        positions=positions,
        velocities=velocities if store_velocities_in_tree else None,
        ids=ids,
        store_velocities=store_velocities_in_tree,
        cache_dir=cache_dir_path,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 2. Phase-space query (C++ batched)
    # ------------------------------------------------------------------
    ps_kwargs = dict(
        hubble=float(hubble),
        little_h=float(little_h),
        sort=False,
        num_threads=int(num_threads),
    )
    if not store_velocities_in_tree:
        ps_kwargs["particle_velocities"] = velocities

    ps = _phusis.query_phase_space(
        tree,
        coordinates,
        group_velocities,
        query_radii,
        **ps_kwargs,
    )
    offsets = np.asarray(ps["offsets"])
    indices = np.asarray(ps["indices"])
    r_all = np.asarray(ps["r"])
    dx_all, dy_all, dz_all = (
        np.asarray(ps["dx"]),
        np.asarray(ps["dy"]),
        np.asarray(ps["dz"]),
    )
    dvx_all, dvy_all, dvz_all = (
        np.asarray(ps["dvx"]),
        np.asarray(ps["dvy"]),
        np.asarray(ps["dvz"]),
    )
    pec_vr_all = np.asarray(ps["peculiar_vr"])
    act_vr_all = np.asarray(ps["actual_vr"])

    # ------------------------------------------------------------------
    # 3. Per-halo filter (r > 0, min_particles) + collect stats
    # ------------------------------------------------------------------
    per_halo_slices: list[tuple[int, np.ndarray]] = []
    r_min_list: list[float] = []
    r_max_list: list[float] = []
    enclosed_radii_list: list[float] = []
    n_total_list: list[int] = []
    n_filtered_list: list[int] = []
    char_R_list: list[float] = []
    valid_halo_indices_list: list[int] = []  # M9: original halo index per entry

    particle_mass = float(box_particles.properties.masses[0])

    for h in range(n_halos_input):
        start, end = int(offsets[h]), int(offsets[h + 1])
        n_total = end - start
        if n_total < min_particles:
            continue

        char_R_i = characteristic_radii[h]
        if radial_mapping == "multiplicative":
            if not np.isfinite(char_R_i) or char_R_i <= 0.0:
                continue

        r_h = r_all[start:end]
        pos_mask = r_h > 0.0
        n_filtered = int(pos_mask.sum())
        if n_filtered < min_particles:
            continue

        # sort indices within the halo (ascending r) — so that per-halo
        # arrays are in radial order before concatenation.
        local_idx = np.nonzero(pos_mask)[0]
        local_idx = local_idx[np.argsort(r_h[local_idx])]

        per_halo_slices.append((start, local_idx))
        r_min_list.append(float(r_h[local_idx[0]]))
        r_max_list.append(float(r_h[local_idx[-1]]))
        enclosed_radii_list.append(float(r_h[local_idx[-1]]))
        n_total_list.append(n_total)
        n_filtered_list.append(n_filtered)
        char_R_list.append(float(char_R_i))
        valid_halo_indices_list.append(h)

    n_valid = len(per_halo_slices)
    if n_valid == 0:
        return None

    r_min_arr = np.asarray(r_min_list)
    r_max_arr = np.asarray(r_max_list)
    enclosed_radii_arr = np.asarray(enclosed_radii_list)
    n_total_arr = np.asarray(n_total_list)
    n_filtered_arr = np.asarray(n_filtered_list)
    char_R_arr = np.asarray(char_R_list)

    # ------------------------------------------------------------------
    # 4. Compute population-level aggregates (mean vs median)
    # ------------------------------------------------------------------
    use_median = weighting == "equal_per_halo_median"
    target_R_enc = _aggregate(enclosed_radii_arr, use_median)

    enclosed_masses = particle_mass * n_total_arr

    # M9: target-mass source. When catalog_masses is provided, use those values
    # (e.g. M_200m from FoF catalog) for target-mass aggregation instead of
    # query-region sums — fixes the 1.5-1.9x overestimation at 200*rho_m crossing.
    if catalog_masses is not None:
        catalog_masses_arr = np.asarray(catalog_masses, dtype=np.float64)
        if len(catalog_masses_arr) != coordinates.shape[0]:
            raise ValueError(
                f"catalog_masses length {len(catalog_masses_arr)} != "
                f"n_halos {coordinates.shape[0]}"
            )
        target_masses_arr = catalog_masses_arr[valid_halo_indices_list]
    else:
        target_masses_arr = enclosed_masses

    target_new_mass = _aggregate(target_masses_arr, use_median)

    # M9: virial-norm — when catalog + multiplicative are set, use n_virial_i
    # (particles within R_200m_i) in the per-halo denominator so the composite's
    # M at r=target_R equals new_mass exactly.
    use_virial_norm = (
        catalog_masses is not None
        and radial_mapping == "multiplicative"
    )
    if use_virial_norm:
        n_virial_arr = np.empty(n_valid, dtype=np.int64)
        for i, (start, local_idx) in enumerate(per_halo_slices):
            R_char = float(char_R_arr[i])
            # local_idx was sorted by r above → r_all[start+local_idx] ascending
            r_sorted = r_all[start + local_idx]
            n_vir = int(np.searchsorted(r_sorted, R_char, side="right"))
            n_virial_arr[i] = max(1, n_vir)
        norm_counts_arr = n_virial_arr
    else:
        norm_counts_arr = n_total_arr

    if radial_mapping == "affine":
        r_median_min = 10.0 ** _aggregate(np.log10(r_min_arr), use_median)
        r_median_max = 10.0 ** _aggregate(np.log10(r_max_arr), use_median)
        median_diff = r_median_max - r_median_min
        target_R_for_remap = target_R_enc
    else:  # multiplicative
        target_R_for_remap = _aggregate(char_R_arr, use_median)
        r_median_min = 0.0
        median_diff = 0.0

    # ------------------------------------------------------------------
    # 5. Per-halo remap + mass-weight assembly
    # ------------------------------------------------------------------
    all_r_out: list[np.ndarray] = []
    all_coords_out: list[np.ndarray] = []
    all_pec_vrel_out: list[np.ndarray] = []
    all_pec_vr_out: list[np.ndarray] = []
    all_act_vrel_out: list[np.ndarray] = []
    all_act_vr_out: list[np.ndarray] = []
    all_mass_values_out: list[np.ndarray] = []
    all_ids_out: list[np.ndarray] = []

    box_ids = np.asarray(box_particles.ids, dtype=np.uint64)

    for i, (start, local_idx) in enumerate(per_halo_slices):
        abs_idx = start + local_idx
        r_h = r_all[abs_idx]
        remapped = _compute_remapped_radii(
            r_halo=r_h,
            r_min_i=r_min_arr[i],
            r_max_i=r_max_arr[i],
            target_R=(
                target_R_for_remap if radial_mapping == "multiplicative"
                else target_R_enc
            ),
            char_R_i=char_R_arr[i],
            r_median_min=r_median_min,
            median_diff=median_diff,
            mapping=radial_mapping,
        )
        all_r_out.append(remapped)

        coords = np.stack(
            [dx_all[abs_idx], dy_all[abs_idx], dz_all[abs_idx]], axis=1
        )
        all_coords_out.append(coords)

        pec_vrel = np.stack(
            [dvx_all[abs_idx], dvy_all[abs_idx], dvz_all[abs_idx]], axis=1
        )
        all_pec_vrel_out.append(pec_vrel)
        all_pec_vr_out.append(pec_vr_all[abs_idx])
        all_act_vrel_out.append(pec_vrel)  # actual relative = peculiar relative (3D)
        all_act_vr_out.append(act_vr_all[abs_idx])

        mass_w = _compute_mass_weight_per_particle(
            scheme=weighting,
            n_halos=n_valid,
            norm_count_i=int(norm_counts_arr[i]),
            n_filtered_i=int(n_filtered_arr[i]),
            target_mass_i=float(target_masses_arr[i]),
            particle_mass=particle_mass,
            target_new_mass=float(target_new_mass),
        )
        # Divide by 1E10 — BoundedParticleProperties.masses does `1E10 *
        # mass_values` (properties.py:95-96).
        mass_arr = np.full(n_filtered_arr[i], mass_w / 1e10)
        all_mass_values_out.append(mass_arr)

        halo_ids = box_ids[np.asarray(indices[abs_idx], dtype=np.int64)]
        all_ids_out.append(halo_ids)

    # ------------------------------------------------------------------
    # 6. Concatenate into a single composite and build properties dict
    # ------------------------------------------------------------------
    concat_r = np.concatenate(all_r_out)
    concat_coords = np.concatenate(all_coords_out, axis=0)
    concat_pec_vrel = np.concatenate(all_pec_vrel_out, axis=0)
    concat_pec_vr = np.concatenate(all_pec_vr_out)
    concat_act_vrel = np.concatenate(all_act_vrel_out, axis=0)
    concat_act_vr = np.concatenate(all_act_vr_out)
    concat_mass_values = np.concatenate(all_mass_values_out)
    concat_ids = np.concatenate(all_ids_out)

    # Accelerations: zeros — the 4 downstream profile methods on
    # BoundedParticleCollection do not read accelerations.
    n_particles = concat_r.shape[0]
    zeros_vec3 = np.zeros((n_particles, 3))
    zeros_1d = np.zeros(n_particles)

    properties = {
        "volume": (4.0 / 3.0) * np.pi * target_R_enc ** 3,
        "moment": box_particles.moment,
        "mass_values": concat_mass_values,
        "particle_ids": concat_ids,
        "positions": {
            "coordinates": concat_coords,
            "radial": concat_r,
        },
        "peculiar": {
            "velocities": {
                "relative": concat_pec_vrel,
                "radial": concat_pec_vr,
            },
            "accelerations": {
                "relative": zeros_vec3,
                "radial": zeros_1d,
            },
        },
        "actual": {
            "velocities": {
                "relative": concat_act_vrel,
                "radial": concat_act_vr,
            },
            "accelerations": {
                "radial": zeros_1d,
            },
        },
        "center_of_mass": coordinates,
        "in_comoving_units": use_comoving,
        "softening_length": float(box_particles.properties.softening_length),
        "run_sort_by_radial_distance": run_sort_by_radial_distance,
        "num_particles_in_query": n_total_list,
    }

    return BoundedParticleCollection.from_dictionary(properties)
