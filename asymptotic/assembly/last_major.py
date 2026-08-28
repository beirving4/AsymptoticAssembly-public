"""Shared last-major-merger root dump (Chapter 5).

The six ``aLastMerger_{Coal,Infall}_{Major,Tenth,Any}`` clocks are MAIN-BRANCH-
propagated tree columns built at full-tree composition (``_propagate_last_event``
over ``TreeMainProgenitor``); they are NOT reconstructable from the topology-free
merger-rate event catalog. This module just selects the a=final root nodes
(``SnapNum == desc_snap``) and reads those columns + halo masses + the identifiers
needed to join back to the assembly-history catalog.

It is column-source agnostic via a ``get_column(name) -> array | None`` callable,
so both drivers reuse it:
  * ``save_last_major.py`` -- standalone, pure-h5py fast path.
  * ``save_merger_rate.py --with_last_major`` -- reuses the ALREADY-open assembly
    ``store`` (no second tree open), writing the same compact separate file.

The output stays a small (few-MB) separate file that transfers back trivially for
the local Figure-2B build, independent of the 1-2 GB merger-rate file.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

SCHEMA_VERSION = 1
# the six last-merger clocks (timing x threshold) carried on the tree
CLOCK_COLS = (
    "aLastMerger_Coal_Major", "aLastMerger_Coal_Tenth", "aLastMerger_Coal_Any",
    "aLastMerger_Infall_Major", "aLastMerger_Infall_Tenth", "aLastMerger_Infall_Any",
)
# mass definitions carried for binning / analytic isolation (splashback primary,
# matching the assembly-history MAH mass; others enable sensitivity checks)
MASS_DEFS = ("Splashback", "Crit200", "Virial", "Bound", "Mean200")
# identifiers needed to join back to the assembly-history catalog + book-keeping
ID_COLS = ("SubhaloNr", "GroupNr", "TreeID", "SnapNum")


def store_get_column(store):
    """Column reader backed by an open assembly ``store`` (TesseraTreeStore)."""
    def get(name):
        return store.column(name) if store.has_column(name) else None
    return get


def h5py_get_column(tree_halos_group):
    """Column reader backed by an open h5py ``TreeHalos`` group."""
    def get(name):
        return tree_halos_group[name][:] if name in tree_halos_group else None
    return get


def tree_halos_group(f):
    """Locate the TreeHalos group across the schema aliases."""
    for key in ("TreeHalos", "Tree/Halos", "tree_halos"):
        if key in f:
            return f[key]
    raise KeyError(f"No TreeHalos group in {f.filename} (keys: {list(f.keys())})")


def root_dump(get_column, desc_snap: int, mass_defs=MASS_DEFS) -> dict:
    """Select a=final roots (``SnapNum == desc_snap``) and read the clock/mass/id
    columns. Missing clock/mass columns are filled NaN (flagged in
    ``_missing_clock_cols``); missing ids are filled -1."""
    snap = np.asarray(get_column("SnapNum"))
    roots = np.flatnonzero(snap == desc_snap)
    if roots.size == 0:
        raise RuntimeError(
            f"No nodes at SnapNum=={desc_snap}; max SnapNum={snap.max()}")
    out = {"root_abs_index": roots.astype(np.int64)}
    missing = []
    for c in CLOCK_COLS:
        col = get_column(c)
        if col is None:
            missing.append(c)
            out[c] = np.full(roots.size, np.nan)
        else:
            out[c] = np.asarray(col)[roots].astype(np.float64)
    for d in mass_defs:
        col = get_column(f"Group_M_{d}")
        out[f"M_{d}"] = (np.asarray(col)[roots].astype(np.float64)
                         if col is not None else np.full(roots.size, np.nan))
    for c in ID_COLS:
        col = get_column(c)
        out[c] = np.asarray(col)[roots] if col is not None else np.full(roots.size, -1)
    out["_missing_clock_cols"] = missing
    return out


def summarize(roots: dict) -> None:
    """Print per-clock no-event fractions + mass coverage (shared reporting)."""
    n = roots["root_abs_index"].size
    if roots["_missing_clock_cols"]:
        print(f"  WARNING: clock cols missing from tree (filled NaN): "
              f"{roots['_missing_clock_cols']} -> run save_last_major --verify to recompute")
    for c in CLOCK_COLS:
        v = roots[c]
        fin = np.isfinite(v)
        print(f"  {c:28s}: finite={fin.sum():>7} ({fin.mean()*100:5.1f}%)  "
              f"no-event(NaN)={100*(~fin).mean():5.1f}%"
              + (f"  median a={np.nanmedian(v):.3f}" if fin.any() else ""))
    with np.errstate(divide="ignore", invalid="ignore"):
        lsp = np.log10(np.where(roots["M_Splashback"] > 0,
                                roots["M_Splashback"], np.nan)) + 10.0
    if np.isfinite(lsp).any():
        print(f"  log10 M_Splashback (+10): finite={np.isfinite(lsp).sum()} "
              f"range=[{np.nanmin(lsp):.2f},{np.nanmax(lsp):.2f}]")


def verify_recompute(backend, desc_snap: int, roots_abs: np.ndarray,
                     tree_vals: dict, mass_def: str = "Virial",
                     min_particles: int = 100) -> dict:
    """Re-derive the clocks from raw tree links (``compute_last_merger_fields``)
    and compare with the stored columns at the roots. Needs an open backend."""
    from .mergers import compute_last_merger_fields
    fields = compute_last_merger_fields(
        backend.store, backend.moments, backend=backend,
        mass_def=mass_def, min_particles=min_particles, a_min=0.0,
    )
    report = {}
    for c in CLOCK_COLS:
        if c not in fields:
            continue
        rec = np.asarray(fields[c])[roots_abs]
        report[c] = rec.astype(np.float64)
        if c in tree_vals:
            stored = tree_vals[c]
            both = np.isfinite(rec) & np.isfinite(stored)
            agree = (np.allclose(rec[both], stored[both], rtol=1e-4, atol=1e-6)
                     if both.any() else True)
            nev_r = np.mean(~np.isfinite(rec))
            nev_s = np.mean(~np.isfinite(stored))
            print(f"    verify {c:28s}: finite match={agree}  "
                  f"no-event recompute={nev_r*100:.1f}% vs stored={nev_s*100:.1f}%")
    return report


def write_last_major(out_path, roots: dict, *, base_attrs: dict, generator: str,
                     mass_defs=MASS_DEFS, verify: dict | None = None,
                     verify_attrs: dict | None = None) -> None:
    """Write the compact per-root last-major dump (schema shared by both drivers)."""
    import h5py
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = roots["root_abs_index"].size
    with h5py.File(out_path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["generated_at"] = datetime.now(timezone.utc).isoformat()
        f.attrs["generator"] = generator
        f.attrs["mass_unit"] = "1e10 Msun/h (Group_M_*; add 10 to log10 for h^-1 Msun)"
        f.attrs["nan_convention"] = ("NaN aLastMerger_* = no qualifying merger "
                                     "(right-censored), not missing")
        f.attrs["join_key"] = "SubhaloNr <-> assembly_histories group/desc_subhalo_nr"
        f.attrs["clock_cols"] = list(CLOCK_COLS)
        f.attrs["n_roots"] = n
        for k, v in base_attrs.items():
            f.attrs[k] = v
        g = f.create_group("root_last_major")
        g.create_dataset("root_abs_index", data=roots["root_abs_index"])
        for c in CLOCK_COLS:
            g.create_dataset(c, data=roots[c])
        for d in mass_defs:
            g.create_dataset(f"M_{d}", data=roots[f"M_{d}"])
        for c in ID_COLS:
            g.create_dataset(c, data=roots[c])
        if verify:
            vg = f.create_group("verify_recompute")
            for k, v in (verify_attrs or {}).items():
                vg.attrs[k] = v
            for c, v in verify.items():
                vg.create_dataset(c, data=v)
