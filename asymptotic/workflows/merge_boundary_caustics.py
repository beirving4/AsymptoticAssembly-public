"""Merge per-snapshot boundary-caustic results for one traced halo.

`save_halo_boundary_caustics.py --base_snap` writes one result HDF5 per
snapshot under ``<root_dir>/results/``. This collects them across the halo's
evolutionary track, sorts by snapshot, and writes::

    <root_dir>/<sim>_root<id>_<method>_boundary_caustics_merged.hdf5

containing a ``/summary`` group of arrays over snapshots and one
``/snapshot_<NNN>`` group per snapshot carrying the full detail.

Explicit inputs only: no configuration object, no private infrastructure, no
argument parsing. The excluded research wrapper and the composed runner both
call this one implementation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict

import h5py
import numpy as np
from numpy.typing import NDArray

from ..utils.sim_naming import get_sim_save_name
from .specs import AnalysisRunSpec, MergeBoundaryCausticsSpec

class SnapshotPayload(TypedDict):
    """One snapshot's decoded contents. ``Any`` survives only for individual
    HDF5 attribute leaves, whose types are whatever the writer stored."""

    meta: dict[str, Any]
    density: NDArray[Any]
    extent: NDArray[Any]
    boundaries: dict[str, float]
    radial: dict[str, NDArray[Any]]

_SUMMARY_META = ("scale_factor", "redshift", "fof_group_id", "M200m",
                 "R_200c_physical")
_SUMMARY_BND = ("200m", "200c", "sp", "vmax", "bound")

_METHOD_PATTERN = re.compile(r"_halo\d+_(?P<m>.+)_boundary_caustics\.hdf5$")


def discover_methods(results_dir: Path) -> list[str]:
    """Distinct render-method keys present among the per-snap result files."""
    methods = set()
    for fp in results_dir.glob("*_boundary_caustics.hdf5"):
        if (mm := _METHOD_PATTERN.search(fp.name)):
            methods.add(mm.group("m"))
    return sorted(methods)


def read_snapshot_results(files: list[Path]) -> dict[int, SnapshotPayload]:
    """Decode per-snapshot result files, keyed by snapshot id. Files are read
    in the order given; a repeated snapshot keeps the last file read."""
    per_snap: dict[int, SnapshotPayload] = {}
    for fp in files:
        with h5py.File(fp, "r") as f:
            meta = {k: f.attrs[k] for k in f.attrs}
            snap = int(meta["snap"])
            bnd = {k: float(f["boundaries_physical_mpch"].attrs[k])
                   for k in f["boundaries_physical_mpch"].attrs}
            radial = {k: f["radial"][k][()] for k in f["radial"]}
            per_snap[snap] = {
                "meta": meta,
                "density": f["density"][()],
                "extent": f["extent"][()],
                "boundaries": bnd,
                "radial": radial,
            }
    return per_snap


def write_merged_file(
        out_path: Path,
        per_snap: dict[int, SnapshotPayload],
        snaps: list[int],
        root_halo_id: int,
        method: str,
        sim_save_name: str,
    ) -> None:
    """Write the merged product. Mode ``"w"``: an existing file is replaced."""
    with h5py.File(out_path, "w") as out:
        out.attrs["sim_save_name"] = str(sim_save_name)
        out.attrs["root_halo_id"] = int(root_halo_id)
        out.attrs["method"] = method
        out.attrs["n_snaps"] = len(snaps)
        out.attrs["snaps"] = np.asarray(snaps, dtype=np.int64)

        # ---- summary table (one row per snapshot, ascending) ----
        g = out.create_group("summary")
        g.create_dataset("snap", data=np.asarray(snaps, dtype=np.int64))
        for key in _SUMMARY_META:
            g.create_dataset(key, data=np.asarray(
                [per_snap[s]["meta"].get(key, np.nan) for s in snaps],
                dtype=np.float64))
        for key in _SUMMARY_BND:
            g.create_dataset(f"R_{key}", data=np.asarray(
                [per_snap[s]["boundaries"].get(key, np.nan) for s in snaps],
                dtype=np.float64))

        # ---- full per-snap detail ----
        for s in snaps:
            d = per_snap[s]
            sg = out.create_group(f"snapshot_{s:03d}")
            for k, v in d["meta"].items():
                try:
                    sg.attrs[k] = v
                except (TypeError, ValueError):
                    sg.attrs[k] = str(v)
            sg.create_dataset("density", data=d["density"], compression="gzip")
            sg.create_dataset("extent", data=d["extent"])
            bg = sg.create_group("boundaries_physical_mpch")
            for k, v in d["boundaries"].items():
                bg.attrs[k] = float(v)
            rg = sg.create_group("radial")
            for k, v in d["radial"].items():
                rg.create_dataset(k, data=v)


def print_boundary_preview(
        per_snap: dict[int, SnapshotPayload], snaps: list[int]
    ) -> None:
    """Console preview of the boundary evolution."""
    print(f"  {'snap':>4} {'a':>7} {'group':>6} {'R_200c':>8} {'R_sp':>8} "
          f"{'R_bound':>8} {'R_200m':>9}")
    for s in snaps:
        m, b = per_snap[s]["meta"], per_snap[s]["boundaries"]
        print(f"  {s:>4} {float(m.get('scale_factor', np.nan)):>7.2f} "
              f"{int(m.get('fof_group_id', -1)):>6} "
              f"{b.get('200c', np.nan):>8.3f} {b.get('sp', np.nan):>8.3f} "
              f"{b.get('bound', np.nan):>8.3f} {b.get('200m', np.nan):>9.3f}")


def merge_one_method(
        root_dir: Path,
        results_dir: Path,
        root_halo_id: int,
        method: str,
        sim_save_name: str,
    ) -> bool:
    """Merge every per-snapshot file of one render method. ``False`` when no
    file matches."""
    suffix = f"_{method}_boundary_caustics.hdf5"
    files = sorted(results_dir.glob(f"*{suffix}"))
    if not files:
        print(f"[merge] no *{suffix} files under {results_dir}")
        return False

    per_snap = read_snapshot_results(files)
    snaps = sorted(per_snap)
    print(f"[merge] {len(snaps)} snapshots: {snaps}")

    out_path = root_dir / (f"{sim_save_name}_root{root_halo_id}"
                           f"_{method}_boundary_caustics_merged.hdf5")
    write_merged_file(
        out_path, per_snap, snaps, root_halo_id, method, sim_save_name)

    print(f"[merge] → {out_path}")
    print_boundary_preview(per_snap, snaps)
    return True


def merge_boundary_caustics(
        root_dir: Path,
        results_dir: Path,
        root_halo_id: int,
        method: str,
        sim_save_name: str,
    ) -> int:
    """Merge one method, or every method present when ``method == "all"``.

    A missing results directory is not an error: it prints and succeeds, which
    is what the afterok merge step relies on.
    """
    if not results_dir.exists():
        print(f"[merge] no results dir: {results_dir}")
        return 0

    if method == "all":
        methods = discover_methods(results_dir)
        if not methods:
            print(f"[merge] no *_boundary_caustics.hdf5 under {results_dir}")
            return 0
        print(f"[merge] methods present: {methods}")
        for m in methods:
            merge_one_method(
                root_dir, results_dir, root_halo_id, m, sim_save_name)
    else:
        merge_one_method(
            root_dir, results_dir, root_halo_id, method, sim_save_name)
    return 0


def resolve_directories(
        out_dir: Path, root_halo_id: int
    ) -> tuple[Path, Path]:
    """``(root_dir, results_dir)`` for one traced halo — pure path building."""
    root_dir = out_dir / "halo_boundary_caustics" / f"root_{root_halo_id}"
    return root_dir, root_dir / "results"


def run(spec: AnalysisRunSpec[MergeBoundaryCausticsSpec]) -> int:
    """Composed-configuration entry point."""
    workflow = spec.workflow
    sim_save_name = get_sim_save_name(
        box_size=spec.simulation.box_size,
        num_particles=spec.simulation.num_particles,
        seed_num=spec.simulation.seed_num,
    )
    root_dir, results_dir = resolve_directories(
        spec.output.out_dir, workflow.root_halo_id)
    return merge_boundary_caustics(
        root_dir=root_dir,
        results_dir=results_dir,
        root_halo_id=workflow.root_halo_id,
        method=workflow.method,
        sim_save_name=sim_save_name,
    )
