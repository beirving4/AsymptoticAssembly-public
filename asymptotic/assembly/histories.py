"""Per-aperture property-evolution container + HDF5 serialization.

One :class:`ApertureHistorySet` corresponds to one output file (one aperture):
the global time axis, descendant ids, the aligned ``[n_halos, n_snaps]`` property
arrays + per-quantity status, per-quantity evolution rates, the mass accretion
rate, the group circular-velocity trajectory, assembly scalars (incl. both
formation times + per-quantity peaks), per-property stats, and the
position-validation/correction record.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from attrs import define, field

from ..simulation.moments import MomentsInTime
from .accretion import AccretionRates, AssemblyScalars, QuantityRates

SCHEMA_VERSION = 2
GENERATOR = "save_assembly_histories.py"

_PCTS = (5, 25, 50, 75, 95)
_PCT_NAMES = {5: "5pct", 25: "25pct", 50: "median", 75: "75pct", 95: "95pct"}


def compute_property_stats(values: np.ndarray, scale_factors: np.ndarray) -> dict[str, np.ndarray]:
    """Per-snapshot stats for a ``[n_halos, n_snaps]`` array, each as ``[n_snaps, 2]``."""
    a = np.asarray(scale_factors, dtype=float)

    def stack(stat_1d: np.ndarray) -> np.ndarray:
        return np.column_stack([a, stat_1d])

    with np.errstate(all="ignore"):
        out = {
            "mean": stack(np.nanmean(values, axis=0)),
            "std": stack(np.nanstd(values, axis=0)),
            "min": stack(np.nanmin(values, axis=0)),
            "max": stack(np.nanmax(values, axis=0)),
        }
        for p in _PCTS:
            out[_PCT_NAMES[p]] = stack(np.nanpercentile(values, p, axis=0))
    return out


@define(slots=True)
class ApertureHistorySet:
    aperture: str
    moments: MomentsInTime
    desc_node: np.ndarray
    desc_tree_id: np.ndarray
    desc_subhalo_nr: np.ndarray
    masses: np.ndarray                          # log10 M_final at this aperture
    properties: dict[str, np.ndarray]           # name -> [n_halos, n_snaps]
    status: dict[str, np.ndarray]               # name -> [n_halos, n_snaps] uint8 (per quantity)
    rates: dict[str, QuantityRates]             # name -> generic d X / d ln a rates
    peaks: dict[str, tuple[np.ndarray, np.ndarray]]  # name -> (peak, a_at_peak)
    accretion: AccretionRates                   # mass log-Gamma
    scalars: AssemblyScalars
    velocity: np.ndarray | None = None          # group V_c trajectory [n_halos, n_snaps]
    pos_status: np.ndarray | None = None        # [n_halos, n_snaps] position-validation status
    validated_group_nr: np.ndarray | None = None  # [n_halos, n_snaps] matched GroupNr (NaN else)
    stats: dict[str, dict[str, np.ndarray]] = field(factory=dict)

    @property
    def n_halos(self) -> int:
        return len(self.desc_node)

    @property
    def n_snaps(self) -> int:
        return len(self.moments)

    def compute_stats(self) -> None:
        a = self.moments.scale_factors
        self.stats = {
            name: compute_property_stats(values, a) for name, values in self.properties.items()
        }
        if self.accretion is not None:
            self.stats["gamma_tdyn"] = compute_property_stats(self.accretion.gamma_tdyn, a)

    def to_hdf5(self, path: Path, extra_attrs: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["generated_at"] = datetime.now(timezone.utc).isoformat()
            f.attrs["generator"] = GENERATOR
            f.attrs["aperture"] = self.aperture
            f.attrs["mass_def"] = f"M{self.aperture}"   # legacy-loader compatibility
            f.attrs["n_halos"] = self.n_halos
            f.attrs["n_snaps"] = self.n_snaps
            for k, v in (extra_attrs or {}).items():
                f.attrs[k] = v

            self.moments.to_hdf5(f.create_group("time_axis"))

            g = f.create_group("group")
            g.create_dataset("desc_index", data=self.desc_node)
            g.create_dataset("desc_tree_id", data=self.desc_tree_id)
            g.create_dataset("desc_subhalo_nr", data=self.desc_subhalo_nr)
            g.create_dataset("masses", data=self.masses)
            if self.velocity is not None:
                g.create_dataset("velocity", data=self.velocity)  # group V_c [n, n_snaps]

            # legacy-compatible (n_halos, n_snaps, 2) = (scale_factor, M / M_final)
            if "M" in self.properties and self.scalars is not None:
                a = np.broadcast_to(self.moments.scale_factors, self.properties["M"].shape)
                with np.errstate(invalid="ignore", divide="ignore"):
                    frac = self.properties["M"] / self.scalars.m_final[:, None]
                g.create_dataset("histories", data=np.stack([a, frac], axis=-1))

            props = g.create_group("properties")
            for name, values in self.properties.items():
                props.create_dataset(name, data=values)

            stat_grp = g.create_group("status")          # per-quantity status arrays
            for name, values in self.status.items():
                stat_grp.create_dataset(name, data=values)

            rate_grp = g.create_group("rates")           # per-quantity d X / d ln a
            for name, r in self.rates.items():
                q = rate_grp.create_group(name)
                q.create_dataset("inst", data=r.rate_inst)
                q.create_dataset("tdyn", data=r.rate_tdyn)
                q.create_dataset("status", data=r.status)

            if self.accretion is not None:
                acc = g.create_group("accretion")        # mass log-Gamma
                acc.create_dataset("gamma_inst", data=self.accretion.gamma_inst)
                acc.create_dataset("gamma_half_tdyn", data=self.accretion.gamma_half_tdyn)
                acc.create_dataset("gamma_tdyn", data=self.accretion.gamma_tdyn)
                acc.create_dataset("status", data=self.accretion.status)

            if self.pos_status is not None:
                val = g.create_group("validation")
                val.create_dataset("pos_status", data=self.pos_status)
                if self.validated_group_nr is not None:
                    val.create_dataset("validated_group_nr", data=self.validated_group_nr)

            if self.scalars is not None:
                sc = g.create_group("scalars")
                s = self.scalars
                sc.create_dataset("m_final", data=s.m_final)
                sc.create_dataset("m_peak", data=s.m_peak)
                sc.create_dataset("a_at_mpeak", data=s.a_at_mpeak)
                sc.create_dataset("v_peak", data=s.v_peak)
                sc.create_dataset("a_at_vpeak", data=s.a_at_vpeak)
                sc.create_dataset("a_half", data=s.a_half)                  # half FINAL mass
                sc.create_dataset("formation_times", data=s.a_half)        # legacy alias
                sc.create_dataset("a_half_present", data=s.a_half_present)  # half a=1 mass
                sc.create_dataset("a_half_present_status", data=s.a_half_present_status)
                # robust-MAH formation diagnostics (raw = uncleaned, legacy interp)
                sc.create_dataset("a_half_raw", data=s.a_half_raw)
                sc.create_dataset("a_half_present_raw", data=s.a_half_present_raw)
                sc.create_dataset("raw_max_over_final", data=s.raw_max_over_final)
                sc.create_dataset("n_extreme_jumps", data=s.n_extreme_jumps)
                sc.create_dataset("was_clipped", data=s.was_clipped)
                sc.create_dataset("a_freeze_out", data=s.a_freeze_out)
                sc.create_dataset("a_mass_accum", data=s.a_mass_accum)
                sc.create_dataset("a_last_major_merger", data=s.a_last_major_merger)
                # per-quantity peaks
                pk = sc.create_group("peaks")
                for name, (peak, a_at_peak) in self.peaks.items():
                    q = pk.create_group(name)
                    q.create_dataset("peak", data=peak)
                    q.create_dataset("a_at_peak", data=a_at_peak)

            if self.stats:
                st = g.create_group("stats")
                for name, blocks in self.stats.items():
                    q = st.create_group(name)
                    for stat_name, arr in blocks.items():
                        q.create_dataset(stat_name, data=arr)
