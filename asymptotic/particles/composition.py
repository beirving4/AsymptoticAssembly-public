from __future__ import annotations

import re

import h5py
import numpy as np

from pathlib import Path
from attrs import define, field

@define(slots=True)
class ParticleClassificationMetrics:
    true_positives: int
    false_positives: int 
    false_negatives: int 

    @property
    def precision(self) -> float:
        return float(precision(self.true_positives, self.false_positives))
    
    @property
    def recall(self) -> float:
        return float(recall(self.true_positives, self.false_negatives))
    
    @property
    def f1(self) -> float:
        return float(f1_score(self.precision, self.recall))
    
    @property
    def as_array(self) -> np.ndarray:
        return np.array([
            self.true_positives, 
            self.false_positives, 
            self.false_negatives,
            self.precision,
            self.recall,
            self.f1
        ])
    
    @classmethod
    def from_particle_ids(
            cls, desc_ids: np.ndarray, prog_ids: np.ndarray
        ) -> ParticleClassificationMetrics:

        num_retained = np.isin(prog_ids, desc_ids, assume_unique=True).sum()
        
        if num_retained == 0: return cls.null()

        return cls(
            true_positives = num_retained,       
            false_positives = prog_ids.size - num_retained, # num_ejected    
            false_negatives = desc_ids.size - num_retained # num_eventually_retained
        )
    
    @classmethod
    def null(cls) -> ParticleClassificationMetrics:
        return cls(0, 0, 0)
    
    @classmethod
    def for_descendant(cls, num_desc_particles: int) -> ParticleClassificationMetrics:
        return cls(
            true_positives = num_desc_particles,
            false_positives = 0,
            false_negatives = 0
        )
    
    @property
    def is_null(self) -> bool:
        return (
            (self.true_positives == 0) and
            (self.false_positives == 0) and
            (self.false_negatives == 0)
        )
    

@define(slots=True)
class PopulationMetrics:
    group_ids: np.ndarray
    true_positives: np.ndarray
    false_positives: np.ndarray
    false_negatives: np.ndarray

    @classmethod
    def joint(cls, metrics: dict[int, ParticleClassificationMetrics]) -> PopulationMetrics:
        
        ids, tps, fps, fns = [], [], [], []

        for group_id, metric in metrics.items():
            ids.append(group_id)
            tps.append(metric.true_positives)
            fps.append(metric.false_positives)
            fns.append(metric.false_negatives)
        

        return cls(
            group_ids=np.array(ids),
            true_positives=np.array(tps),   
            false_positives=np.array(fps),
            false_negatives=np.array(fns),
        )
    
    @property
    def precision(self) -> np.ndarray:
        return np.asarray(precision(self.true_positives, self.false_positives))
    
    @property
    def recall(self) -> np.ndarray:
        return np.asarray(recall(self.true_positives, self.false_negatives))
    
    @property
    def f1(self) -> np.ndarray:
        return np.asarray(f1_score(self.precision, self.recall))
    
    @property
    def false_discovery_rate(self) -> np.ndarray:
        return 1 - self.precision
    
    @property
    def as_array(self) -> np.ndarray:
        return np.array([
            self.group_ids, 
            self.true_positives, 
            self.false_positives, 
            self.false_negatives,
            self.precision,
            self.recall,
            self.f1
        ])
    
    @property
    def total_metrics(self) -> ParticleClassificationMetrics:
        return ParticleClassificationMetrics(
            true_positives=np.nansum(self.true_positives),
            false_positives=np.nansum(self.false_positives),
            false_negatives=np.nansum(self.false_negatives)
        )
    
    @property
    def mean_metrics(self) -> ParticleClassificationMetrics:
        return ParticleClassificationMetrics(
            true_positives=int(np.nanmean(self.true_positives)),
            false_positives=int(np.nanmean(self.false_positives)),
            false_negatives=int(np.nanmean(self.false_negatives))
        )
    
    @property
    def median_metrics(self) -> ParticleClassificationMetrics:
        return ParticleClassificationMetrics(
            true_positives=int(np.nanmedian(self.true_positives)),
            false_positives=int(np.nanmedian(self.false_positives)),
            false_negatives=int(np.nanmedian(self.false_negatives))
        )
    
@define(slots=True)
class CompositionRates:
    snapshot_ids: np.ndarray
    true_positive_evo: np.ndarray
    false_positive_evo: np.ndarray
    false_negative_evo: np.ndarray
    retention_rate: np.ndarray
    purity_rate: np.ndarray
    fidelity_rate: np.ndarray

    @property
    def contamination_rate(self) -> np.ndarray:
        return 1 - self.purity_rate

    @classmethod
    def from_pop_metric_evo(
            cls, 
            pop_metrics: dict[int, PopulationMetrics],
            metric_type: str = "total"
        ) -> CompositionRates:

        if metric_type == "total":
            getmetric = lambda x: x.total_metrics
        elif metric_type == "mean":
            getmetric = lambda x: x.mean_metrics
        elif metric_type == "median":
            getmetric = lambda x: x.median_metrics
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")
        
        stacked_array = lambda attr: np.stack([
            getattr(getmetric(pop_metrics[snap_id]), attr)
            for snap_id in sorted(list(pop_metrics.keys()))
        ])

        return cls(
            snapshot_ids=np.array(sorted(pop_metrics.keys())),
            true_positive_evo=stacked_array("true_positives"),
            false_positive_evo=stacked_array("false_positives"),
            false_negative_evo=stacked_array("false_negatives"),
            # retention=recall, purity=precision, fidelity=f1 (the attrs that
            # actually exist on ParticleClassificationMetrics). The original
            # names raised AttributeError.
            retention_rate=stacked_array("recall"),
            purity_rate=stacked_array("precision"),
            fidelity_rate=stacked_array("f1")
        )
    
@define(slots=True)
class CompositionData: 
    data: dict[int, PopulationMetrics]

    @property
    def snapshot_ids(self) -> np.ndarray:
        return np.array(sorted(list(self.data.keys())))

    # @property
    # def as_array(self) -> np.ndarray:
    #     return np.stack([
    #         self.data[snap_id].as_array for snap_id in self.snapshot_ids
    #     ])

    def get_composition_rates(self, metric_type: str = "total") -> CompositionRates:
        return CompositionRates.from_pop_metric_evo(self.data, metric_type)
    
    @property
    def total_retention_rate(self) -> np.ndarray:
        composition = self.get_composition_rates("total")
        return np.column_stack([composition.snapshot_ids, composition.retention_rate])

    @property
    def mean_retention_rate(self) -> np.ndarray:
        composition = self.get_composition_rates("mean")
        return np.column_stack([composition.snapshot_ids, composition.retention_rate])

    @property
    def median_retention_rate(self) -> np.ndarray:
        composition = self.get_composition_rates("median")
        return np.column_stack([composition.snapshot_ids, composition.retention_rate])
    

def precision(tp: float | np.ndarray, fp: float | np.ndarray) -> float | np.ndarray:
    # Convert to arrays for vectorized operations
    tp_arr = np.asarray(tp, dtype=np.float64)
    fp_arr = np.asarray(fp, dtype=np.float64)
    denom = tp_arr + fp_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom > 0, tp_arr / denom, np.nan)
    return result.item() if result.ndim == 0 else result

def recall(tp: float | np.ndarray, fn: float | np.ndarray) -> float | np.ndarray:
    tp_arr = np.asarray(tp, dtype=np.float64)
    fn_arr = np.asarray(fn, dtype=np.float64)
    denom = tp_arr + fn_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom > 0, tp_arr / denom, np.nan)
    return result.item() if result.ndim == 0 else result

def f1_score(prec: float | np.ndarray, rec: float | np.ndarray) -> float | np.ndarray:
    # Note: renamed parameters to avoid shadowing the function names
    prec_arr = np.asarray(prec, dtype=np.float64)
    rec_arr = np.asarray(rec, dtype=np.float64)
    denom = prec_arr + rec_arr
    # Only compute F1 score where both prec and rec are not nan and denom > 0
    valid = (denom > 0) & ~(np.isnan(prec_arr) | np.isnan(rec_arr))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(valid, 2 * prec_arr * rec_arr / denom, np.nan)
    return result.item() if result.ndim == 0 else result


def save_population_metrics(
        filepath: Path, 
        metrics: dict[int, dict[str, PopulationMetrics]],
        snap_to_scales: dict[int, float]
    ) -> None:

    snap_ids = []
    
    with h5py.File(filepath, "w") as f:
        for snap_idx, snap_metrics in metrics.items():
            snap_ids.append(int(snap_idx))
            snap_grp = f.create_group(f"{snap_idx}")
            for mdef, pop_metrics in snap_metrics.items():
                mdef_grp = snap_grp.create_group(f"{mdef}")
                mdef_grp.create_dataset("group_ids", data=pop_metrics.group_ids)
                mdef_grp.create_dataset("true_positives", data=pop_metrics.true_positives)
                mdef_grp.create_dataset("false_positives", data=pop_metrics.false_positives)
                mdef_grp.create_dataset("false_negatives", data=pop_metrics.false_negatives)
                mdef_grp.create_dataset("precision", data=pop_metrics.precision)
                mdef_grp.create_dataset("recall", data=pop_metrics.recall)
                mdef_grp.create_dataset("f1", data=pop_metrics.f1)

        f.create_dataset(
            "scale_factors", 
            data=np.asarray([
                snap_to_scales[snap_idx] for snap_idx in sorted(snap_ids)
            ])
        )

def load_population_metrics(filepath: Path) -> dict:
    metrics = {}
    with h5py.File(filepath, "r") as f:
        loaded_scale_factors = f["scale_factors"][:]
        for snap_idx in sorted(f.keys()):
            if snap_idx == "scale_factors": continue
            snap_grp = f[f"{snap_idx}"]
            metrics[int(snap_idx)] = {
                mdef: PopulationMetrics(
                    group_ids=snap_grp[mdef]["group_ids"][:],
                    true_positives=snap_grp[mdef]["true_positives"][:],
                    false_positives=snap_grp[mdef]["false_positives"][:],
                    false_negatives=snap_grp[mdef]["false_negatives"][:]
                )
                for mdef in snap_grp.keys()
            }

    return {"population_metrics": metrics, "scale_factors": loaded_scale_factors}


# ---------------------------------------------------------------------------
# Loading retention HDF5 files into analysis objects.
#
# `save_composition_phusis.py` writes one retention file per mass bin per mode:
#   {sim}_retention_M{bin}.hdf5                    symmetric (each boundary vs
#                                                  its OWN a=100 particle set)
#   {sim}_retention_M{bin}_descbound-{mdef}.hdf5   fixed-target / "asymmetric"
#                                                  (every boundary vs the {mdef}
#                                                  a=100 reference set)
# Schema: /{snap}/{mdef}/{group_ids, true_positives, false_positives,
# false_negatives, precision, recall, f1} + /scale_factors (aligned to the
# sorted snap ids). group_ids are the DESCENDANT (reference) FOF ids and are
# constant across the progenitor snaps, so a halo is followed across time by
# its descendant id. These classes expose BOTH the population-level rate
# (median over halos + IQR, matching plot_composition_evolution.py) and the
# per-halo track (one descendant halo across a).
#
# Chapter naming: retention η = recall, purity π = precision, fidelity φ = F1.
# ---------------------------------------------------------------------------

def _parse_retention_filename(name: str) -> tuple[float, str, str | None]:
    """(mass_bin, mode, reference_mdef) from a retention filename. mode is
    'symmetric' or 'descbound'/'progbound'; reference_mdef is the fixed target
    (e.g. 'bound') or None for symmetric."""
    m = re.search(r"_retention_M([0-9]+(?:\.[0-9]+)?)", name)
    mass_bin = float(m.group(1)) if m else float("nan")
    bnd = re.search(r"_(descbound|progbound)-([A-Za-z0-9_]+)\.hdf5$", name)
    if bnd:
        return mass_bin, bnd.group(1), bnd.group(2)
    return mass_bin, "symmetric", None


@define(slots=True)
class PopulationRate:
    """Population-level retention η / purity π / fidelity φ vs scale factor for
    ONE boundary: the median over halos at each snap, with 25/75 IQR bands
    (matches plot_composition_evolution.py). Nulls (unresolved progenitors)
    are dropped before the median."""
    mdef: str
    snapshot_ids: np.ndarray
    scale_factors: np.ndarray
    n_halos: np.ndarray
    retention: np.ndarray
    purity: np.ndarray
    fidelity: np.ndarray
    retention_lo: np.ndarray
    retention_hi: np.ndarray
    purity_lo: np.ndarray
    purity_hi: np.ndarray
    fidelity_lo: np.ndarray
    fidelity_hi: np.ndarray
    # Composition-derived normalized MAH = (TP+FP)/(TP+FN) = |R_X(a)|/|R_X(a_f)|,
    # the boundary's particle-count mass history. Equals retention + FP/|ref|, so
    # mah - retention is the transient/contaminant fraction.
    mah: np.ndarray = field(factory=lambda: np.array([]))
    mah_lo: np.ndarray = field(factory=lambda: np.array([]))
    mah_hi: np.ndarray = field(factory=lambda: np.array([]))
    mode: str = "symmetric"
    reference_mdef: str | None = None
    mass_bin: float = float("nan")

    # Classifier-name aliases.
    @property
    def recall(self) -> np.ndarray: return self.retention
    @property
    def precision(self) -> np.ndarray: return self.purity
    @property
    def f1(self) -> np.ndarray: return self.fidelity
    @property
    def contamination(self) -> np.ndarray: return 1.0 - self.purity


@define(slots=True)
class HaloTrack:
    """Retention track of ONE descendant halo (one boundary) across scale
    factor. tp/fp/fn are float arrays aligned with snapshot_ids; NaN where the
    halo has no entry at that snap. η/π/φ are computed per snap."""
    group_id: int
    mdef: str
    snapshot_ids: np.ndarray
    scale_factors: np.ndarray
    true_positives: np.ndarray
    false_positives: np.ndarray
    false_negatives: np.ndarray
    mode: str = "symmetric"
    reference_mdef: str | None = None
    mass_bin: float = float("nan")

    @property
    def retention(self) -> np.ndarray:
        return np.asarray(recall(self.true_positives, self.false_negatives))
    @property
    def purity(self) -> np.ndarray:
        return np.asarray(precision(self.true_positives, self.false_positives))
    @property
    def fidelity(self) -> np.ndarray:
        return np.asarray(f1_score(self.purity, self.retention))
    @property
    def mah(self) -> np.ndarray:
        """Composition-derived normalized MAH for this halo:
        (TP+FP)/(TP+FN) = |R_X(a)|/|R_X(a_f)|. = retention + FP/|ref|."""
        within = self.true_positives + self.false_positives
        ref = self.true_positives + self.false_negatives
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ref > 0, within / ref, np.nan)
    # Classifier-name aliases.
    @property
    def recall(self) -> np.ndarray: return self.retention
    @property
    def precision(self) -> np.ndarray: return self.purity
    @property
    def f1(self) -> np.ndarray: return self.fidelity

    @property
    def n_reference(self) -> np.ndarray:
        """|S_ref| = TP + FN, the descendant's eventual particle count
        (constant across snaps where the halo is present)."""
        return self.true_positives + self.false_negatives

    @property
    def is_present(self) -> np.ndarray:
        return np.isfinite(self.true_positives)


@define(slots=True)
class RetentionCatalog:
    """One loaded retention file (one mass bin, one mode). Holds per-(snap,
    mdef) PopulationMetrics and exposes population rates + per-halo tracks.

    Use `max_snap_id` to cap the descendant/reference at a chosen snapshot
    (e.g. 74 = a=100) and drop everything after it."""
    mode: str
    mass_bin: float
    descendant_snap: int
    snapshot_ids: np.ndarray
    scale_factors: np.ndarray
    metrics: dict = field(factory=dict)          # snap -> {mdef: PopulationMetrics}
    reference_mdef: str | None = None
    source_path: str = ""

    @classmethod
    def from_file(
            cls,
            filepath: str | Path,
            max_snap_id: int | None = None,
            mdefs: list[str] | None = None,
        ) -> RetentionCatalog:
        filepath = Path(filepath)
        mass_bin, mode, reference_mdef = _parse_retention_filename(filepath.name)
        metrics: dict[int, dict[str, PopulationMetrics]] = {}
        with h5py.File(filepath, "r") as f:
            all_snaps = sorted(int(k) for k in f.keys() if k != "scale_factors")
            scales = np.asarray(f["scale_factors"][:], dtype=np.float64)
            sf_by_snap = {s: float(scales[i]) for i, s in enumerate(all_snaps)}
            keep = [s for s in all_snaps
                    if max_snap_id is None or s <= int(max_snap_id)]
            for s in keep:
                grp = f[str(s)]
                use = mdefs if mdefs is not None else list(grp.keys())
                metrics[s] = {
                    m: PopulationMetrics(
                        group_ids=np.asarray(grp[m]["group_ids"][:]),
                        true_positives=np.asarray(grp[m]["true_positives"][:]),
                        false_positives=np.asarray(grp[m]["false_positives"][:]),
                        false_negatives=np.asarray(grp[m]["false_negatives"][:]),
                    )
                    for m in use if m in grp
                }
        snaps = np.asarray(keep, dtype=np.int64)
        scale_factors = np.asarray([sf_by_snap[s] for s in keep], dtype=np.float64)
        descendant_snap = int(keep[-1]) if keep else -1
        return cls(
            mode=mode, mass_bin=mass_bin, descendant_snap=descendant_snap,
            snapshot_ids=snaps, scale_factors=scale_factors, metrics=metrics,
            reference_mdef=reference_mdef, source_path=str(filepath),
        )

    @property
    def mdefs(self) -> list[str]:
        return sorted({m for snap in self.metrics.values() for m in snap})

    def group_ids(self, mdef: str) -> np.ndarray:
        """Trackable descendant FOF ids for `mdef` (the reference sample at the
        descendant snap)."""
        return self.metrics[self.descendant_snap][mdef].group_ids

    def population_rate(
            self, mdef: str, percentiles: tuple[float, float] = (25.0, 75.0),
        ) -> PopulationRate:
        n = len(self.snapshot_ids)
        out = {k: np.full(n, np.nan) for k in
               ("retention", "purity", "fidelity", "mah",
                "retention_lo", "retention_hi", "purity_lo", "purity_hi",
                "fidelity_lo", "fidelity_hi", "mah_lo", "mah_hi")}
        n_halos = np.zeros(n, dtype=np.int64)
        for i, s in enumerate(self.snapshot_ids.tolist()):
            pm = self.metrics.get(s, {}).get(mdef)
            if pm is None:
                continue
            tp = pm.true_positives.astype(np.float64)
            fp = pm.false_positives.astype(np.float64)
            fn = pm.false_negatives.astype(np.float64)
            nonnull = (tp + fp + fn) > 0
            if not nonnull.any():
                continue
            rec = np.asarray(recall(tp, fn))[nonnull]
            prec = np.asarray(precision(tp, fp))[nonnull]
            fid = np.asarray(f1_score(prec, rec))
            # composition MAH = |R_X(a)|/|R_X(a_f)| = (TP+FP)/(TP+FN)
            ref_ct = (tp + fn)[nonnull]
            within = (tp + fp)[nonnull]
            with np.errstate(invalid="ignore", divide="ignore"):
                mah = np.where(ref_ct > 0, within / ref_ct, np.nan)
            n_halos[i] = int(nonnull.sum())
            out["retention"][i] = np.nanmedian(rec)
            out["purity"][i] = np.nanmedian(prec)
            out["fidelity"][i] = np.nanmedian(fid)
            out["mah"][i] = np.nanmedian(mah)
            out["retention_lo"][i], out["retention_hi"][i] = np.nanpercentile(rec, percentiles)
            out["purity_lo"][i], out["purity_hi"][i] = np.nanpercentile(prec, percentiles)
            out["fidelity_lo"][i], out["fidelity_hi"][i] = np.nanpercentile(fid, percentiles)
            out["mah_lo"][i], out["mah_hi"][i] = np.nanpercentile(mah, percentiles)
        return PopulationRate(
            mdef=mdef, snapshot_ids=self.snapshot_ids, scale_factors=self.scale_factors,
            n_halos=n_halos, mode=self.mode, reference_mdef=self.reference_mdef,
            mass_bin=self.mass_bin, **out,
        )

    def halo_track(self, group_id: int, mdef: str) -> HaloTrack:
        n = len(self.snapshot_ids)
        tp = np.full(n, np.nan); fp = np.full(n, np.nan); fn = np.full(n, np.nan)
        gid = int(group_id)
        for i, s in enumerate(self.snapshot_ids.tolist()):
            pm = self.metrics.get(s, {}).get(mdef)
            if pm is None:
                continue
            gids = pm.group_ids
            j = int(np.searchsorted(gids, gid))   # group_ids are sorted ascending
            if j < len(gids) and int(gids[j]) == gid:
                tp[i] = float(pm.true_positives[j])
                fp[i] = float(pm.false_positives[j])
                fn[i] = float(pm.false_negatives[j])
        return HaloTrack(
            group_id=gid, mdef=mdef, snapshot_ids=self.snapshot_ids,
            scale_factors=self.scale_factors, true_positives=tp,
            false_positives=fp, false_negatives=fn, mode=self.mode,
            reference_mdef=self.reference_mdef, mass_bin=self.mass_bin,
        )

    def population_metrics_at(self, snap_id: int, mdef: str) -> PopulationMetrics:
        """The raw per-halo PopulationMetrics at one snapshot (population-level
        analysis at a single epoch)."""
        return self.metrics[int(snap_id)][mdef]


def load_retention_catalogs(
        directory: str | Path,
        mass_bin: float,
        max_snap_id: int | None = None,
        reference_mdef: str = "bound",
        mdefs: list[str] | None = None,
    ) -> dict[str, RetentionCatalog]:
    """Load the symmetric AND fixed-target (descbound-<reference_mdef>) retention
    catalogs for one mass bin from a composition directory. Returns
    {'symmetric': cat, 'descbound-<ref>': cat}; missing variants are omitted."""
    directory = Path(directory)
    tag = f"M{mass_bin}"
    out: dict[str, RetentionCatalog] = {}
    sym = sorted(directory.glob(f"*_retention_{tag}.hdf5"))
    if sym:
        out["symmetric"] = RetentionCatalog.from_file(sym[0], max_snap_id, mdefs)
    asym = sorted(directory.glob(f"*_retention_{tag}_descbound-{reference_mdef}.hdf5"))
    if asym:
        out[f"descbound-{reference_mdef}"] = RetentionCatalog.from_file(
            asym[0], max_snap_id, mdefs)
    if not out:
        raise FileNotFoundError(
            f"No retention files for {tag} under {directory} "
            f"(looked for symmetric and descbound-{reference_mdef}).")
    return out


def load_retention_catalog_tree(
        paths: dict[str, dict[str, str | Path]],
        max_snap_id: int | None = 74,
        mdefs: list[str] | None = None,
    ) -> dict[str, dict[str, RetentionCatalog]]:
    """Map a nested dict of retention-file paths to a matching nested dict of
    loaded ``RetentionCatalog`` instances, preserving the keys.

    Works for an arbitrary 2-level structure, e.g.
        {"symmetric": {"dwarf": p32, "galactic": p128, "cluster": p2048},
         "asymmetric": {"dwarf": ..., ...}}
    -> {"symmetric": {"dwarf": RetentionCatalog, ...}, "asymmetric": {...}}.

    ``max_snap_id`` caps the descendant/reference snapshot (74 = a=100, dropping
    any later/spurious snaps); pass None to keep every snapshot. ``mdefs``
    optionally restricts which boundaries are loaded. Each leaf value may be a
    str or Path; the file's own mode/mass_bin/reference_mdef are parsed from its
    name, so the loaded catalog carries them regardless of the outer keys.
    """
    return {
        group: {
            label: RetentionCatalog.from_file(path, max_snap_id=max_snap_id, mdefs=mdefs)
            for label, path in inner.items()
        }
        for group, inner in paths.items()
    }