from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple, Optional, Dict, Any, List

import numpy as np, pdb
from treecorr import NNCorrelation


# -----------------------------
# Stitch plan
# -----------------------------

@dataclass(frozen=True)
class StitchPlan:
    """
    Maps each output bin -> (source_index, source_bin_index).
    """
    source_index: np.ndarray      # shape (M,)
    source_bin_index: np.ndarray  # shape (M,)

    def __post_init__(self) -> None:
        if self.source_index.shape != self.source_bin_index.shape:
            raise ValueError("source_index and source_bin_index must have the same shape.")
        if self.source_index.ndim != 1:
            raise ValueError("StitchPlan arrays must be 1D.")


def stitch_1d_from_plan(arrs: Sequence[np.ndarray], plan: StitchPlan) -> np.ndarray:
    """
    Generic stitching for 1D arrays.
    """
    m = plan.source_index.size
    out = np.empty(m, dtype=np.asarray(arrs[0]).dtype)
    for k in range(m):
        si = int(plan.source_index[k])
        bi = int(plan.source_bin_index[k])
        out[k] = arrs[si][bi]
    return out


# -----------------------------
# Validation helpers
# -----------------------------

def _as_config_dict(nn: NNCorrelation) -> Dict[str, Any]:
    """Normalize TreeCorr's config into a plain dict across TreeCorr versions.

    Some TreeCorr releases store `nn.config` as a custom dict-like where `dict(nn.config)`
    can fail (e.g. raising `TypeError: 'NoneType' object is not iterable`).

    This helper tries several coercions and falls back to extracting common keys via
    `get`/attribute access.
    """
    cfg = getattr(nn, "config", None)
    if cfg is None:
        return {}

    # Common fast paths
    if isinstance(cfg, dict):
        return dict(cfg)

    try:
        # Covers true Mapping implementations
        if isinstance(cfg, Mapping):
            return dict(cfg)
    except Exception:
        pass

    # Try explicit helpers
    for meth in ("to_dict", "as_dict"):
        fn = getattr(cfg, meth, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return dict(d)
            except Exception:
                pass

    # Try .items()
    items = getattr(cfg, "items", None)
    if callable(items):
        try:
            it = items()
            if it is not None:
                return dict(it)
        except Exception:
            pass

    # Last resort: try dict(cfg), but catch and provide a helpful fallback.
    try:
        return dict(cfg)
    except Exception as e:
        out: Dict[str, Any] = {}
        common_keys = (
            "min_sep",
            "max_sep",
            "nbins",
            "bin_size",
            "period",
            "metric",
            "var_method",
            "cross_patch_weight",
            "split_method",
            "bin_type",
            "brute",
            "verbose",
            "max_top",
            "precision",
            "m2_uform",
            "num_bootstrap",
        )

        # Try `get` first if present
        getter = getattr(cfg, "get", None)
        for k in common_keys:
            v = None
            if callable(getter):
                try:
                    v = getter(k)
                except Exception:
                    v = None
            if v is None and hasattr(cfg, k):
                try:
                    v = getattr(cfg, k)
                except Exception:
                    v = None
            if v is not None:
                out[k] = v

        if out:
            return out

        raise TypeError(
            f"Unable to coerce TreeCorr NNCorrelation config into a dict. "
            f"config_type={type(cfg)!r}. Original error: {e}"
        ) from e


def _require_same_config(
    nns: Sequence[NNCorrelation],
    keys: Sequence[str],
) -> None:
    ref = _as_config_dict(nns[0])
    for j, nn in enumerate(nns[1:], start=1):
        cfg = _as_config_dict(nn)
        for k in keys:
            if ref.get(k) != cfg.get(k):
                raise ValueError(f"Config mismatch for key '{k}': "
                                 f"ref={ref.get(k)!r} vs nns[{j}]={cfg.get(k)!r}")


def _require_same_patch_keys(nns: Sequence[NNCorrelation]) -> None:
    ref_keys = set(nns[0].results.keys())
    for j, nn in enumerate(nns[1:], start=1):
        kset = set(nn.results.keys())
        if kset != ref_keys:
            missing = ref_keys - kset
            extra = kset - ref_keys
            raise ValueError(
                f"Patch results key mismatch for nns[{j}]. "
                f"missing={sorted(missing)[:10]} extra={sorted(extra)[:10]}"
            )


def _check_tot_ratios(
    dd_list: Sequence[NNCorrelation],
    rr_list: Sequence[NNCorrelation],
    dr_list: Optional[Sequence[NNCorrelation]] = None,
    rd_list: Optional[Sequence[NNCorrelation]] = None,
    rtol: float = 5e-6,
    atol: float = 0.0,
) -> None:
    """
    Enforce that the *global* normalization ratios used by TreeCorr are consistent across sources.
    TreeCorr uses dd.tot / rr.tot (and similarly vs dr/rd) as a scalar factor.

    If these ratios drift between boxes, a single stitched NNCorrelation cannot be internally
    self-consistent for covariance recomputation.
    """
    def ratios(a: Sequence[NNCorrelation], b: Sequence[NNCorrelation]) -> np.ndarray:
        ra = np.array([x.tot for x in a], dtype=float)
        rb = np.array([x.tot for x in b], dtype=float)
        if np.any(rb == 0):
            raise ValueError("Found a random-correlation object with tot=0.")
        return ra / rb

    ref_r = ratios(dd_list, rr_list)
    if not np.allclose(ref_r, ref_r[0], rtol=rtol, atol=atol):
        raise ValueError(f"dd.tot/rr.tot not consistent across sources: {ref_r}")

    if dr_list is not None:
        r = ratios(dd_list, dr_list)
        if not np.allclose(r, r[0], rtol=rtol, atol=atol):
            raise ValueError(f"dd.tot/dr.tot not consistent across sources: {r}")

    if rd_list is not None:
        r = ratios(dd_list, rd_list)
        if not np.allclose(r, r[0], rtol=rtol, atol=atol):
            raise ValueError(f"dd.tot/rd.tot not consistent across sources: {r}")


# -----------------------------
# Core stitching for NNCorrelation
# -----------------------------


def _clear_treecorr_caches(nn: NNCorrelation) -> None:
    """
    Clear TreeCorr cached lazies that can go stale after we overwrite arrays.
    """
    nn.__dict__.pop('_zero_array', None)  # lazy_property depends on weight shape
    nn.__dict__.pop('_nonzero', None)     # lazy_property depends on arrays
    nn.__dict__.pop('_ok', None)          # TreeCorr may cache ok-pairs set


def _ensure_treecorr_cached_attrs(nn: NNCorrelation) -> None:
    """Ensure TreeCorr cached attributes exist on an NNCorrelation.

    Some TreeCorr versions expect these attributes to exist during `.copy()`.
    In particular, patch objects may not have had calculateXi run, so `_xi`
    (and friends) can be missing, which can raise `AttributeError` inside
    TreeCorr's copy implementation.

    This function is intentionally conservative: it only creates attributes
    that are missing, and sets them to None.
    """
    for name in ("_xi", "_varxi", "_cov", "_rr_weight"):
        if not hasattr(nn, name):
            setattr(nn, name, None)



def _make_leaf_patch_from_template(template_patch: NNCorrelation) -> NNCorrelation:
    """
    Make a patch-result container that doesn't itself carry a .results dict, rr/dr pointers, etc.
    """
    _ensure_treecorr_cached_attrs(template_patch)

    try:
        leaf = template_patch.copy()
    except Exception:
        # Fall back to constructing a fresh object if TreeCorr's copy() is brittle
        try:
            leaf = template_patch.__class__(config=_as_config_dict(template_patch))
        except TypeError:
            leaf = template_patch.__class__(_as_config_dict(template_patch))
        _ensure_treecorr_cached_attrs(leaf)

    # Ensure patch objects are leaves (no nested patch structure).
    # TreeCorr does not require patch results-of-results.
    if hasattr(leaf, "results"):
        try:
            leaf.results = None
        except Exception:
            pass

    leaf._rr = None
    leaf._dr = None
    leaf._rd = None
    leaf._rr_weight = None
    leaf._cov = None
    leaf._varxi = None
    leaf._xi = None
    leaf.__dict__.pop("xi", None)
    leaf.__dict__.pop("varxi", None)

    # Preserve patch layout metadata from the template patch if present.
    if hasattr(template_patch, "npatch1"):
        leaf.npatch1 = int(getattr(template_patch, "npatch1"))
    if hasattr(template_patch, "npatch2"):
        leaf.npatch2 = int(getattr(template_patch, "npatch2"))

    _clear_treecorr_caches(leaf)
    return leaf


def stitch_nncorrelation(
    nns: Sequence[NNCorrelation],
    plan: StitchPlan,
    *,
    new_config: Dict[str, Any],
    template_index: int = 0,
    stitch_results: bool = True,
    prefer_leaf_results: bool = True,
) -> NNCorrelation:
    """
    Stitch NNCorrelation objects into a new NNCorrelation that can recompute covariance.

    This does NOT compute covariance by itself; it constructs a consistent object. Typically
    you will call calculateXi(rr=..., dr=..., rd=...) on the stitched DD after also stitching
    RR/DR/RD.
    """
    if len(nns) == 0:
        raise ValueError("Need at least one NNCorrelation to stitch.")

    # Some TreeCorr versions expect cached attributes to exist on objects
    # during `.copy()`. Ensure they exist on all inputs (and their patches).
    for nn in nns:
        _ensure_treecorr_cached_attrs(nn)
        if hasattr(nn, "results") and isinstance(nn.results, dict):
            for p in nn.results.values():
                _ensure_treecorr_cached_attrs(p)

    _ensure_treecorr_cached_attrs(nns[template_index])

    # Template clone to preserve TreeCorr internals without re-implementing them.
    # Some TreeCorr versions have brittle copy() implementations for patch/random objects;
    # fall back to constructing a fresh NNCorrelation if copy() fails.
    tmpl = nns[template_index]
    try:
        out = tmpl.copy()
    except Exception:
        # Construct a fresh object using the stitched config.
        try:
            out = tmpl.__class__(config=dict(new_config))
        except TypeError:
            out = tmpl.__class__(dict(new_config))
        # Ensure expected cached attrs exist.
        _ensure_treecorr_cached_attrs(out)
        # Carry over a few optional attributes if present (harmless if absent).
        for name in ("logger", "_logger", "sep_units", "metric"):
            if hasattr(tmpl, name) and not hasattr(out, name):
                try:
                    setattr(out, name, getattr(tmpl, name))
                except Exception:
                    pass

    # Overwrite config metadata to reflect stitched binning/range.
    out.config = dict(new_config)

    # Stitch the core 1D arrays.
    out.meanr = stitch_1d_from_plan([np.asarray(nn.meanr) for nn in nns], plan)
    out.meanlogr = stitch_1d_from_plan([np.asarray(nn.meanlogr) for nn in nns], plan)
    out.weight = stitch_1d_from_plan([np.asarray(nn.weight) for nn in nns], plan)
    out.npairs = stitch_1d_from_plan([np.asarray(nn.npairs) for nn in nns], plan)

    # tot is global; keep template tot by default (ratios are sanity-checked elsewhere).
    out.tot = float(tmpl.tot)

    # Clear derived quantities so TreeCorr recomputes them cleanly.
    out._rr_weight = None
    out._cov = None
    out._varxi = None
    out._xi = None
    out.__dict__.pop('xi', None)
    out.__dict__.pop('varxi', None)

    # Patch-results stitching
    if stitch_results:
        _require_same_patch_keys(nns)
        keys = list(tmpl.results.keys())
        new_results: Dict[Tuple[int, int], NNCorrelation] = {}

        for ij in keys:
            patches = [nn.results[ij] for nn in nns]
            _ensure_treecorr_cached_attrs(patches[template_index])
            # If you want leaf patches (recommended), use a leaf template.
            base_patch = _make_leaf_patch_from_template(patches[template_index]) if prefer_leaf_results else patches[template_index].copy()

            # Stitch patch arrays.
            base_patch.meanr = stitch_1d_from_plan([np.asarray(p.meanr) for p in patches], plan)
            base_patch.meanlogr = stitch_1d_from_plan([np.asarray(p.meanlogr) for p in patches], plan)
            base_patch.weight = stitch_1d_from_plan([np.asarray(p.weight) for p in patches], plan)
            base_patch.npairs = stitch_1d_from_plan([np.asarray(p.npairs) for p in patches], plan)

            # Patch tot is also global; keep template patch tot. This mainly matters if R lacks patches.
            base_patch.tot = float(patches[template_index].tot)

            # Clear caches on patch.
            base_patch._rr_weight = None
            base_patch._cov = None
            base_patch._varxi = None
            base_patch._xi = None
            base_patch.__dict__.pop('xi', None)
            base_patch.__dict__.pop('varxi', None)
            _clear_treecorr_caches(base_patch)

            new_results[ij] = base_patch

        out.results = new_results

        # Ensure these match the DD patch layout.
        out.npatch1 = int(tmpl.npatch1)
        out.npatch2 = int(tmpl.npatch2)

    _clear_treecorr_caches(out)
    return out


# -----------------------------
# High-level convenience: stitch DD + RR/DR/RD and recompute cov
# -----------------------------

@dataclass(frozen=True)
class StitchedTreeCorrBundle:
    dd: NNCorrelation
    rr: NNCorrelation
    dr: Optional[NNCorrelation]
    rd: Optional[NNCorrelation]


def _has_patch_results(nn: NNCorrelation) -> bool:
    res = getattr(nn, "results", None)
    return isinstance(res, dict) and (len(res) > 0)

def _patch_stats(nn: NNCorrelation) -> str:
    res = getattr(nn, "results", None)
    nres = len(res) if isinstance(res, dict) else 0
    np1 = getattr(nn, "npatch1", None)
    np2 = getattr(nn, "npatch2", None)
    cfg = _as_config_dict(nn)
    vm = cfg.get("var_method", None)
    return f"var_method={vm!r}, npatch1={np1}, npatch2={np2}, n_results={nres}"


def build_stitched_treecorr_bundle(
    dd_list: Sequence[NNCorrelation],
    rr_list: Sequence[NNCorrelation],
    dr_list: Optional[Sequence[NNCorrelation]],
    rd_list: Optional[Sequence[NNCorrelation]],
    plan: StitchPlan,
    *,
    template_index: int = 0,
    rtol_tot: float = 5e-6,
) -> StitchedTreeCorrBundle:
    """
    Build stitched (DD, RR, DR, RD) with consistent metadata and patch structure,
    then the caller can run dd.calculateXi(rr=rr, dr=dr, rd=rd) to get cov.

    Assumes all lists are aligned by source (same order used by the stitch plan).
    """
    if len(dd_list) != len(rr_list):
        raise ValueError("dd_list and rr_list must have the same length.")
    if dr_list is not None and len(dr_list) != len(dd_list):
        raise ValueError("dr_list must match dd_list length.")
    if rd_list is not None and len(rd_list) != len(dd_list):
        raise ValueError("rd_list must match dd_list length.")
    
    # --- Patch/covariance prerequisites ---
    # TreeCorr only produces a non-diagonal covariance for jackknife/bootstrap if patch
    # results exist on *all* correlations used by calculateXi (DD and its RR/DR/RD).
    # If any are missing patches, TreeCorr effectively falls back to a diagonal covariance.
    dd_has = _has_patch_results(dd_list[template_index])
    rr_has = _has_patch_results(rr_list[template_index])
    dr_has = _has_patch_results(dr_list[template_index]) if dr_list is not None else True
    rd_has = _has_patch_results(rd_list[template_index]) if rd_list is not None else True

    if dd_has and (not rr_has or not dr_has or not rd_has):
        raise ValueError(
            "TreeCorr patch results missing on one or more RR/DR/RD objects. "
            "This would yield a diagonal covariance. "
            f"DD({_patch_stats(dd_list[template_index])}), "
            f"RR({_patch_stats(rr_list[template_index])}), "
            f"DR({_patch_stats(dr_list[template_index]) if dr_list is not None else 'n/a'}), "
            f"RD({_patch_stats(rd_list[template_index]) if rd_list is not None else 'n/a'})"
        )

    # Validate stable config keys across sources.
    fixed_keys = [
        'metric', 'var_method', 'cross_patch_weight', 'split_method', 'bin_type',
        'brute', 'verbose', 'max_top', 'precision', 'm2_uform', 'num_bootstrap'
    ]
    _require_same_config(dd_list, fixed_keys)
    _require_same_config(rr_list, fixed_keys)
    if dr_list is not None:
        _require_same_config(dr_list, fixed_keys)
    if rd_list is not None:
        _require_same_config(rd_list, fixed_keys)

    # Check normalization ratios used by TreeCorr.
    _check_tot_ratios(dd_list, rr_list, dr_list=dr_list, rd_list=rd_list, rtol=rtol_tot)

    # Synthesize stitched config from user rules (robust version).
    dd_cfgs = [_as_config_dict(nn) for nn in dd_list]
    new_config = dict(dd_cfgs[template_index])

    def _coerce_float(x: Any) -> float | None:
        try:
            if x is None:
                return None
            v = float(x)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    max_seps = [_coerce_float(c.get("max_sep")) for c in dd_cfgs]
    max_seps = [v for v in max_seps if v is not None]
    if not max_seps:
        raise ValueError(
            "Unable to infer stitched TreeCorr config: no valid 'max_sep' found in source configs."
        )
    new_config["max_sep"] = float(np.max(max_seps))

    min_seps = [_coerce_float(c.get("min_sep")) for c in dd_cfgs]
    min_seps = [v for v in min_seps if v is not None]
    if not min_seps:
        raise ValueError(
            "Unable to infer stitched TreeCorr config: no valid 'min_sep' found in source configs."
        )
    new_config["min_sep"] = float(np.min(min_seps))

    new_config["nbins"] = int(plan.source_index.size)

    periods = [_coerce_float(c.get("period")) for c in dd_cfgs]
    periods = [v for v in periods if v is not None]
    if not periods:
        raise ValueError(
            "Unable to infer stitched TreeCorr config: no valid 'period' found in source configs."
        )
    new_config["period"] = float(np.max(periods))

    dd = stitch_nncorrelation(dd_list, plan, new_config=new_config, template_index=template_index)
    rr = stitch_nncorrelation(rr_list, plan, new_config=new_config, template_index=template_index)

    dr = None
    rd = None
    if dr_list is not None:
        dr = stitch_nncorrelation(dr_list, plan, new_config=new_config, template_index=template_index)
    if rd_list is not None:
        rd = stitch_nncorrelation(rd_list, plan, new_config=new_config, template_index=template_index)

    # Set totals for rr/dr/rd consistently with template (ratios already validated).
    rr.tot = float(rr_list[template_index].tot)
    if dr is not None:
        dr.tot = float(dr_list[template_index].tot)  # type: ignore[index]
    if rd is not None:
        rd.tot = float(rd_list[template_index].tot)  # type: ignore[index]

    # Ensure caches cleared again after tot edits.
    _clear_treecorr_caches(dd)
    _clear_treecorr_caches(rr)
    if dr is not None:
        _clear_treecorr_caches(dr)
    if rd is not None:
        _clear_treecorr_caches(rd)

    return StitchedTreeCorrBundle(dd=dd, rr=rr, dr=dr, rd=rd)


def recompute_dd_xi_and_cov(bundle: StitchedTreeCorrBundle) -> NNCorrelation:
    dd = bundle.dd
    rr = bundle.rr
    dr = bundle.dr
    rd = bundle.rd

    dd.calculateXi(rr=rr, dr=dr, rd=rd)

    # Clear cached covariance, then recompute using the configured var_method.
    dd._cov = None  # yes, private, but effective if you truly mean "recompute"
    dd._varxi = None  # optional; only if you also rely on _varxi caches

    dd._cov = dd.estimate_cov(dd.var_method)
    return dd