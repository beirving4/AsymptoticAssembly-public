"""Compatibility re-exports for the density-PDF fit.

The implementation moved to the thesis-model boundary: the equation, its
normalization and the CDF helpers are in `asymptotic.model.density_pdf`, and
trust masking, fitting and diagnostics are in
`asymptotic.model.curve.density_pdf`.

Every name below is the *identical object* as its canonical counterpart, not a
forwarding function, so existing callers and pickles keep working unchanged.
New code should import from the canonical modules.
"""
from __future__ import annotations

from ..model.curve.density_pdf import (
    DensityPdfFitResult,
    build_trust_mask,
    fit_density_pdf,
)
from ..model.density_pdf import (
    KLYPIN_EXPONENTS,
    direct_cdf_above,
    evaluate_pv,
    model_cdf_above,
    model_ln_pv,
    normalized_params,
)

__all__ = [
    "KLYPIN_EXPONENTS",
    "DensityPdfFitResult",
    "model_ln_pv",
    "normalized_params",
    "evaluate_pv",
    "model_cdf_above",
    "direct_cdf_above",
    "build_trust_mask",
    "fit_density_pdf",
]
