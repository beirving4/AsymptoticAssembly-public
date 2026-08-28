"""Shared scientific metadata for the published thesis models.

`AsymptoticCalibration` travels with every named calibration so a coefficient
set is never anonymous: it records where the numbers came from, what they are
in, and where they stop being valid. It is public scientific metadata, not
runtime configuration — it reads no file, no environment, and no clock.
"""
from __future__ import annotations

import math
from numbers import Real

from attrs import define, field

__all__ = ["AsymptoticCalibration"]


def _as_float(value: object) -> float:
    """Accept any real scalar — Python or NumPy — and return a builtin float.

    `bool` is rejected before conversion: `True` is not a measurement. So are
    complex numbers, strings, and anything else outside `numbers.Real`.
    """
    if isinstance(value, bool):
        raise TypeError("a boolean is not a real measurement")
    if not isinstance(value, Real):
        raise TypeError(
            f"expected a real number, got {type(value).__name__}")
    return float(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else _as_float(value)


def _required_text(instance: object, attribute: object, value: object) -> None:
    """A descriptive metadata field: present, textual, and not blank."""
    name = getattr(attribute, "name", "field")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _text(instance: object, attribute: object, value: object) -> None:
    """Textual, but legitimately empty (see the `citation` policy below)."""
    name = getattr(attribute, "name", "field")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")


def _positive_epoch(instance: object, attribute: object,
                    value: float | None) -> None:
    if value is None:
        return
    name = getattr(attribute, "name", "field")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


@define(frozen=True, slots=True)
class AsymptoticCalibration:
    """Provenance and validity domain of one published coefficient set.

    Field order is the M0-frozen vocabulary and is part of the contract.

    `source` is the stable human-readable origin or category; `citation` is the
    bibliographic identifier. They are separate fields. `citation` may be empty
    **only** when the calibration is original thesis work that `source` and
    `thesis_label` already identify. That is a semantic policy about what the
    strings mean, so it is upheld where the meaning is known — in the named
    calibration factories that construct published records, and in review — not
    by a validator here, which would have to guess intent from arbitrary text.
    Every other field is required: pass an explicit empty string rather than
    omitting a key, so a missing field is always a defect.

    `reference_epoch` is the epoch the calibration was normalized at, or `None`
    when a calibration does not have one. A fit that carries this record must
    agree with it.
    """

    source: str = field(validator=_required_text)
    citation: str = field(validator=_text)
    thesis_label: str = field(validator=_required_text)
    version: str = field(validator=_text)
    independent_units: str = field(validator=_text)
    dependent_units: str = field(validator=_text)
    mass_definition: str = field(validator=_text)
    radius_definition: str = field(validator=_text)
    cosmology_scope: str = field(validator=_text)
    simulation_scope: str = field(validator=_text)
    fitted_range: str = field(validator=_text)
    reference_epoch: float | None = field(
        converter=_optional_float, validator=_positive_epoch)
    selection: str = field(validator=_text)
    limitations: str = field(validator=_text)
