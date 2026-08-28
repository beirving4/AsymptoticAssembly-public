"""Frozen input records for a composed workflow run.

Input state only (`*Spec`); derived runtime state is built by a workflow that
needs it. Construction performs no filesystem discovery, no directory
creation, no private lookup, and no numerical import — every record is plain
data with converters and validators.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar

from attrs import Attribute, define, field

SCHEMA_VERSION: Literal[1] = 1

#: a composed configuration node as OmegaConf hands it over, before typing.
#: read-only: the dynamic boundary ends where these records begin.
RawNode: TypeAlias = Mapping[str, object]


class WorkflowSpec(Protocol):
    """Structural bound for workflow specs: a typed discriminator, nothing
    more. Deliberately not a base class — records compose, they do not
    inherit."""

    @property
    def name(self) -> str: ...


WorkflowSpecT = TypeVar("WorkflowSpecT", bound=WorkflowSpec)


# --- validators and converters ---------------------------------------------- #
#
# Annotations are not runtime checks. Every field that matters validates its
# own type, because YAML hands over whatever the user wrote.

def _require_int(attribute: Attribute[object], value: object) -> int:
    """`bool` is a subclass of `int`; it is never a valid integer field."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{attribute.name} must be an int, got {type(value).__name__}")
    return value


def positive_int(instance: object, attribute: Attribute[object], value: object) -> None:
    if _require_int(attribute, value) <= 0:
        raise ValueError(f"{attribute.name} must be > 0, got {value}")


def any_int(instance: object, attribute: Attribute[object], value: object) -> None:
    """Signed values are meaningful (seed numbers, halo ids)."""
    _require_int(attribute, value)


def optional_positive_int(
        instance: object, attribute: Attribute[object], value: object
    ) -> None:
    if value is None:
        return
    positive_int(instance, attribute, value)


def strict_bool(instance: object, attribute: Attribute[object], value: object) -> None:
    """`1` and `"true"` are not booleans here."""
    if not isinstance(value, bool):
        raise TypeError(
            f"{attribute.name} must be a bool, got {type(value).__name__}")


def optional_str(instance: object, attribute: Attribute[object], value: object) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(
            f"{attribute.name} must be a str or None, got "
            f"{type(value).__name__}")


def non_empty_str(instance: object, attribute: Attribute[object], value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{attribute.name} must be a non-empty string")


def schema_version_is_supported(
        instance: object, attribute: Attribute[object], value: object
    ) -> None:
    """Exactly the integer 1 — not ``True``, not ``1.0``."""
    if _require_int(attribute, value) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {value!r}; this build understands "
            f"{SCHEMA_VERSION!r}")


def literal_str(expected: str) -> Callable[[object, Attribute[object], object], None]:
    """Validator factory for a discriminator field's exact literal."""
    def check(instance: object, attribute: Attribute[object], value: object) -> None:
        if value != expected or not isinstance(value, str):
            raise ValueError(
                f"{attribute.name} must be {expected!r}, got {value!r}")
    return check


def optional_positive_float(
        instance: object, attribute: Attribute[object], value: float | None
    ) -> None:
    """Semantic check after conversion: rejects NaN, zero, and negatives."""
    if value is None:
        return
    if value != value:                                    # NaN
        raise ValueError(f"{attribute.name} must not be NaN")
    if value <= 0.0:
        raise ValueError(f"{attribute.name} must be > 0, got {value}")


def to_optional_float(value: float | int | None) -> float | None:
    """Coerce an optional float field, rejecting wrong types here.

    The annotation is the intended input boundary and gives the generated
    constructor an honest signature; the runtime checks stay because composed
    YAML hands over whatever the user wrote. ``bool`` is a subclass of ``int``
    and is never a valid numeric value.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a float, got {type(value).__name__}")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"expected a float, got {type(value).__name__}")


def to_path(value: str | Path) -> Path:
    """`Path` without `resolve()` — a relative path stays relative."""
    return value if isinstance(value, Path) else Path(value)


def to_optional_path(value: str | Path | None) -> Path | None:
    return None if value is None else to_path(value)


# --- shared groups ----------------------------------------------------------- #

@define(frozen=True, slots=True, kw_only=True)
class SimulationSpec:
    box_size: int = field(validator=positive_int)
    num_particles: int = field(validator=positive_int)
    seed_num: int = field(validator=any_int)
    is_batched: bool = field(default=False, validator=strict_bool)


@define(frozen=True, slots=True, kw_only=True)
class CosmologySpec:
    name: str = field(validator=non_empty_str)
    #: consumed only by the cosmology modes that read a parameter file
    param_path: Path | None = field(default=None, converter=to_optional_path)


@define(frozen=True, slots=True, kw_only=True)
class ResolutionSpec:
    min_resolved_particles: int = field(validator=positive_int)
    #: None selects the pure derivation from ``seed_num``
    link_fraction: float | None = field(
        default=None, converter=to_optional_float,
        validator=optional_positive_float)


@define(frozen=True, slots=True, kw_only=True)
class StorageSpec:
    """Explicit portable paths. Some have pure fallbacks derived from
    ``sim_data_dir``; the rest are simply required by the workflows that read
    them and stay ``None`` otherwise. Nothing here is discovered."""

    sim_data_dir: Path | None = field(default=None, converter=to_optional_path)
    full_catalog_dir: Path | None = field(default=None, converter=to_optional_path)
    power_spectra_dir: Path | None = field(default=None, converter=to_optional_path)
    analysis_dir: Path | None = field(default=None, converter=to_optional_path)
    tree_order_path: Path | None = field(default=None, converter=to_optional_path)
    prog_order_path: Path | None = field(default=None, converter=to_optional_path)
    full_tree_path: Path | None = field(default=None, converter=to_optional_path)
    raw_tree_data_path: Path | None = field(default=None, converter=to_optional_path)
    use_tree: bool = field(default=False, validator=strict_bool)


@define(frozen=True, slots=True, kw_only=True)
class ComputeSpec:
    """Every axis defaults to None meaning *preserve the workflow's current
    behavior* — imposing 1 would silently serialize runs that use every core."""

    processes: int | None = field(default=None, validator=optional_positive_int)
    threads_per_process: int | None = field(
        default=None, validator=optional_positive_int)
    native_threads: int | None = field(default=None, validator=optional_positive_int)
    chunk_size: int | None = field(default=None, validator=optional_positive_int)


@define(frozen=True, slots=True, kw_only=True)
class OutputSpec:
    """Common output state only. Resume, checkpoint, and overwrite semantics
    stay workflow-specific until equivalence is established for each one."""

    out_dir: Path = field(converter=to_path)
    run_name: str | None = field(default=None, validator=optional_str)


# --- workflow specs ---------------------------------------------------------- #

@define(frozen=True, slots=True, kw_only=True)
class MergeBoundaryCausticsSpec:
    """Merge per-snapshot boundary-caustic results for one traced halo."""

    name: Literal["merge_boundary_caustics"] = field(
        default="merge_boundary_caustics",
        validator=literal_str("merge_boundary_caustics"))
    #: no range constraint — matches the legacy `type=int, required=True`
    root_halo_id: int = field(validator=any_int)
    #: "all" keeps its special meaning: discover and merge every method present
    method: str = field(default="tessellation", validator=non_empty_str)


# --- composed root ----------------------------------------------------------- #

@define(frozen=True, slots=True, kw_only=True)
class AnalysisRunSpec(Generic[WorkflowSpecT]):
    """One composed root. Groups a workflow does not use stay ``None`` rather
    than being fabricated; the registry declares which are required."""

    schema_version: Literal[1] = field(validator=schema_version_is_supported)
    simulation: SimulationSpec = field()
    output: OutputSpec = field()
    workflow: WorkflowSpecT = field()
    compute: ComputeSpec = field(factory=ComputeSpec)
    cosmology: CosmologySpec | None = field(default=None)
    resolution: ResolutionSpec | None = field(default=None)
    storage: StorageSpec | None = field(default=None)
