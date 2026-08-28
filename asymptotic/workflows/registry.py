"""Built-in workflow registry and typed composition.

Hard-coded in package Python: a workflow name is a typed discriminator, never
an import path, and no dotted path here can come from YAML. Nothing is
imported eagerly — the selected runner is loaded only after composition,
validation, and thread-environment handling, so no scientific module is
imported before the environment is set.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from attrs import Attribute, define, field, fields

from .specs import (
    AnalysisRunSpec,
    ComputeSpec,
    CosmologySpec,
    MergeBoundaryCausticsSpec,
    OutputSpec,
    RawNode,
    ResolutionSpec,
    SimulationSpec,
    StorageSpec,
    WorkflowSpec,
    WorkflowSpecT,
)

#: the only import prefix a registry entry may name
WORKFLOW_PACKAGE = "asymptotic.workflows."

#: optional root groups a workflow may declare it requires
OPTIONAL_GROUPS: tuple[str, ...] = ("cosmology", "resolution", "storage")

#: every compute axis a workflow may declare it honours
COMPUTE_FIELDS: tuple[str, ...] = (
    "processes", "threads_per_process", "native_threads", "chunk_size")

#: a validated root whose workflow satisfies the structural bound
RunSpec: TypeAlias = AnalysisRunSpec[WorkflowSpec]
#: what a workflow module's ``run`` must be
Runner: TypeAlias = Callable[[RunSpec], int]


@define(frozen=True, slots=True, kw_only=True)
class WorkflowEntry:
    name: str = field()
    spec_type: type[WorkflowSpec] = field()
    module: str = field()
    required_groups: tuple[str, ...] = field(default=())
    #: compute axes this workflow actually honours; anything else must stay
    #: None so a run cannot appear to control parallelism while doing nothing
    supported_compute: tuple[str, ...] = field(default=())
    requires_context: bool = field(default=False)

    @module.validator
    def _check_module(self, attribute: Attribute[object], value: str) -> None:
        if not value.startswith(WORKFLOW_PACKAGE):
            raise ValueError(
                f"registry module must be under {WORKFLOW_PACKAGE!r}, got {value!r}")

    @required_groups.validator
    def _check_groups(
            self, attribute: Attribute[object], value: tuple[str, ...]
        ) -> None:
        if unknown := set(value) - set(OPTIONAL_GROUPS):
            raise ValueError(f"unknown required groups: {sorted(unknown)}")

    @supported_compute.validator
    def _check_compute(
            self, attribute: Attribute[object], value: tuple[str, ...]
        ) -> None:
        if unknown := set(value) - set(COMPUTE_FIELDS):
            raise ValueError(f"unknown compute fields: {sorted(unknown)}")


REGISTRY: Mapping[str, WorkflowEntry] = MappingProxyType({
    "merge_boundary_caustics": WorkflowEntry(
        name="merge_boundary_caustics",
        spec_type=MergeBoundaryCausticsSpec,
        module="asymptotic.workflows.merge_boundary_caustics",
        # merging existing HDF5 files needs no cosmology, resolution, storage,
        # or runtime context
        required_groups=(),
        # only the thread environment is honoured; the merge is single-process
        supported_compute=("threads_per_process",),
        requires_context=False,
    ),
})

SHARED_GROUPS: Mapping[str, type] = MappingProxyType({
    "simulation": SimulationSpec,
    "output": OutputSpec,
    "compute": ComputeSpec,
    "cosmology": CosmologySpec,
    "resolution": ResolutionSpec,
    "storage": StorageSpec,
})


def get_entry(name: object) -> WorkflowEntry:
    if not isinstance(name, str) or name not in REGISTRY:
        raise ValueError(
            f"unknown workflow {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def _build(record_type: type, node: RawNode, where: str) -> Any:
    """Construct one record, rejecting keys the record does not declare."""
    if not isinstance(node, dict):
        raise TypeError(f"{where} must be a mapping, got {type(node).__name__}")
    known = {f.name for f in fields(record_type)}
    if unknown := sorted(set(node) - known):
        raise ValueError(f"unknown keys in {where}: {unknown}")
    return record_type(**node)


def build_run_spec(raw: RawNode) -> RunSpec:
    """Typed root from a composed configuration mapping.

    Strict at every level: unknown keys, unknown workflow names, and
    name/type disagreement are all errors.
    """
    known = {"schema_version", "workflow", *SHARED_GROUPS}
    if unknown := sorted(set(raw) - known):
        raise ValueError(f"unknown keys at configuration root: {unknown}")

    workflow_node = raw.get("workflow")
    if not isinstance(workflow_node, dict) or "name" not in workflow_node:
        raise ValueError("workflow node must be a mapping carrying 'name'")
    entry = get_entry(workflow_node["name"])
    workflow = _build(entry.spec_type, workflow_node, "workflow")
    if workflow.name != entry.name:
        raise ValueError(
            f"workflow name {workflow.name!r} does not match registered "
            f"{entry.name!r}")

    groups: dict[str, Any] = {}
    for group, record_type in SHARED_GROUPS.items():
        node = raw.get(group)
        if node is None:
            continue
        groups[group] = _build(record_type, node, group)

    spec: RunSpec = AnalysisRunSpec(
        schema_version=cast(Any, raw.get("schema_version")),
        workflow=workflow,
        **groups,
    )
    validate_run_spec(spec)
    return spec


def validate_run_spec(spec: AnalysisRunSpec[WorkflowSpecT]) -> None:
    """Cross-group invariants, workflow-aware.

    Every rule keys off what the *selected* workflow declares: a group it does
    not consume imposes nothing, and a compute axis it does not honour may not
    be set to something that would silently do nothing.
    """
    entry = get_entry(spec.workflow.name)
    if type(spec.workflow) is not entry.spec_type:
        raise TypeError(
            f"workflow {entry.name!r} expects {entry.spec_type.__name__}, got "
            f"{type(spec.workflow).__name__}")

    for group in entry.required_groups:
        if getattr(spec, group) is None:
            raise ValueError(
                f"workflow {entry.name!r} requires the {group!r} group")

    # a cosmology this workflow never reads imposes no parameter file
    if ("cosmology" in entry.required_groups
            and spec.cosmology is not None
            and spec.cosmology.param_path is None
            and spec.cosmology.name in PARAM_PATH_MODES):
        raise ValueError(
            f"cosmology {spec.cosmology.name!r} reads a parameter file; "
            "set cosmology.param_path")

    unsupported = [
        name for name in COMPUTE_FIELDS
        if name not in entry.supported_compute
        and getattr(spec.compute, name) is not None
    ]
    if unsupported:
        raise ValueError(
            f"workflow {entry.name!r} does not honour compute "
            f"{sorted(unsupported)}; leave them null "
            f"(supported: {sorted(entry.supported_compute) or 'none'})")


#: cosmology names whose model construction consumes ``param_path``
PARAM_PATH_MODES = frozenset({"primary", "toy_model_a", "planck"})


def load_runner(entry: WorkflowEntry) -> Runner:
    """Import the registered module and return its ``run``. Called only after
    the thread environment is configured."""
    if not entry.module.startswith(WORKFLOW_PACKAGE):
        raise ValueError(f"refusing to import {entry.module!r}")
    module = importlib.import_module(entry.module)
    runner = getattr(module, "run", None)
    if not callable(runner):
        raise TypeError(f"{entry.module} has no callable run()")
    return cast(Runner, runner)
