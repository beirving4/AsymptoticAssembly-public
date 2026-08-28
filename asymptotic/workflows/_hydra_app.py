"""Hydra application boundary.

The only module that imports Hydra or OmegaConf. It composes, converts the
dynamic configuration into typed records, configures the thread environment,
and only then imports the selected workflow — so no numerical or scientific
module is imported before the environment is set.
"""
from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from .environment import configure_thread_environment
from .registry import build_run_spec, get_entry, load_runner

#: the Hydra task wrapper discards return values, so the code is handed back here
_EXIT_CODE: list[int] = [0]


def run_composed(cfg: DictConfig) -> int:
    """Compose → typed records → environment → lazy dispatch."""
    # resolve=True and throw_on_missing=True reject unresolved '???' values
    raw: Any = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise TypeError("composed configuration must be a mapping")

    spec = build_run_spec(raw)                  # strict + typed + validated
    applied = configure_thread_environment(spec.compute)  # before any import
    if applied:
        print(f"[workflow] thread environment: {applied}")

    # build_run_spec has already matched the discriminator to the registered
    # concrete type, so the runner's narrower parameter type is satisfied
    entry = get_entry(spec.workflow.name)
    runner = load_runner(entry)                 # lazy import
    return runner(spec)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _task(cfg: DictConfig) -> None:
    _EXIT_CODE[0] = run_composed(cfg)


def main() -> int:
    _task()
    return _EXIT_CODE[0]
