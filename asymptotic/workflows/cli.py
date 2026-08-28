"""Console entry point for ``asymptotic-run``.

Importable with the optional workflow dependencies absent: this module uses
only the standard library, so a core-only installation can import it and get a
clear message instead of an ImportError traceback.
"""
from __future__ import annotations

import sys

#: deterministic exit code for "the optional extra is not installed"
MISSING_EXTRA_EXIT_CODE = 3

#: top-level packages the ``workflows`` extra provides
_EXTRA_PACKAGES = frozenset({"hydra", "omegaconf"})

_INSTALL_HINT = (
    "asymptotic-run needs the optional workflow dependencies.\n"
    "Install asymptotic-assembly[workflows]."
)


def main() -> int:
    try:
        from . import _hydra_app
    except ModuleNotFoundError as exc:
        top = (exc.name or "").split(".")[0]
        if top not in _EXTRA_PACKAGES:
            # a missing internal module is a packaging bug, not a missing
            # extra; let it surface
            raise
        print(_INSTALL_HINT, file=sys.stderr)
        return MISSING_EXTRA_EXIT_CODE

    return _hydra_app.main()


if __name__ == "__main__":
    raise SystemExit(main())
