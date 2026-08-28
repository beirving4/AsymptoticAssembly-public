"""Packaged neutral configuration tree for ``asymptotic-run``.

Hydra resolves ``config_path="conf"`` as an importable config module, so this
marker file is what makes the packaged tree discoverable from an installed
wheel regardless of the working directory.
"""
