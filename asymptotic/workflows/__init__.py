"""Composed workflow configuration and runners (H1 series).

Application-boundary code only: YAML composition and typed input records live
here, scientific algorithms do not. Importing this package must not import
NumPy, SciPy, h5py, Tessera, or Hydra — the runners import what they need
after the thread environment is configured.
"""
from __future__ import annotations
