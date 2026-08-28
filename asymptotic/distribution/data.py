from __future__ import annotations

import h5py
import numpy as np

from attrs import define, field
from abc import ABC, abstractmethod

from functools import cached_property

@define(slots=False)
class DistributionData(ABC):
    bin_count: int 
    num_objects: int
    radii: np.ndarray
    weights: np.ndarray

    radial_bins: dict[str, np.ndarray]

    def __attrs_post_init__(self) -> None:
        if self.num_objects < 1:
            raise ValueError("Number of objects must be greater than 0")

    def get_radii(self, radii_units: float) -> np.ndarray:
        return self.radii * radii_units

    @cached_property
    def radial_shell_bins(self) -> np.ndarray:
        outer_volume = self.radial_bins["edges"][1:]**3
        inner_volume = self.radial_bins["edges"][:-1]**3
        return (outer_volume - inner_volume)

    def volume_bins(self, volume_factor: float) -> np.ndarray:
        return self.radial_shell_bins * volume_factor

    @abstractmethod
    def histogram(self, *args, **kwargs) -> np.ndarray:
        ...

    @abstractmethod
    def update_radial_bins(self, bin_count: int, *args, **kwargs) -> DistributionData:
        ... 
        
    @classmethod
    @abstractmethod
    def from_hdf5_group(cls, hdf5_group: h5py.Group) -> DistributionData:
        ...