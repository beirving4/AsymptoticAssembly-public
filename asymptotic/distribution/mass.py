from __future__ import annotations


import h5py
import numpy as np

from attrs import define, field

from functools import cached_property

from .data import DistributionData

def get_mass_distribution_bins(
        radii: np.ndarray, lower: float, upper: float, bin_count: int
    ) -> dict[str, np.ndarray]:


    bin_edges = np.logspace(np.log10(lower), np.log10(upper), bin_count)
    median_radii = lambda r, b, i: np.median(r[(r >= b[i]) & (r < b[i + 1])])

    return {
        "edges": bin_edges,
        "centers": np.asarray([
            median_radii(radii, bin_edges, i) for i in range(bin_count - 1)
        ])
    }


@define(slots=False)
class MassDistributionData(DistributionData):
    mean_density: float
    cumulative: np.ndarray

    def get_masses(self, mass_unit: float) -> np.ndarray:
        return self.weights * mass_unit
    
    @property
    def average_weights(self) -> np.ndarray:
        return self.weights / self.num_objects

    @cached_property
    def histogram(self) -> np.ndarray:
        return np.histogram(
            self.radii, 
            bins=self.radial_bins["edges"], 
            weights=self.average_weights
        )[0]
    
    def update_radial_bins(
            self, bin_count: int, lower_bound: float, upper_bound: float
        ) -> MassDistributionData:
        
        new_radial_bins = get_mass_distribution_bins(
            self.radii, lower_bound, upper_bound, bin_count
        )
        return self.__class__(
            bin_count=bin_count,
            num_objects=self.num_objects,
            radii=self.radii,
            weights=self.weights,
            mean_density=self.mean_density,
            cumulative=self.cumulative,
            radial_bins=new_radial_bins
        )

    @property
    def density(self) -> np.ndarray:
        return self.histogram / self.radial_shell_bins

    @property
    def scaled_density(self) -> np.ndarray:
        return self.density * self.radial_bins["centers"]**2 
    
    @property
    def enclosed(self) -> np.ndarray:
        return self.cumulative
    
    def enclosed_density(self, volume_factor: float) -> np.ndarray:
        volume_shells = volume_factor * self.radii[1:]**3
        return self.enclosed[1:] / volume_shells
    
    @property
    def enclosed_overdensity(self) -> np.ndarray:
        return (self.enclosed_density(1.0) / self.mean_density) - 1.0
    
    @classmethod
    def from_hdf5_group(cls, hdf5_group: h5py.Group) -> MassDistributionData:
        return cls(
            bin_count=hdf5_group.attrs["bin_count"],
            num_objects=hdf5_group.attrs["num_objects"],
            radii=hdf5_group["radii"][()],
            weights=hdf5_group["weights"][()],
            mean_density=hdf5_group.attrs["mean_density"],
            cumulative=hdf5_group["cumulative"][()],
            radial_bins={
                "edges": hdf5_group["radial_bins/edges"][()],
                "centers": hdf5_group["radial_bins/centers"][()]
            }
        )