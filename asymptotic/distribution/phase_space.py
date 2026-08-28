from __future__ import annotations

import h5py
import numpy as np

from numba import jit
from attrs import define, field
from functools import cached_property

from .data import DistributionData


def get_phase_space_distribution_bins(
        velocities: np.ndarray, 
        radii: np.ndarray | None = None,
        bin_edges: np.ndarray | None = None, 
        bin_count: int = 100
    ) -> dict[str, np.ndarray]:
    
    if bin_edges is None:
        if radii is None:
            raise ValueError("Must provide radii if bin_edges is not provided")
        non_zero_radii = radii[radii > 0]
        bin_edges = np.geomspace(
            np.nanmin(non_zero_radii), 
            np.nanmax(non_zero_radii), 
            bin_count
        )
    else:
        bin_count = len(bin_edges)
    
    return {
        "edges":bin_edges, # type: ignore
        "centers" : np.sqrt(bin_edges[1:] * bin_edges[:-1]), # type: ignore
        "velocity": np.linspace(
            np.nanmin(velocities), np.nanmax(velocities), bin_count
        )
    }


@jit(fastmath=True)
def get_phase_space_histogram(
        radii: np.ndarray, velocities: np.ndarray, masses: np.ndarray,
        radial_bins: np.ndarray, velocity_bins: np.ndarray,
    ) -> np.ndarray:
    return np.histogram2d(
        radii, velocities, bins=[radial_bins, velocity_bins], weights=masses
    )[0]

@jit(fastmath=True)
def compute_radial_velocity_profile(
        radii: np.ndarray, 
        velocities: np.ndarray, 
        radial_bins: np.ndarray,
    ) -> dict[str, np.ndarray]:

    bin_indices = np.digitize(radii, radial_bins) - 1 
    medians, twenty_fives, seventy_fives = [], [], []
    for i in range(len(radial_bins) - 1):
        bin_weights = velocities[bin_indices == i]
        if len(bin_weights) > 0:
            medians.append(np.nanmedian(bin_weights))
            twenty_fives.append(np.nanpercentile(bin_weights, 25))
            seventy_fives.append(np.nanpercentile(bin_weights, 75))

    return {
        "median": np.asarray(medians),
        "25pct": np.asarray(twenty_fives),
        "75pct": np.asarray(seventy_fives),
    }


@define(slots=False)
class PhaseSpaceDistributionData(DistributionData):
    critical_overdenisty: float

    velocity_bins: np.ndarray
    _histogram: np.ndarray | None = field(default=None)

    def get_radial_velocities(self, velocity_unit: float) -> np.ndarray:
        return self.weights * velocity_unit
    
    def get_peculiar_velocities(self, velocity_unit: float) -> np.ndarray:
        peculiar_weights = self.weights - self.hubble * self.radii
        return peculiar_weights * velocity_unit
    
    @property
    def hubble(self) -> float:
        ''' Hubble constant in critical overdenisty units '''
        return np.sqrt(2.0 /self.critical_overdenisty)
    
    @property
    def histogram(self) -> np.ndarray | None:
        return self._histogram
    
    @histogram.setter
    def histogram(self, value: np.ndarray) -> None:
        self._histogram = value

    def compute_histogram2D(self, mass_weights: np.ndarray) -> None:
        self.histogram = np.histogram2d(
            self.radii,
            self.weights,
            bins=(self.radial_bins["edges"], self.velocity_bins), 
            weights=mass_weights
        )[0]
    
    def update_radial_bins(self, bin_count: int) -> PhaseSpaceDistributionData:
        new_bins = get_phase_space_distribution_bins(
            self.radii, self.weights, bin_count
        )
        return self.__class__(
            bin_count=bin_count,
            num_objects=self.num_objects,
            radii=self.radii,
            weights=self.weights,
            critical_overdenisty=self.critical_overdenisty,
            radial_bins={
                "edges": new_bins["edges"], "centers": new_bins["centers"]
            },
            velocity_bins=new_bins["velocity"]
        )
            

    @classmethod
    def from_hdf5_group(cls, hdf5_group: h5py.Group) -> PhaseSpaceDistributionData:
        return cls(
            bin_count=hdf5_group.attrs["bin_count"],
            num_objects=hdf5_group.attrs["num_objects"],
            radii=np.asarray(hdf5_group["radii"]),
            weights=np.asarray(hdf5_group["weights"]), 
            critical_overdenisty=hdf5_group.attrs["critical_overdenisty"],
            radial_bins={"edges": np.asarray(hdf5_group["radial_bins/edges"])},
            velocity_bins=np.asarray(hdf5_group["velocity_bins"])
        )
    
    @property
    def distribution(self) -> np.ndarray:
        return self.histogram / self.radial_shell_bins[:, None]
    
    @cached_property
    def profile(self) -> dict[str, np.ndarray]:
        return compute_radial_velocity_profile(
            radii=self.radii, 
            velocities=self.weights, 
            radial_bins=self.radial_bins["edges"]
        )

    @property
    def radial_to_hubble_ratio(self) -> np.ndarray:
        return self.profile["median"] / self.hubble
    
    @property
    def peculiar_velocity_profile(self) -> np.ndarray:
        return self.profile["median"] - self.hubble * self.radial_bins["centers"]