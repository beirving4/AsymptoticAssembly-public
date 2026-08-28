from __future__ import annotations

import numpy as np

from typing import Iterable
from attrs import define, field

@define(slots=True)
class ShapeParameters: 
    major_axis: float
    intermediate_axis: float
    minor_axis: float

    @property
    def axis_lengths(self) -> np.ndarray:
        return np.array([self.major_axis, self.intermediate_axis, self.minor_axis])
    
    @property
    def sphericity(self) -> float:
        return self.minor_axis / self.major_axis
    
    @property
    def polar_flattening(self) -> float:
        return self.minor_axis / self.intermediate_axis
    
    @property
    def equatorial_flattening(self) -> float:
        return self.intermediate_axis / self.major_axis
    
    @property
    def triaxiality(self) -> float:
        return (1 - self.equatorial_flattening**2) / (1 - self.sphericity**2)
    
    def get_elliptical_radii(self, positions: np.ndarray) -> np.ndarray:
        return get_elliptical_radii(positions, self.axis_lengths)
    
    
@define(slots=True)
class ShapeParameterData:
    base: ShapeParameters
    reduced: ShapeParameters | None = field(default=None) 


    def add_reduced_shape_parameters(self, positions: np.ndarray) -> None:
        S_ij = reduced_inertia_tensor(positions, self.base.axis_lengths)
        reduced_axis_lengths = get_axis_lengths(S_ij)
        self.reduced = ShapeParameters(
            major_axis=reduced_axis_lengths[0],
            intermediate_axis=reduced_axis_lengths[1],
            minor_axis=reduced_axis_lengths[2]
        )

    @classmethod
    def from_coordinates(cls, positions: np.ndarray) -> ShapeParameterData:
        I_ij = get_intertia_tensor(positions)
        axis_lengths = get_axis_lengths(I_ij)
        S_ij = reduced_inertia_tensor(positions, axis_lengths)
        reduced_axis_lengths = get_axis_lengths(S_ij)
        return cls(
            base=ShapeParameters(
                major_axis=axis_lengths[0],
                intermediate_axis=axis_lengths[1],
                minor_axis=axis_lengths[2]
            ),
            reduced=ShapeParameters(
                major_axis=reduced_axis_lengths[0],
                intermediate_axis=reduced_axis_lengths[1],
                minor_axis=reduced_axis_lengths[2]
            )
        )


@define(slots=True)
class PopulationShapeParameters: 
    major_axes: np.ndarray
    intermediate_axes: np.ndarray
    minor_axes: np.ndarray

    def __getitem__(self, index: int) -> ShapeParameters:
        return ShapeParameters(
            major_axis=self.major_axes[index],
            intermediate_axis=self.intermediate_axes[index],
            minor_axis=self.minor_axes[index]
        )
    
    def __len__(self) -> int:
        return len(self.major_axes)
    
    def __iter__(self) -> Iterable[ShapeParameters]:
        for i in range(len(self)):
            yield self[i]   

    @property
    def axis_lengths(self) -> np.ndarray:
        return np.column_stack([
            self.major_axes, 
            self.intermediate_axes, 
            self.minor_axes
        ])
    
    @property
    def sphericities(self) -> np.ndarray:
        return self.minor_axes / self.major_axes
    
    @property
    def polar_flattenings(self) -> np.ndarray:
        return self.minor_axes / self.intermediate_axes
    
    @property
    def equatorial_flattenings(self) -> np.ndarray:
        return self.intermediate_axes / self.major_axes
    
    @property
    def triaxialities(self) -> np.ndarray:
        return (1 - self.equatorial_flattenings**2) / (1 - self.sphericities**2)
    

@define(slots=True)
class PopulationShapeParameterData:
    base: PopulationShapeParameters
    reduced: PopulationShapeParameters | None = field(default=None) 

    def __getitem__(self, index: int) -> ShapeParameterData:
        return ShapeParameterData(
            base=self.base[index], 
            reduced=self.reduced[index] if self.reduced is not None else None
        )
    
    def __len__(self) -> int:
        return len(self.base)
    
    def __iter__(self) -> Iterable[ShapeParameterData]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_physical_catalog(cls, physical_catalog: dict) -> PopulationShapeParameterData:
        ...

# No need for an evo class because we can pull this from the halo populations evo instead
@define(slots=True)
class ShapeProfile:
    major_axes: np.ndarray
    intermediate_axes: np.ndarray
    minor_axes: np.ndarray

    @property
    def sphericities(self) -> np.ndarray:
        return self.minor_axes / self.major_axes
    
    @property
    def polar_flattenings(self) -> np.ndarray:
        return self.minor_axes / self.intermediate_axes
    
    @property
    def equatorial_flattenings(self) -> np.ndarray:
        return self.intermediate_axes / self.major_axes
    
    @property
    def triaxialities(self) -> np.ndarray:
        return (1 - self.equatorial_flattenings**2) / (1 - self.sphericities**2)
    
    @property
    def to_numpy(self) -> np.ndarray:
        return np.array([ 
            self.major_axes,
            self.intermediate_axes,
            self.minor_axes,
            self.sphericities,
            self.polar_flattenings,
            self.equatorial_flattenings,
            self.triaxialities
        ]).T
    
@define(slots=True)
class ShapeProfileData:
    radii: np.ndarray
    base: ShapeProfile
    reduced: ShapeProfile | None = field(default=None)

    @classmethod
    def from_particles(
            cls, 
            positions: np.ndarray,
            radii: np.ndarray, 
            radial_bins: np.ndarray,
        ) -> ShapeProfileData:
        
        shape_profile = compute_shape_profiles(positions, radii, radial_bins)
        
        return cls(
            radii=shape_profile['radial_bins'],
            base=ShapeProfile(
                major_axes=shape_profile['major_axes'],
                intermediate_axes=shape_profile['intermediate_axes'],
                minor_axes=shape_profile['minor_axes']
            ),
            reduced=ShapeProfile(
                major_axes=shape_profile['reduced_major_axes'],
                intermediate_axes=shape_profile['reduced_intermediate_axes'],
                minor_axes=shape_profile['reduced_minor_axes']
            )
        )
    
    @property
    def to_numpy(self) -> np.ndarray:
        if self.reduced is None:
            return np.column_stack([self.radii, self.base.to_numpy])
        else:
            return np.column_stack([
                self.radii, self.base.to_numpy, self.reduced.to_numpy
            ])


def get_intertia_tensor(positions: np.ndarray) -> np.ndarray:
    return np.einsum('ki,kj->ij', positions, positions)

def get_axis_lengths(inertia_tensor: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sort(np.linalg.eigvals(inertia_tensor))[::-1])

def get_axes(axis_lengths: np.ndarray) -> tuple[float, float, float]:
    a, b, c = np.sort(axis_lengths)[::-1] # sort largest to smallest axis_lengths
    return a, b, c

def get_elliptical_radii(
        positions: np.ndarray, axis_lengths: np.ndarray
    ) -> np.ndarray:

    a, b, c = get_axes(axis_lengths)
    x, y, z = positions.T
    s, q = (c/a), (b/a)

    return np.sqrt(x**2 + (y/q)**2 + (z/s)**2)

def reduced_inertia_tensor(
        positions: np.ndarray, axis_lengths: np.ndarray
    ) -> np.ndarray:
    
    elliptical_radii = get_elliptical_radii(positions, axis_lengths)
    mask = np.nonzero(elliptical_radii)[0]
    masked_positions = positions[mask]
    masked_elliptical_radii = elliptical_radii[mask]
    weighted_r = masked_positions / masked_elliptical_radii[:, np.newaxis]
    return np.einsum('ki,kj->ij', weighted_r, weighted_r)


def compute_shape_profiles(
        positions: np.ndarray,
        radii: np.ndarray, 
        radial_bins: np.ndarray,
    ) -> dict[str, np.ndarray]:

    bin_indices = np.digitize(radii, radial_bins) - 1 
    bin_centers = np.sqrt(radial_bins[1:] * radial_bins[:-1])

    nonzero_r_bins, majors, intermediates, minors = [], [], [], []
    reduced_majors, reduced_intermediates, reduced_minors = [], [], []

    for i, r in enumerate(bin_centers):
        bin_positions = positions[bin_indices == i]
        if len(bin_positions) > 0:

            I_ij = np.einsum('ki,kj->ij', bin_positions, bin_positions)
            I_ij = get_intertia_tensor(bin_positions)
            axis_lengths = get_axis_lengths(I_ij)
            S_ij = reduced_inertia_tensor(bin_positions, axis_lengths)
            reduced_axis_lengths = get_axis_lengths(S_ij)

            nonzero_r_bins.append(r)
            majors.append(axis_lengths[0])
            intermediates.append(axis_lengths[1])
            minors.append(axis_lengths[2])
            reduced_majors.append(reduced_axis_lengths[0])
            reduced_intermediates.append(reduced_axis_lengths[1])
            reduced_minors.append(reduced_axis_lengths[2])

    return {
        'radial_bins' : np.asarray(nonzero_r_bins),
        "major_axis": np.asarray(majors),
        "intermediate_axis": np.asarray(intermediates),
        "minor_axis": np.asarray(minors),
        "reduced_major_axis": np.asarray(reduced_majors),
        "reduced_intermediate_axis": np.asarray(reduced_intermediates),
        "reduced_minor_axis": np.asarray(reduced_minors),
    }

