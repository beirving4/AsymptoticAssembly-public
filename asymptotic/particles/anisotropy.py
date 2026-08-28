from __future__ import annotations

import numpy as np

from typing import Iterable
from attrs import define, field

from .orbital import cartesioan_to_spherical_velocities

@define(slots=True)
class AnisotropyParameters:
    major_axis_variance: float
    intermediate_axis_variance: float
    minor_axis_variance: float

    @property
    def tangential_variance(self) -> float:
        return self.intermediate_axis_variance + self.minor_axis_variance

    @property
    def anisotropy(self) -> float:
        return 1.0 - 0.5 * (self.tangential_variance / self.major_axis_variance)
    
    
@define(slots=True)
class AnisotropyParameterData:
    spherical: AnisotropyParameters
    principal: AnisotropyParameters | None = field(default=None)

    @property
    def radial_variance(self) -> float:
        return self.spherical.major_axis_variance
    
    @property
    def polar_variance(self) -> float:
        return self.spherical.intermediate_axis_variance
    
    @property
    def azimuthal_variance(self) -> float:
        return self.spherical.minor_axis_variance
    
    @property
    def radial(self) -> float:
        return self.spherical.anisotropy
    
    @property
    def cylindrical(self) -> float:
        if self.principal is None:
            raise ValueError("Principal anisotropy parameters are not set.")
        return self.principal.anisotropy 
    

    @classmethod
    def from_halo(
            cls,
            velocities: np.ndarray,
            spherical_coordinates: np.ndarray
        ) -> AnisotropyParameterData:

        spherical_velocities = cartesioan_to_spherical_velocities(
            cartiesian_velocities=velocities,
            spherical_coordinates=spherical_coordinates
        )
        
        orbitals = get_orbital_velocity_dispersions(spherical_velocities)
        T_ij = get_velocity_dispersion_tensor(velocities)
        principals = get_principal_velocity_dispersions(T_ij)

        return cls(
            spherical=AnisotropyParameters(
                major_axis_variance=orbitals[0],
                intermediate_axis_variance=orbitals[1],
                minor_axis_variance=orbitals[2]
            ),
            principal=AnisotropyParameters(
                major_axis_variance=principals[0],
                intermediate_axis_variance=principals[1],
                minor_axis_variance=principals[2]
            )
        )
    
@define(slots=True)
class PopulationAnisotropyParameters:
    major_axis_variances: np.ndarray
    intermediate_axis_variances: np.ndarray
    minor_axis_variances: np.ndarray

    def __getitem__(self, index: int) -> AnisotropyParameters:
        return AnisotropyParameters(
            major_axis_variance=self.major_axis_variances[index],
            intermediate_axis_variance=self.intermediate_axis_variances[index],
            minor_axis_variance=self.minor_axis_variances[index]
        )
    
    def __len__(self) -> int:
        return len(self.major_axis_variances)
    
    def __iter__(self) -> Iterable[AnisotropyParameters]:
        for i in range(len(self)):
            yield self[i]
    

    @property
    def tangential_variances(self) -> np.ndarray:
        return self.intermediate_axis_variances + self.minor_axis_variances
    
    @property
    def anisotropies(self) -> np.ndarray:
        return 1.0 - 0.5 * (self.tangential_variances / self.major_axis_variances)
    
    @property
    def to_numpy(self) -> np.ndarray:
        return np.column_stack([
            self.major_axis_variances,
            self.intermediate_axis_variances,
            self.minor_axis_variances,
            self.tangential_variances,
            self.anisotropies
        ])
    
@define(slots=True)
class PopulationAnisotropyParameterData:
    spherical: PopulationAnisotropyParameters
    principal: PopulationAnisotropyParameters | None = field(default=None)

    def __getitem__(self, index: int) -> AnisotropyParameterData:
        return AnisotropyParameterData(
            spherical=self.spherical[index],
            principal=self.principal[index] if self.principal is not None else None
        )
    
    def __len__(self) -> int:
        return len(self.spherical)
    
    def __iter__(self) -> Iterable[AnisotropyParameterData]:
        for i in range(len(self)):
            yield self[i]

    @property
    def radial_anisotropies(self) -> np.ndarray:
        return self.spherical.anisotropies
    
    @property
    def cylindrical_anisotropies(self) -> np.ndarray:
        if self.principal is None:
            raise ValueError("Principal anisotropy parameters are not set.")
        return self.principal.anisotropies
    
    @property
    def to_numpy(self) -> np.ndarray:
        if self.principal is None:
            return self.spherical.to_numpy  
        return np.column_stack([self.spherical.to_numpy, self.principal.to_numpy])
    
    @classmethod
    def from_physical_catalog(cls, physical_catalog: dict) -> PopulationAnisotropyParameterData:
        ...


@define(slots=True)
class AnisotropyProfile: 
    major_axis_variances: np.ndarray
    intermediate_axis_variances: np.ndarray
    minor_axis_variances: np.ndarray

    @property
    def tangential_variances(self) -> np.ndarray:
        return self.intermediate_axis_variances + self.minor_axis_variances
    
    @property
    def anisotropies(self) -> np.ndarray:
        return 1.0 - 0.5 * (self.tangential_variances / self.major_axis_variances)
    
    @property
    def to_numpy(self) -> np.ndarray:
        return np.column_stack([
            self.major_axis_variances,
            self.intermediate_axis_variances,
            self.minor_axis_variances,
            self.tangential_variances,
            self.anisotropies
        ])
    

@define(slots=True)
class AnisotropyProfileData:
    radii: np.ndarray
    spherical: AnisotropyProfile
    principal: AnisotropyProfile | None = field(default=None)

    @property
    def to_numpy(self) -> np.ndarray:
        if self.principal is None:
            return np.column_stack([self.radii, self.spherical.to_numpy])
        else:
            return np.column_stack([
                self.radii, self.spherical.to_numpy, self.principal.to_numpy
            ])

def get_velocity_dispersion_tensor(velocities: np.ndarray) -> np.ndarray:
    n = velocities.shape[0]
    if n < 2:
        raise ValueError(
            "At least two velocities are required to compute the dispersion tensor."
        )
    if velocities.shape[1] != 3:
        raise ValueError("Velocities must be 3D vectors.")
    return np.einsum('ki,kj->ij', velocities, velocities) / (n - 1)

def get_orbital_velocity_dispersions(velocities: np.ndarray) -> np.ndarray:
    if velocities.shape[0] < 2:
        raise ValueError(
            "At least two velocities are required to compute the dispersion tensor."
        )
    if velocities.shape[1] != 3:
        raise ValueError("Velocities must be 3D vectors.")
    return np.nanvar(velocities, axis=0)

def get_principal_velocity_dispersions(dispersion_tensor: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvals(dispersion_tensor))[::-1]

def get_total_velocity_variance(velocities: np.ndarray) -> float:
    if velocities.shape[0] < 2:
        raise ValueError(
            "At least two velocities are required to compute the variance."
        )
    if velocities.shape[1] != 3:
        raise ValueError("Velocities must be 3D vectors.")
    return np.nanvar(velocities, axis=0).sum()

def radial_anisotropy(v_tan: np.ndarray, v_rad: np.ndarray) -> float:
    tangential_variance = get_total_velocity_variance(v_tan)
    radial_variance = get_total_velocity_variance(v_rad)
    return 1 - 0.5 * (tangential_variance / radial_variance)

def cylindrical_anisotropy(velocities: np.ndarray) -> float:
    V_ij = get_velocity_dispersion_tensor(velocities)
    var_1, var_2, var_3 = get_principal_velocity_dispersions(V_ij)
    return 1 - 0.5 * (var_2 + var_3) / var_1


def compute_anisotropy_profile(
        velocities: np.ndarray,
        spherical_coordinates: np.ndarray,   
        radial_bins: np.ndarray,
    ) -> dict[str, np.ndarray]:

    
    bin_indices = np.digitize(spherical_coordinates[:, 0], radial_bins) - 1 
    bin_centers = np.sqrt(radial_bins[1:] * radial_bins[:-1])

    spherical_velocities = cartesioan_to_spherical_velocities(
        cartiesian_velocities=velocities,
        spherical_coordinates=spherical_coordinates
    )

    nonzero_r_bins = []
    vr_vars, v_theta_vars, v_phi_vars = [], [], []
    v1_vars, v2_vars, v3_vars = [], [], []

    for i, r in enumerate(bin_centers):

        velocity_bin_weights = velocities[bin_indices == i]
        spherical_bin_weights = spherical_velocities[bin_indices == i]

        if spherical_bin_weights.shape[0] > 1:

            nonzero_r_bins.append(r)
            orbitals = get_orbital_velocity_dispersions(spherical_bin_weights)
            T_ij = get_velocity_dispersion_tensor(velocity_bin_weights)
            principals = get_principal_velocity_dispersions(T_ij)

            vr_vars.append(orbitals[0])
            v_theta_vars.append(orbitals[1])
            v_phi_vars.append(orbitals[2])
            v1_vars.append(principals[0])
            v2_vars.append(principals[1])
            v3_vars.append(principals[2])


    return {
        'radial_bins' : np.asarray(nonzero_r_bins),
        'vr_vars' : np.asarray(vr_vars),
        'v_theta_vars' : np.asarray(v_theta_vars),
        'v_phi_vars' : np.asarray(v_phi_vars),
        'v1_vars' : np.asarray(v1_vars),
        'v2_vars' : np.asarray(v2_vars),
        'v3_vars' : np.asarray(v3_vars),
    }