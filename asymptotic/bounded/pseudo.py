from __future__ import annotations

import numpy as np

from attrs import define, field
from scipy.interpolate import interp1d

@define(slots=True)
class PseudoEvolutionData:
    enclosed_density: np.ndarray
    enclosed_mass: np.ndarray

    mass_interpolation: interp1d = field(init=False)
    density_interpolation: interp1d = field(init=False)

    def __attrs_post_init__(self) -> None: 
        self.mass_interpolation = interp1d(
            np.log10(self.enclosed_density), 
            np.log10(self.enclosed_mass), 
            fill_value="extrapolate"
        )
        self.density_interpolation = interp1d(
            np.log10(self.enclosed_mass), 
            np.log10(self.enclosed_density),
            fill_value="extrapolate"
        )

    def get_interpolated_mass(self, ref_density: float | np.ndarray) -> float | np.ndarray:
        return 10.0**self.mass_interpolation(np.log10(ref_density))
    
    def get_interpolated_density(self, ref_mass: float | np.ndarray) -> float | np.ndarray:
        return 10.0**self.density_interpolation(np.log10(ref_mass))
    
    def get_radius(
            self, 
            mass: float | np.ndarray | None = None, 
            density: float | np.ndarray | None = None, 
        ) -> float | np.ndarray:

        if mass is not None:
            density = self.get_interpolated_density(mass)
        elif density is not None:
            mass = self.get_interpolated_mass(density)
        else:
            raise ValueError("Must provide either a reference mass or density")

        return np.cbrt(3.0 * mass / (4.0 * np.pi * density))

    
    def above_density_threshold(self, threshold: float) -> np.ndarray:
        ... 

    def below_mass_threshold(self, threshold: float) -> np.ndarray:
        ... 

    # def get_pseudo_evolution(
    #     self, ref_mass: float | None = None, 
    #     ref_density: float | None, 
    # ) -> np.ndarray:
    #     if ref_mass is not None
    #         dens