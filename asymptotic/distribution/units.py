from __future__ import annotations

import numpy as np

from attrs import define, field

from ..mass_def.data import MassDefinition


@define(slots=True)
class DistributionUnitSystem:
    length: float
    mass: float
    velocity: float
    hubble: float | None = field(default=None)

    _in_physical_units: bool = field(default=False)

    @classmethod
    def from_mass_def(cls, mass_def: MassDefinition) -> DistributionUnitSystem:
        return cls(
            length=mass_def.radius,
            mass=mass_def.mass,
            velocity=mass_def.velocity
        )

    @property
    def in_physical_units(self) -> bool:
        return self._in_physical_units
    
    @in_physical_units.setter
    def in_physical_units(self, value: bool) -> None:
        self._in_physical_units = value

    @property
    def radii(self) -> float:
        return self.length if self.in_physical_units else 1
    
    @property
    def masses(self) -> float:
        return 10.0**self.mass if self.in_physical_units else 1
    
    @property
    def velocities(self) -> float:
        return self.velocity if self.in_physical_units else 1
    
    @property
    def volume_factor(self) -> float:
        return (4. / 3) * np.pi if self.in_physical_units else 1
    
    @property
    def densities(self) -> float:
        return self.masses / (self.volume_factor * self.radii**3)
    
    @property
    def scaled_densities(self) -> float:
        return self.densities * self.radii**2 