from __future__ import annotations

import numpy as np
from attrs import define, field

@define(slots=True)
class NBodyPhaseSpaceProperties:
    velocity: np.ndarray
    radial_velocity: np.ndarray | None = field(default=None)

    acceleration: np.ndarray | None = field(default=None)
    radial_acceleration: np.ndarray | None = field(default=None)

    @property
    def x_velocity(self) -> float:
        return self.velocity[0]
    
    @property
    def y_velocity(self) -> float:
        return self.velocity[1]
    
    @property
    def z_velocity(self) -> float:
        return self.velocity[2]
    
    @property
    def x_acceleration(self) -> float | None:
        return self.acceleration[0] if self.acceleration is not None else None
    
    @property
    def y_acceleration(self) -> float | None:
        return self.acceleration[1] if self.acceleration is not None else None
    
    @property
    def z_acceleration(self) -> float | None:
        return self.acceleration[2] if self.acceleration is not None else None


@define(slots=True)
class NBodyProperties:
    mass: float
    position: np.ndarray

    peculiar: NBodyPhaseSpaceProperties
    actual: NBodyPhaseSpaceProperties | None


    # velocity: np.ndarray
    # acceleration: np.ndarray | None = field(default=None)
    # radial_distance: float | None = field(default=None)
    # radial_velocity: float | None = field(default=None)
    # radial_acceleration: float | None = field(default=None)   

    @property
    def x_position(self) -> float:
        return self.position[0]
    
    @property
    def y_position(self) -> float:
        return self.position[1]
    
    @property
    def z_position(self) -> float:
        return self.position[2]
    
    @property
    def x_velocity(self) -> float:
        return self.peculiar.x_velocity
    
    @property
    def y_velocity(self) -> float:
        return self.peculiar.y_velocity
    
    @property
    def z_velocity(self) -> float:
        return self.peculiar.z_velocity
    
    @property
    def x_acceleration(self) -> float | None:
        return self.peculiar.x_acceleration
    @property
    def y_acceleration(self) -> float | None:
        return self.peculiar.y_acceleration
    
    @property
    def z_acceleration(self) -> float | None:
        return self.peculiar.z_acceleration
    

@define(slots=True)
class NBodyParticle:
    id_number: int
    properties: NBodyProperties

    def get_property(self, property_name: str) -> float | np.ndarray | None:
        return getattr(self.properties, property_name, None)