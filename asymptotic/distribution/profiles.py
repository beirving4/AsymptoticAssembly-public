from __future__ import annotations

import numpy as np, pdb
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib import gridspec
from attrs import define, field
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ..particles.collection import BoundedParticleCollection

from .units import DistributionUnitSystem
from ..mass_def.data import MassDefinitions
from .mass import MassDistributionData, get_mass_distribution_bins
from .phase_space import (
    PhaseSpaceDistributionData, 
    get_phase_space_distribution_bins   
)
from .viz import plot_mass_radial_profiles, plot_radial_velocity_distribution


@define(slots=False)
class ProfileData: 
    source_mass: float | None

    units: DistributionUnitSystem

    mass: MassDistributionData = field(repr=False)
    phase_space: PhaseSpaceDistributionData = field(repr=False)

    for_group: bool = field(default=True)

    def __attrs_post_init__(self) -> None:
        self.phase_space.compute_histogram2D(self.mass.average_weights)

    @property
    def bounded_object_type(self) -> str:
        return "FOFGroup" if self.for_group else "Subhalo"

    
    def _distribution_title(self, _type: str) -> str:
        if _type == "mass":
            profile_type = "Mass Distribution"
        elif _type == "phase_space":
            profile_type = "Phase Space Distribution"
        elif _type == "all": 
            profile_type = "Radial Profiles"
        else:
            raise ValueError("Invalid profile type")

        if self.source_mass is not None:
            return (
                rf"{profile_type} for a $\log_{{10}}M$={self.source_mass:.2f} "
                rf"$h^{{-1}}M_{{\odot}}$ {self.bounded_object_type}"
            )
        else:
            return f"{profile_type} for a All {self.bounded_object_type}s"

    @classmethod
    def from_bounded_objects(
            cls,
            particles: BoundedParticleCollection,
            region_radius: float,
            mass_defs: MassDefinitions,
            hubble: float,
            overdenisty: float = 200.0,
            bin_count: int = 50,
            num_objects: int = 1,
            use_softening: bool = False,
            cumulative_masses: np.ndarray | None = None,
            for_group: bool = True,
            mean_density: float | None = None
        ) -> ProfileData:



        scaling_def = f"crit{int(overdenisty)}" if for_group else "subfind"
        units = DistributionUnitSystem.from_mass_def(
            getattr(mass_defs, scaling_def)
        )
        units.hubble = hubble
        units.in_physical_units = True

        if use_softening:
            lower_bound = particles.properties.softening_length / units.length
        else:
            lower_bound = 0.1

        if mass_defs.mean200 is not None:
            normalized_def = mass_defs.normalized_definition(
                "mean200", scaling_def, overdenisty=overdenisty
            )
            mean_density = normalized_def.density
        elif mean_density is not None:
            mean_density /= getattr(mass_defs, scaling_def).density
        else:
            raise ValueError("Must provide mean density or normalized mass definition")
        
        if mean_density is None:
            raise ValueError("Mean density must be provided")
        

        radii = particles.properties.radial_distances 
        masses = particles.properties.masses 
        velocities = particles.properties.radial_velocities



        if num_objects == 1:
            radii /= units.length
            masses /= units.masses
            velocities /= units.velocities
    

        if cumulative_masses is None:
            sorted_by_radii = np.argsort(radii)
            radii, masses = radii[sorted_by_radii], masses[sorted_by_radii]
            velocities = velocities[sorted_by_radii]
            cumulative_masses = np.cumsum(masses)  / num_objects

        mass_distribution_bins = get_mass_distribution_bins(
            radii=(radii / units.length) if (num_objects > 1) else radii, 
            lower=lower_bound, 
            upper=(region_radius / units.length), 
            bin_count=bin_count
        )

        phase_space_bins = get_phase_space_distribution_bins(
            velocities=velocities, bin_edges=mass_distribution_bins["edges"]
        )


        return cls(
            source_mass=getattr(mass_defs, scaling_def).mass,
            units=units,
            for_group=for_group,
            mass=MassDistributionData(
                bin_count=bin_count,
                num_objects=num_objects,
                radii=radii,
                weights=masses,
                mean_density=mean_density,
                cumulative=cumulative_masses,
                radial_bins=mass_distribution_bins
            ),
            phase_space=PhaseSpaceDistributionData(
                bin_count=bin_count,
                num_objects=num_objects,
                radii=radii,
                weights=velocities,
                critical_overdenisty=overdenisty,
                radial_bins={
                    "edges": phase_space_bins["edges"],
                    "centers": phase_space_bins["centers"]
                },
                velocity_bins=phase_space_bins["velocity"]
            )
        )

    
    def get_radii(self, use_physical_units: bool = False) -> np.ndarray:
        self.units.in_physical_units = use_physical_units
        return self.mass.get_radii(self.units.radii)
    
    def get_masses(self, use_physical_units: bool = False) -> np.ndarray:
        self.units.in_physical_units = use_physical_units
        return self.mass.get_masses(self.units.masses)
    
    def get_velocities(self, use_physical_units: bool = False) -> np.ndarray:
        self.units.in_physical_units = use_physical_units
        return self.phase_space.get_radial_velocities(self.units.velocities)
    
    def radial_bins(self, use_physical_units: bool = False) -> dict[str, np.ndarray]:
        self.units.in_physical_units = use_physical_units
        return {
            "edges": self.mass.radial_bins["edges"] * self.units.radii,
            "centers": self.mass.radial_bins["centers"] * self.units.radii,
        }
    
    def velocity_bins(self, use_physical_units: bool = False) -> np.ndarray:
        self.units.in_physical_units = use_physical_units
        return self.phase_space.velocity_bins * self.units.velocities

    def density(self, use_physical_units: bool = False) -> np.ndarray:
        self.units.in_physical_units = use_physical_units
        return self.mass.density * self.units.densities
    
    def get_mean_density(self, use_physical_units: bool = False) -> float:
        self.units.in_physical_units = use_physical_units
        return self.mass.mean_density * self.units.densities
    
    def scaled_density(self, use_physical_units: bool = False) -> np.ndarray:
        self.units.in_physical_units = use_physical_units
        return self.mass.scaled_density * self.units.scaled_densities
    
    def mass_enclosed(self, use_physical_units: bool = False) -> np.ndarray:
        self.units.in_physical_units = use_physical_units
        radial_bins = self.radial_bins(use_physical_units)["centers"]
        enclosed_masses = np.interp(
            np.log10(radial_bins), 
            np.log10(self.mass.radii * self.units.radii),
            np.log10(self.mass.enclosed * self.units.masses)
        )
        return 10.0**enclosed_masses
    
    def enclosed_density(self, use_physical_units: bool = False) -> np.ndarray:
        self.units.in_physical_units = use_physical_units
        radial_bins = self.radial_bins(use_physical_units)["centers"]
        normalized_enclosed_density = self.mass.enclosed_density(1.0)
        enclosed_densities = np.interp(
            np.log10(radial_bins),
            np.log10(self.mass.radii[1:] * self.units.radii),
            np.log10(normalized_enclosed_density * self.units.densities)
        )
        return 10.0**enclosed_densities
    
    @property
    def enclosed_overdensity(self) -> np.ndarray:
        return (self.enclosed_density() / self.get_mean_density()) - 1.0
    
    def radial_velocity(self, use_physical_units: bool = False) -> dict[str, np.ndarray]:
        self.units.in_physical_units = use_physical_units
        profile = self.phase_space.profile
        return {
            "median": profile["median"] * self.units.velocities,
            "25pct": profile["25pct"] * self.units.velocities,
            "75pct": profile["75pct"] * self.units.velocities,
        }
    
    @property
    def radial_to_recession_velocity_ratio(self) -> np.ndarray:
        return self.phase_space.radial_to_hubble_ratio
    
    def phase_space_distribution(self, use_physical_units: bool = False) -> np.ndarray | None:
        if self.phase_space.num_objects == 1: return None
        self.units.in_physical_units = use_physical_units
        return self.phase_space.distribution
    

    def peculiar_velocity_profile(self, use_physical_units: bool = False) -> np.ndarray:
        self.units.in_physical_units = use_physical_units
        return self.phase_space.peculiar_velocity_profile * self.units.velocities
    
    def get_pseudo_evolutions(
            self, mass_defs: MassDefinitions, use_physical_units: bool = False
        ) -> np.ndarray:

        self.units.in_physical_units = use_physical_units

        return np.empty((0, 2))


    def to_numpy(self, use_physical_units: bool = False) -> np.ndarray:
        radial_bins = self.radial_bins(use_physical_units)
        vr_profile = self.radial_velocity(use_physical_units)
        return np.vstack(
            [
                radial_bins["centers"],
                self.density(use_physical_units),
                self.scaled_density(use_physical_units),
                self.mass_enclosed(use_physical_units),
                self.enclosed_overdensity,
                radial_bins["centers"],
                vr_profile["median"],
                vr_profile["25pct"],
                vr_profile["75pct"],
                self.peculiar_velocity_profile(use_physical_units),
                self.radial_to_recession_velocity_ratio
            ]
        )

    def save_to_numpy(self, save_to_dir: Path, use_physical_units: bool = False) -> None:
        filename = (
            f"{'dimensionalized' if use_physical_units else 'units'}_profile_data"
        )
        if self.source_mass is not None:
            filename += f"_mass_{int(self.source_mass)}.npy"
        profile_data = self.to_numpy(use_physical_units=use_physical_units)
        np.save(Path(save_to_dir, filename), profile_data)

    def display_mass_radial_profiles(
            self, 
            ax: Axes | None = None,
            use_physical_units: bool = False, 
        ) -> None:

        radial_bins = self.radial_bins(use_physical_units)["centers"]
        density = np.asarray([radial_bins, self.density(use_physical_units)])
        scaled_density = np.asarray([radial_bins, self.scaled_density(use_physical_units)])
        mass_enclosed = np.asarray([radial_bins, self.mass_enclosed(use_physical_units)])
        enclosed_overdensity = np.asarray([radial_bins, self.enclosed_overdensity])

        self.units.in_physical_units = use_physical_units

        plot_mass_radial_profiles(
            density,
            scaled_density,
            mass_enclosed,  
            enclosed_overdensity,
            ax=ax,
            radius_units=self.units.radii,
            mass_units=self.units.masses,
            density_units=self.units.densities,
            scaled_density_units=self.units.scaled_densities,
            plot_all=True
        )

        plt.suptitle(self._distribution_title("mass"))

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if ax is None:
            plt.show()


    def display_velocity_distribution(
            self, 
            ax: Axes | None = None,
            fig: Figure | None = None,
            x_scale: bool = True,
            y_scale: bool = False,
            use_physical_units: bool = False,
            density_label: str | None = None,
        ) -> AxesImage | None:

        self.units.in_physical_units = use_physical_units
        
        distribution = self.phase_space_distribution(use_physical_units)
        radial_bins = self.radial_bins(use_physical_units)
        velocity_profile = self.radial_velocity(use_physical_units)

        im = plot_radial_velocity_distribution(
            radii=(
                self.get_radii(use_physical_units)
                if distribution is None else
                radial_bins["edges"]
            ),
            velocity=(
                self.get_velocities(use_physical_units)
                if distribution is None else
                self.velocity_bins(use_physical_units)
            ),
            radial_bins=radial_bins["centers"], # Need to change this (may not need it anymore)
            velocity_medians=velocity_profile["median"],
            velocity_25pct=velocity_profile["25pct"],
            velocity_75pct=velocity_profile["75pct"],
            density=distribution,
            fig=fig,
            ax=ax,
            x_scale=x_scale,
            y_scale=y_scale,
            # May not need these anymore 
            radius_units=self.units.radii,
            velocity_units=self.units.velocities,
            density_units=self.units.densities,  
            density_label=density_label,
        )


        if im is not None:
            return im
        
        plt.suptitle(self._distribution_title("phase_space"))

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if ax is None:
            plt.show()
    
    def display(
        self,
        x_scale: bool = True,
        y_scale: bool = False,
        use_physical_units: bool = False,
    ) -> None:
        # Create a figure
        fig = plt.figure(figsize=(12, 15))

        # Define the grid layout
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.5])

        # Use the first four sections for mass radial profiles
        axes_top = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        self.display_mass_radial_profiles(
            ax=axes_top, use_physical_units=use_physical_units
        )

        # Use the bottom section for velocity distribution
        ax_bottom = fig.add_subplot(gs[2, :])  # This spans both columns
        self.display_velocity_distribution(
            ax=ax_bottom,
            fig=fig,
            x_scale=x_scale,
            y_scale=y_scale,
            use_physical_units=use_physical_units
        )

        plt.suptitle(self._distribution_title("all"))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()