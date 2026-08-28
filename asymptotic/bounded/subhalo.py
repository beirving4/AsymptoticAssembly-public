from __future__ import annotations

import numpy as np
import pdb

from pathlib import Path
from attrs import define, field
from collections import OrderedDict
from collections.abc import Collection

from ..simulation.evo import EvolutionData
from ..simulation.moments import Moment, MomentsInTime
from ..cosmo.model import Cosmology
# from .mass_def import MassDefinitions
from ..mass_def.data import MassDefinitions
from ..simulation.sub_box import SubBoxes
from ..utils.get_data import get_fof_subfind_data
from .objects import BoundedObject, BoundedObjectEvolutionData
from .population import (
    HaloPopulation, HaloPopulationEvolutionData
)
from ..particles.collection import (
    BoundedParticleCollection, BoxParticleCollection
)
from ..particles.shapes import (
    ShapeParameterData, 
    ShapeProfileData, 
    PopulationShapeParameterData
)
from ..particles.anisotropy import (
    AnisotropyParameterData, 
    AnisotropyProfileData, 
    PopulationAnisotropyParameterData
)

@define(slots=True)
class LineageMapping:
    ids: np.ndarray
    first_ids: np.ndarray
    next_ids: np.ndarray = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.next_ids = np.setdiff1d(self.ids, self.first_ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> LineageMapping:
        return LineageMapping(ids=self.ids[idx], first_ids=self.first_ids[idx])

    def get_subset(self, ids: int | Collection[int]) -> LineageMapping:
        if not isinstance(ids, np.ndarray):
            ids = np.array(ids)
        return LineageMapping(
            ids=self.ids[np.isin(self.ids, ids)],
            first_ids=self.first_ids[np.isin(self.ids, ids)],
        )


@define(slots=True)
class SubhaloMap:
    group_id: int 
    subhalo_ids: np.ndarray | None = field(default=None)
    ranks: np.ndarray | None = field(default=None)
    first_subhalo_id: int | None = field(default=None)
    prog: LineageMapping | None = field(default=None)
    desc: LineageMapping | None = field(default=None)

    @property
    def descendant_ids(self) -> np.ndarray | None:
        return None if self.desc is None else self.desc.ids

    @property
    def progenitor_ids(self) -> np.ndarray | None:
        return None if self.prog is None else self.prog.ids



@define(slots=True)
class SubhaloMapping:
    group_ids: np.ndarray
    subhalo_ids: np.ndarray
    ranks: np.ndarray
    subhalo_group_numbers: np.ndarray
    first_subhalo_ids: np.ndarray
    prog: LineageMapping | None = field(default=None)
    desc: LineageMapping | None = field(default=None)

    def __len__(self) -> int:
        return len(self.group_ids)
    
    def __getitem__(self, group_id: int) -> SubhaloMap:

        if not np.isin(group_id, self.group_ids):
            raise IndexError("Group ID not found in the data")

        subhalo_map = SubhaloMap(group_id=group_id)

        index = np.where(self.group_ids == group_id)[0][0]
        target_idxs = np.where(self.subhalo_group_numbers == group_id)[0]

        if len(target_idxs) == 0:
            return subhalo_map

        subhalo_map.first_subhalo_id = int(self.first_subhalo_ids[index])
        subhalo_map.subhalo_ids = self.subhalo_ids[target_idxs]
        subhalo_map.ranks = self.ranks[target_idxs]

        subhalo_map.prog = (
            self.prog.get_subset(self.subhalo_ids[target_idxs])
            if self.prog is not None else None
        )
        subhalo_map.desc = (
            self.desc.get_subset(self.subhalo_ids[target_idxs])
            if self.desc is not None else None
        )

        return subhalo_map

    @property
    def descendant_ids(self) -> np.ndarray | None:
        return None if self.desc is None else self.desc.ids

    @property
    def progenitor_ids(self) -> np.ndarray | None:
        return None if self.prog is None else self.prog.ids

    def get_subset(self, group_ids: int | Collection[int]) -> SubhaloMapping:

        if not isinstance(group_ids, np.ndarray):
            group_ids = np.array(group_ids)
            
        
        target_group_locs = np.isin(self.group_ids, group_ids)
        target_group_ids = self.group_ids[target_group_locs]
        target_subhalo_locs = np.isin(self.subhalo_group_numbers, target_group_ids)

        return SubhaloMapping(
            group_ids=target_group_ids,
            subhalo_ids=self.subhalo_ids[target_subhalo_locs],
            ranks=self.ranks[target_subhalo_locs],
            subhalo_group_numbers=self.subhalo_group_numbers[target_subhalo_locs],
            first_subhalo_ids=self.first_subhalo_ids[target_group_locs],
            prog=(
                self.prog.get_subset(self.subhalo_ids[target_subhalo_locs])
                if self.prog is not None else None
            ),
            desc=(
                self.desc.get_subset(self.subhalo_ids[target_subhalo_locs])
                if self.desc is not None else None
            )
        )
    
    def get_group_ids_from_subhalo_ids(self, subhalo_ids: int | Collection[int]) -> np.ndarray:
        if not isinstance(subhalo_ids, np.ndarray):
            subhalo_ids = np.array(subhalo_ids)

        subhalo_locations = np.isin(self.subhalo_ids, subhalo_ids)

        if ((subhalo_locations.size == 0) or (int(subhalo_locations.sum()) == 0)):
            raise IndexError("No subhalos found with the given ids")
        
        group_nums = self.subhalo_group_numbers[subhalo_locations]
        
        return np.unique(group_nums)
    
    def get_progenitor_ids(self, subhalo_ids: int | Collection[int]) -> np.ndarray:
        if not isinstance(subhalo_ids, np.ndarray):
            subhalo_ids = np.array(subhalo_ids)
        
        if self.progenitor_ids is None:
            raise ValueError("No progenitors are present in this data")
        
        return self.progenitor_ids[np.isin(self.subhalo_ids, subhalo_ids)]
    
    
    def get_descendant_ids(self, subhalo_ids: int | Collection[int]) -> np.ndarray:
        if not isinstance(subhalo_ids, np.ndarray):
            subhalo_ids = np.array(subhalo_ids)
        
        if self.descendant_ids is None:
            raise ValueError("No descendants are present in this data")
        
        return self.descendant_ids[np.isin(self.subhalo_ids, subhalo_ids)]
    
    def get_progenitor_ids_from_group_ids(self, group_ids: int | Collection[int]) -> np.ndarray:
        if not isinstance(group_ids, np.ndarray):
            group_ids = np.array(group_ids)
        
        if self.progenitor_ids is None:
            raise ValueError("No progenitors are present in this data")
        
        group_locs = np.isin(self.group_ids, group_ids)

        if (group_locs.sum() == 0):
            raise ValueError("No progenitors are present in the given groups")
        
        first_subhalo_ids = self.first_subhalo_ids[group_locs]
        # subhalo_locs = np.isin(self.subhalo_ids, first_subhalo_ids)
        subhalo_locs = np.intersect1d(
            ar1=self.subhalo_ids,
            ar2=first_subhalo_ids, 
            return_indices=True
        )[1]

        '''

        There's a bug when going from high snapshot to small snapshot in track by descendant. 
        It's due to issue when earlier snapshot has more subhalos than the later snapshot. 
        Meaning there are fewer progenitor ids than the subhalo ids.
        
        '''

        try:
            prog_ids = self.progenitor_ids[subhalo_locs]
        except IndexError as error:
            print(f"Error: {error}")
            pdb.set_trace()

        # pdb.set_trace()
        
        return self.progenitor_ids[subhalo_locs]
    
    def get_descendant_ids_from_group_ids(self, group_ids: int | Collection[int]) -> np.ndarray:
        if not isinstance(group_ids, np.ndarray):
            group_ids = np.array(group_ids)
        
        if self.descendant_ids is None:
            raise ValueError("No descendants are present in this data")
        
        group_locs = np.isin(self.group_ids, group_ids)

        if (group_locs.sum() == 0):
            raise ValueError("No descendants are present in the given groups")
        
        first_subhalo_ids = self.first_subhalo_ids[group_locs]
        # subhalo_locs = np.isin(self.subhalo_ids, first_subhalo_ids)
        subhalo_locs = np.intersect1d(
            ar1=self.subhalo_ids,
            ar2=first_subhalo_ids,  
            return_indices=True
        )[1]
        
        return self.descendant_ids[subhalo_locs]

    @classmethod
    def from_data(cls, data: dict) -> SubhaloMapping:

        if ((prog_ids := data["prog"].get("Subhalo/ProgSubhaloNr")) is not None):
            prog = LineageMapping(
                ids=prog_ids,
                first_ids=data["prog"]["Subhalo/FirstProgSubhaloNr"] 
            )
        else:
            prog = None

        if ((desc_ids := data["desc"].get("Subhalo/DescSubhaloNr")) is not None):
            desc = LineageMapping(
                ids=desc_ids,
                first_ids=data["desc"]["Subhalo/FirstDescSubhaloNr"]
            )
        else:
            desc = None

        return cls(
            group_ids=data["fof"]["group_ids"],
            subhalo_ids=data["prog"]["Subhalo/SubhaloNr"],
            ranks=data["subfind"]["SubhaloRankInGr"],
            subhalo_group_numbers=data["subfind"]["SubhaloGroupNr"],
            first_subhalo_ids=data["fof"]["group_first_subhalo_ids"],
            prog=prog,
            desc=desc
        )

@define(slots=True)
class SubhaloMappingEvo(EvolutionData):
    data: OrderedDict[int, SubhaloMapping]


    # def get_progenitor_group_ids(
    #         self, 
    #         group_ids: int | Collection[int],
    #         reference_snap_id: int,
    #         target_snap_id: int
    #     ) -> np.ndarray:

    #     if not isinstance(group_ids, np.ndarray):
    #         group_ids = np.array(group_ids)

    #     ...

    # def get_descendant_group_ids(
    #         self, 
    #         group_ids: int | Collection[int],
    #         reference_snap_id: int,
    #         target_snap_id: int
    #     ) -> np.ndarray:

    #     if not isinstance(group_ids, np.ndarray):
    #         group_ids = np.array(group_ids)

    #     ...

@define(slots=True)
class Subhalo(BoundedObject):

    rank: int
    spin: float 
    velocity_dispersion: float
    max_circular_velocity: float

    particles: BoundedParticleCollection | None = field(default=None, repr=False)

    shape: ShapeParameterData | None = field(default=None, repr=False)
    anisotropy: AnisotropyParameterData | None = field(default=None, repr=False)

    has_physical_boundaries: bool = field(default=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self.mass_defs.subfind is None:
            raise ValueError("Subfind mass definition must be provided")
        
    def __repr__(self) -> str:
        return (
            f"Subhalo(id={self.id_number}, rank={self.rank}, "
            f"M={10.0**self.mass:.3e} Msun/h)"
        )
    
    @property
    def mass(self) -> float:
        return self.mass_defs.subfind.mass
    
    @property
    def radius(self) -> float | None:
        if self.mass_defs.subfind.radius is None:
            raise ValueError("Subfind radius must be provided")
        return self.mass_defs.subfind.radius


    def set_particles(self, box_particles: BoxParticleCollection) -> None:
        
        # pdb.set_trace()
        
        self.particles = BoundedParticleCollection.from_box_particles(
            box_particles=box_particles, 
            group_coordinates=self.coordinate, 
            group_velocities=self.velocity,
            num_neighbors=self.particle_count,
            use_comoving=False
        )
    
    def get_particles(
            self, box_particles: BoxParticleCollection,
        ) -> BoundedParticleCollection:
        if self.particles is None:
            self.set_particles(box_particles)
        return self.particles
    

@define(slots=True)
class Subhalos(HaloPopulation):

    ranks: np.ndarray
    velocity_dispersions: np.ndarray
    max_velocities: np.ndarray

    ids: np.ndarray | None = field(default=None)
    parent_ranks: np.ndarray | None = field(default=None)
    progenitor_ids: np.ndarray | None = field(default=None)
    descendant_ids: np.ndarray | None = field(default=None)

    sub_boxes: SubBoxes | None = field(default=None, repr=False)

    min_resolved_mass: float = field(default=0)
    min_count_above_max_bin_hist: int = field(default=0)

    shapes: PopulationShapeParameterData | None = field(default=None)
    anisotropies: PopulationAnisotropyParameterData | None = field(default=None)

    has_physical_boundaries: bool = field(default=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self.ids is None:
            self.ids = np.arange(self.n_subhalos)
    
    def __repr__(self) -> str:
        return f"Subhalos(n_subhalos={self.n_subhalos})"

    def __getitem__(self, subhalo_id: int) -> Subhalo:
        if self.ids is None:
            raise ValueError("This object does not have Subhalo IDs to perform subset operation")
        
        idx = np.isin(self.ids, subhalo_id)

        # pdb.set_trace()

        return Subhalo(
            rank=self.ranks[idx],
            id_number=self.ids[idx],
            coordinate=self.positions[idx].flatten(),
            velocity=self.velocities[idx].flatten(),
            particle_count=int(self.particle_counts[idx]),
            spin=self.spins[idx, :] if self.spins is not None else None,
            velocity_dispersion=float(self.velocity_dispersions[idx]),
            max_circular_velocity=float(self.max_velocities[idx]),
            moment=self.moment,
            mass_defs=self.mass_defs.get_mass_defs_for_object(idx),
            shape=self.shapes[idx] if self.shapes is not None else None,
            anisotropy=self.anisotropies[idx] if self.anisotropies is not None else None,
        )
    
    def get_subset(self, subhalo_ids: int | Collection[int]) -> Subhalos | None:

        if self.ids is None:
            print("This object does not have Subhalo IDs to perform subset operation")
            return None
        
        if not isinstance(subhalo_ids, np.ndarray):
            subhalo_ids = np.array(subhalo_ids)

        locations = np.isin(self.ids, subhalo_ids)

        assert np.array_equal(self.ids[locations], subhalo_ids)

        # Verify if the group doesn't have a subhalo. If so, then we need to change the FOFGroup 
        # attribute to be Subhalos | None or create an empty Subhalos object
        if ((locations.size == 0) or (locations.sum() == 0)):
            print("No subhalos found with the given ids")
            return None

        return Subhalos(
            ranks=self.ranks[locations],
            ids=self.ids[locations],
            positions=self.positions[locations],
            velocities=self.velocities[locations],
            particle_counts=self.particle_counts[locations],
            spins=self.spins[locations] if self.spins is not None else None,
            velocity_dispersions=self.velocity_dispersions[locations],
            max_velocities=self.max_velocities[locations],
            moment=self.moment,
            mass_defs=self.mass_defs.get_subset(locations),
            progenitor_ids=(
                self.progenitor_ids[locations] 
                if self.progenitor_ids is not None else 
                None
            ),
            descendant_ids=(
                self.descendant_ids[locations] 
                if self.descendant_ids is not None else 
                None
            ),
            parent_ranks=(
                self.parent_ranks[locations] 
                if self.parent_ranks is not None else   
                None
            ),
            shapes=self.shapes[locations] if self.shapes is not None else None,
            anisotropies=self.anisotropies[locations] if self.anisotropies is not None else None,
        )
    
    def get_subset_by_index(self, locations: int | Collection[int]) -> Subhalos | None:
        if not isinstance(locations, np.ndarray):
            locations = np.array(idxs)

        if self.ids is None:
            raise ValueError("This object does not have Subhalo IDs to perform subset operation")

        return Subhalos(
            ranks=self.ranks[locations],
            ids=self.ids[locations],
            positions=self.positions[locations],
            velocities=self.velocities[locations],
            particle_counts=self.particle_counts[locations],
            spins=self.spins[locations] if self.spins is not None else None,
            velocity_dispersions=self.velocity_dispersions[locations],
            max_velocities=self.max_velocities[locations],
            moment=self.moment,
            mass_defs=self.mass_defs.get_subset(locations),
            progenitor_ids=(
                self.progenitor_ids[locations] 
                if self.progenitor_ids is not None else 
                None
            ),
            descendant_ids=(
                self.descendant_ids[locations] 
                if self.descendant_ids is not None else 
                None
            ),
            parent_ranks=(
                self.parent_ranks[locations] 
                if self.parent_ranks is not None else   
                None
            ),
            shapes=self.shapes[locations] if self.shapes is not None else None,
            anisotropies=self.anisotropies[locations] if self.anisotropies is not None else None,
        )

    def sample(self, n_samples: int) -> Subhalos:
        if n_samples > len(self):
            raise ValueError("Number of samples must be less than the number of subhalos")
        idxs = np.random.choice(self.ids.size, size=n_samples, replace=False)
        return self.get_subset_by_index(idxs)

    def get_top_n_by_mass(self, n: int) -> Subhalos:
        top_n_indices = np.argsort(self.particle_counts)[-n:]
        return self.get_subset_by_index(top_n_indices)

    def setup_sub_boxes(self, box_size: float, num_splits: int) -> None:
        ids = self.ids if self.ids is not None else np.arange(self.n_subhalos)
        self.sub_boxes = SubBoxes.from_subhalos(
            box_size=box_size, 
            num_splits=num_splits, 
            subhalo_ids=ids,
            subhalo_positions=self.positions
        )
    
    @property
    def n_subhalos(self) -> int:
        return len(self)
    
    @property
    def lead_subhalo_id(self) -> int:
        return self.ids[self.ranks == 0][0]
    
    @property
    def lead_subhalo(self) -> Subhalo:
        return self[self.lead_subhalo_id]
    
    @classmethod
    def from_catalog(
            cls, 
            catalog_dir: Path,
            snapshot_id: int, 
            hubble: float,
            min_resolved_mass: float = 0.0,
            min_count_above_max_bin_hist: int = 0,
            is_batched: bool = False,
            in_comoving: bool = True,
            cosmo: Cosmology | None = None
        ) -> Subhalos:

        data = get_fof_subfind_data(
            directory=catalog_dir, 
            snap_idx=snapshot_id, 
            is_batched=is_batched,
            in_comoving=in_comoving
        )

        moment = Moment(
            snapshot_id=snapshot_id, 
            scale_factor=data["fof"]["scale_factor"]
        )

        if cosmo is not None:
            moment.add_times(cosmo)

        return cls.from_catalog_data(
            data=data, 
            moment=moment, 
            hubble=hubble,
            min_resolved_mass=min_resolved_mass,
            min_count_above_max_bin_hist=min_count_above_max_bin_hist
        )


    @classmethod
    def from_catalog_data(
            cls, data: dict,
            moment: Moment, 
            hubble: float,
            min_resolved_mass: float = 0.0,
            min_count_above_max_bin_hist: int = 0,
        ) -> Subhalos:

        # if moment.snapshot_id == 34: pdb.set_trace()

        return cls(
            ids=data["prog"]["Subhalo/SubhaloNr"],
            ranks=data["subfind"]["SubhaloRankInGr"],
            parent_ranks=data["subfind"]["SubhaloParentRank"],
            positions=data["subfind"]["SubhaloPos"],
            velocities=data["subfind"]["SubhaloVel"],
            particle_counts=data["subfind"]["SubhaloLen"],
            spins=data["subfind"]["SubhaloSpin"],
            velocity_dispersions=data["subfind"]["SubhaloVelDisp"],
            max_velocities=data["subfind"]["SubhaloVmax"],
            progenitor_ids=data["prog"].get("Subhalo/ProgSubhaloNr"),
            descendant_ids=data["desc"].get("Subhalo/DescSubhaloNr"),
            moment=moment,
            mass_defs=MassDefinitions.from_subfind_data(data["subfind"], hubble),
            min_resolved_mass=min_resolved_mass,
            min_count_above_max_bin_hist=min_count_above_max_bin_hist
        )
    

    def set_particles(
            self, 
            box_particles: BoxParticleCollection,
            cosmo: Cosmology | None = None,
            use_comoving: bool = True
        ) -> None:
        ...
    

@define(slots=True)
class SubhaloEvoData(BoundedObjectEvolutionData):
    id: int 
    data: OrderedDict[int, Subhalo]

    def __repr__(self) -> str:
        return super().__repr__()
    
    @property
    def mass_accretion_history(self) -> np.ndarray:
        return super().get_mass_accretion_history(mass_def="subfind")
    
    def get_boundary_evolution_history(self, return_in_comoving: bool = False) -> np.ndarray:
        return super().get_boundary_evolution_history(
            mass_def="subfind", 
            return_in_comoving=return_in_comoving
        )
    
    def _get_bounded_object_with_particles(
            self, snapshot_id: int,
            box_particles: BoxParticleCollection | None = None,
            cosmo: Cosmology | None = None
        ) -> Subhalo:

        ...
        


@define(slots=True)
class SubhalosEvoData(HaloPopulationEvolutionData):
    # ids: np.ndarray
    data: OrderedDict[int, Subhalos]

    def __repr__(self) -> str:
        return super().__repr__()
        

    @classmethod
    def from_data(
            cls, data: dict, 
            moments: MomentsInTime, 
            hubble: float, 
            *args, **kwargs
        ) -> SubhalosEvoData:

        ...

    @classmethod
    def from_directory(
            cls, sim_dir: Path, 
            cosmo: Cosmology,
            in_comoving: bool,
            is_batched: bool,
            *args, **kwargs
        ) -> SubhalosEvoData:

        ...