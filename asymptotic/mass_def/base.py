from __future__ import annotations

import numpy as np, pdb


from typing import Any, Iterable
from abc import ABC, abstractmethod
from collections import OrderedDict
from attrs import define, field, fields


from ..model.evo import EvolutionModel
from ..simulation.evo import EvolutionData
from .mapping import global_mass_def_mapping
from ..simulation.moments import Moment, MomentsInTime


IntType = int | np.integer
FloatType = float | np.floating


MASS_DEF_COLORS = {
    "fof": "#1f77b4",  # blue
    "subfind": "#ff7f0e",  # orange
    "crit200": "#2ca02c",  # green
    "crit500": "#d62728",  # red
    "mean200": "#9467bd",  # purple
    "virial": "#8c564b",  # brown
    "scale": "#e377c2",  # pink
    "four_scale": "#7f7f7f",  # gray
    "max_circ": "#bcbd22",  # olive
    "bound": "#17becf",  # cyan
    "splashback": "#aec7e8",  # light blue
    "asymptotic": "#ffbb78"  # peach
}


MASS_DEF_PLOT_LINES = {
    "fof": (0, (5, 5)),  # dashed
    "subfind": (0, (1, 10)),  # loosely dotted
    "crit200": (0, ()),  # solid
    "crit500": (0, (3, 5, 1, 5)),  # dashdot
    "mean200": (0, (1, 1)),  # dotted
    "virial": (0, (5, 10)),  # loosely dashed
    "scale": (0, (3, 1, 1, 1, 1, 1)),  # dashdotdotted
    "four_scale": (0, (7, 3, 1, 3)),  # long dash-dot pattern
    "max_circ": (0, (7, 2)),  # widely spaced dashes
    "bound": (0, (2, 1, 2)),  # densely dashed with double lines
    "splashback": (0, (3, 1, 1, 1)),  # densely dashdotted
    "asymptotic": (0, (5, 1))  # densely dashed
}

MASS_DEF_MARKER_STYLES = {
    "fof": "o",  # circle
    "subfind": "d",  # diamond
    "crit200": "^",  # triangle up
    "crit500": "*",  # star
    "mean200": "s",  # square
    "virial": "P",  # plus (filled)
    "scale": "v",  # triangle down
    "four_scale": "<",  # triangle left
    "max_circ": "D",  # diamond (filled)
    "bound": ">",  # triangle right
    "splashback": "h",  # hexagon
    "asymptotic": "X"  # x (filled)
}

@define(slots=True)
class MassDefinitionDependentType(ABC):
    fof: Any | None = field(default=None)
    subfind: Any | None = field(default=None)
    crit200: Any | None = field(default=None)
    crit500: Any | None = field(default=None)
    mean200: Any | None = field(default=None)
    virial: Any | None = field(default=None)
    scale: Any | None = field(default=None)
    four_scale: Any | None = field(default=None)
    max_circ: Any | None = field(default=None)
    splashback: Any | None = field(default=None)
    bound: Any | None = field(default=None)
    # Solo-derived boundary defs — every entry in global_mass_def_mapping.fields must
    # exist as a slot here, since is_valid_mass_def(...) does getattr(self, <field>)
    # for every mapping field. Subclasses that only need a subset (or that already
    # declare these) inherit/override them; all default None.
    hydrostatic: Any | None = field(default=None)
    turnaround: Any | None = field(default=None)
    infall: Any | None = field(default=None)
    marginally_bound: Any | None = field(default=None)
    asymptotic: Any | None = field(default=None)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self.contained_keys)})"
    
    def __getitem__(self, mass_def: str) -> Any:
        for attr_name in global_mass_def_mapping.fields:
            if self.is_correct_key(attr_name , mass_def):
                return getattr(self, attr_name)
        raise ValueError(f"{mass_def} information not found!")
    

    def __setitem__(self, mass_def: str, value: Any) -> None:
        for attr_name in global_mass_def_mapping.fields:
            if getattr(global_mass_def_mapping, attr_name).key == mass_def:
                setattr(self, attr_name, value)
                return
        
        raise ValueError(f"{mass_def} information not found!")
    

    def __contains__(self, mass_def: str) -> bool:
        for attr_name in global_mass_def_mapping.fields:
            if getattr(self, attr_name) is None: continue
            if self.is_correct_key(attr_name, mass_def):
                return True
        return False

    def __iter__(self) -> Iterable[Any]:
        for attr_name in global_mass_def_mapping.fields:
            if self.is_valid_mass_def(attr_name):

                yield getattr(self, attr_name)

    def contained_property(
            self, prop_name: str | None = None, 
            nested_prop_name: str | None = None
        ) -> list[str]:

        properties = []

        for mass_def in global_mass_def_mapping.fields:

            if not self.is_valid_mass_def(mass_def): continue

            if prop_name is None: 
                properties.append(mass_def)
                continue

            prop_map = getattr(global_mass_def_mapping, mass_def)
            prop = (
                getattr(getattr(prop_map, prop_name), nested_prop_name)
                if (nested_prop_name is not None)
                else getattr(prop_map, prop_name)
            )

            properties.append(prop)

        return properties 
    
    @property
    def contained_definitions(self) -> list[str]:
        return self.contained_property()

    @property
    def contained_keys(self) -> list[str]:
        return self.contained_property(prop_name="key")
    
    @property
    def contained_display_names(self) -> list[str]:
        return self.contained_property(prop_name="display")

    @property
    def contained_mass_eqn(self) -> list[str]:
        return self.contained_property(prop_name="mass", nested_prop_name="eqn")

    @property
    def contained_radius_eqn(self) -> list[str]:
        return self.contained_property(prop_name="radius", nested_prop_name="eqn")
    
    @property
    def contained_peaks_eqn(self) -> list[str]:
        return self.contained_property(prop_name="peak_height", nested_prop_name="eqn")

    def is_valid_mass_def(self, mass_def: str) -> bool:
        if mass_def not in global_mass_def_mapping.fields:
            return False
        return getattr(self, mass_def) is not None

    def is_correct_key(self, attr_name: str, mass_def: str) -> bool:
        return (
            self.is_valid_mass_def(attr_name)
            and getattr(global_mass_def_mapping, attr_name).key == mass_def
        )

    @property
    def has_fof(self) -> bool:
        return self.fof is not None

    @property
    def has_subfind(self) -> bool:
        return self.subfind is not None
    
    @property
    def has_200c(self) -> bool:
        return self.crit200 is not None
    
    @property
    def has_500c(self) -> bool:
        return self.crit500 is not None

    @property
    def has_200m(self) -> bool:
        return self.mean200 is not None

    @property
    def has_vir(self) -> bool:
        return self.virial is not None
    
    @property
    def has_splashback(self) -> bool:
        return self.splashback is not None
    
    @property
    def has_max_circ(self) -> bool:
        return self.max_circ is not None
    
    @property
    def has_scale(self) -> bool:
        return self.scale is not None
    
    @property
    def has_four_scale(self) -> bool:
        return self.four_scale is not None
    
    @property
    def has_bound(self) -> bool:
        return self.bound is not None
    
    @property
    def has_asymptotic(self) -> bool:
        return self.asymptotic is not None

    # Solo-derived boundary defs (keys hs/ta/inf/marg_bound). get_mdef_attr tries
    # has_{key} then has_{slot}; these slot-named properties are the fallback so
    # those defs are recognized. getattr(..., None) keeps them safe on any subclass
    # whose schema predates these slots.
    @property
    def has_hydrostatic(self) -> bool:
        return getattr(self, "hydrostatic", None) is not None

    @property
    def has_turnaround(self) -> bool:
        return getattr(self, "turnaround", None) is not None

    @property
    def has_infall(self) -> bool:
        return getattr(self, "infall", None) is not None

    @property
    def has_marginally_bound(self) -> bool:
        return getattr(self, "marginally_bound", None) is not None

    def get_mdef_attr(self, mass_def_name: str, attr_name: str) -> Any | None:
        key = getattr(global_mass_def_mapping, mass_def_name).key
        # has_X properties mix two naming conventions across the mass-def
        # registry: traditional defs (200c/500c/200m/vir/bound) name them
        # after the key (has_200c, has_bound), while newer defs
        # (splashback/max_circ/four_scale) name them after the slot
        # (has_splashback, has_four_scale). Try the key first, then fall
        # back to the slot name.
        has_mass_def = getattr(self, f"has_{key}", None)
        if has_mass_def is None:
            has_mass_def = getattr(self, f"has_{mass_def_name}", False)
        return getattr(self[key], attr_name) if has_mass_def else None

    def get_mdef_attr_instance(
            self, mass_def_name: str, attr_name: str, instance_id: int
        ) -> Any | None:
        mdef_attr = self.get_mdef_attr(mass_def_name, attr_name)
        return mdef_attr.get(instance_id) if mdef_attr is not None else None

    def get_mass_def_attrs(self, attr_name: str) -> dict[str, Any | None]:
        return {
            mass_def_name : self.get_mdef_attr(mass_def_name, attr_name)
            for mass_def_name in global_mass_def_mapping.fields
            if self.is_valid_mass_def(mass_def_name)
        }

    def get_mass_def_attrs_instance(
            self, attr_name: str, instance_id: IntType | FloatType | str
        ) -> dict[str, Any | None]:
        mdef_attrs = self.get_mass_def_attrs(attr_name)
        if isinstance(instance_id, FloatType):
            instance_id = f"{instance_id:.1f}"
        return {
            mass_def_name : mdef_attr.get(instance_id)
            for mass_def_name, mdef_attr in mdef_attrs.items()
            if self.is_valid_mass_def(mass_def_name)
        }

    
    

@define(slots=True)
class MassDefinitionDependentEvoType(MassDefinitionDependentType, ABC):
    fof: EvolutionData | EvolutionModel | None = field(default=None)
    subfind: EvolutionData | EvolutionModel | None = field(default=None)
    crit200: EvolutionData | EvolutionModel | None = field(default=None)
    crit500: EvolutionData | EvolutionModel | None = field(default=None)
    mean200: EvolutionData | EvolutionModel | None = field(default=None)
    virial: EvolutionData | EvolutionModel | None = field(default=None)
    splashback: EvolutionData | EvolutionModel | None = field(default=None)
    max_circ: EvolutionData | EvolutionModel | None = field(default=None)
    scale: EvolutionData | EvolutionModel | None = field(default=None)
    four_scale: EvolutionData | EvolutionModel | None = field(default=None)
    bound: EvolutionData | EvolutionModel | None = field(default=None)
    # Solo-derived boundary defs the full_catalog loader populates. These match the
    # per-snapshot MassDefinitions/HaloMassFunction schema (and global_mass_def_mapping);
    # without them the evolution containers could not represent a full catalog and
    # is_valid_mass_def(...) raised on these keys.
    hydrostatic: EvolutionData | EvolutionModel | None = field(default=None)
    turnaround: EvolutionData | EvolutionModel | None = field(default=None)
    infall: EvolutionData | EvolutionModel | None = field(default=None)
    marginally_bound: EvolutionData | EvolutionModel | None = field(default=None)
    asymptotic: EvolutionData | EvolutionModel | None = field(default=None)

    def __repr__(self) -> str:
        contained_items_list = [
            f"{snap_id}: {data}" 
            for snap_id, data in self.contained_at_each_snapshot.items()
        ]
        return (
            f"{self.__class__.__name__}(\n"
            f"\t" + ",\n\t".join(contained_items_list) + "\n)"
        )
    
    def __iter__(self) -> Iterable[MassDefinitionDependentType]:
        snapshot_ids = self.snapshot_ids
        for snap_id in snapshot_ids:
            yield self.get_snapshot_instance(snap_id)

    @property
    def moments(self) -> MomentsInTime:
        
        ids = set()
        scale_factors = set()
        redshifts = set()
        ages = set()
        lookback_times = set()
        hubble_epochs = set()
        for mass_def in self.contained_definitions:
            data = getattr(self, mass_def)
            ids.update(data.moments.snapshot_ids)
            scale_factors.update(data.moments.scale_factors)
            redshifts.update(data.moments.redshifts)
            ages.update(data.moments.ages)
            lookback_times.update(data.moments.lookback_times)
            hubble_epochs.update(data.moments.hubble_epochs)

        return MomentsInTime(
            snapshot_ids=np.asarray(sorted(list(ids))),
            scale_factors=np.asarray(sorted(list(scale_factors))),
            redshifts=np.asarray(sorted(list(redshifts), reverse=True)),
            ages=np.asarray(sorted(list(ages))),
            lookback_times=np.asarray(sorted(list(lookback_times), reverse=True)),
            hubble_epochs=np.asarray(sorted(list(hubble_epochs), reverse=True)),
        )
    
    @property
    def as_evo_dict(self) -> OrderedDict[int, EvolutionData | EvolutionModel]:
        return OrderedDict({
            snap_id : self.get_snapshot_instance(snap_id)
            for snap_id in self.snapshot_ids
        })
    
    @property
    def snapshot_ids(self) -> np.ndarray:
        return self.moments.snapshot_ids
    
    @property
    def scale_factors(self) -> np.ndarray:
        return self.moments.scale_factors
    
    @property
    def redshifts(self) -> np.ndarray:
        return self.moments.redshifts

    @property
    def ages(self) -> np.ndarray:
        return self.moments.ages
    
    @property
    def lookback_times(self) -> np.ndarray:
        return self.moments.lookback_times

    @property
    def contained_definitions_at_each_snapshot(self) -> dict[int, list[str]]:
        return {
            snapshot_id: [
                mass_def
                for mass_def in self.contained_definitions
                if (snapshot_id in getattr(self, mass_def).snapshot_ids)
            ]
            for snapshot_id in self.snapshot_ids
        }

    @property
    def contained_keys_at_each_snapshot(self) -> dict[int, list[str]]:
        return {
            snapshot_id: [
                mass_def
                for mass_def in self.contained_keys
                if (snapshot_id in self[mass_def].snapshot_ids)
            ]
            for snapshot_id in self.snapshot_ids
        }
    
    @property
    def contained_type(self) -> type:
        return next(iter(self)).__class__

    @abstractmethod
    def get_snapshot_instance(self, snapshot_id: int) -> MassDefinitionDependentType:
        ...


    @property
    def contained_at_each_snapshot(self) -> dict[int, MassDefinitionDependentType]:
        results = {}
        for snapshot_id in self.snapshot_ids:
            try:
                results[snapshot_id] = self.get_snapshot_instance(snapshot_id)
            except ValueError:
                continue
        return results
        
        
