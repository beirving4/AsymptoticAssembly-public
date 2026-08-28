from __future__ import annotations

from abc import ABC
from typing import Any
from attrs import define
from collections import OrderedDict

from ..simulation.moments import MomentsInTime

@define(slots=True)
class EvolutionModel(ABC):
    moments: MomentsInTime
    fits: OrderedDict[int, Any]  

    # Maybe add a call method to this class to make it callable
    # add the take the mass, peak, etc. and scale factors as as arguments
    # then use the values from the time-dependent interpolated fit parameters 

    def __attrs_post_init__(self) -> None:
        self.fits = OrderedDict(sorted(self.fits.items(), key=lambda x: x[0]))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\t" + ",\n\t".join([f"{k}: {v}" for k, v in self.fits.items()]) +
            f"\n)"
        )

    def __len__(self) -> int:
        return len(self.fits)
    
    def __getitem__(self, snapshot_id: int) -> Any:
        return self.fits[snapshot_id]

    @property
    def snapshot_ids(self) -> list[int]:
        return list(self.fits.keys())


    # def __call__(self, *args: Any, **kwds: Any) -> Any:
    #     ...