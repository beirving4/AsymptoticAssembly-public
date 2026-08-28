from __future__ import annotations

import numpy as np

from abc import ABC
from typing import Any, Iterator
from attrs import define
from collections import OrderedDict

from .moments import MomentsInTime

@define(slots=True)
class EvolutionData(ABC):
    moments: MomentsInTime
    data: OrderedDict[int, Any]

    def __attrs_post_init__(self) -> None:
        self.data = OrderedDict(sorted(self.data.items(), key=lambda x: x[0]))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\t" + ",\n\t".join([f"{k}: {v}" for k, v in self.data.items()]) +
            f"\n)"
        )

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, snapshot_id: int) -> Any:
        return self.data[snapshot_id]
    
    def __iter__(self) -> Iterator[Any]:
        return iter(self.data.values())

    @property
    def snapshot_ids(self) -> list[int]:
        return list(self.data.keys())
    
    def get_snapshot_instance(self, snapshot_id: int) -> Any:
        return self.data[snapshot_id]
    
    @property
    def data_type(self) -> type:
        return type(next(iter(self.data.values())))
    
    @property
    def data_type_name(self) -> str:
        return type(next(iter(self.data.values()))).__name__
    
    def _get_time_steps(self, snap_ids: list, wrt: str = "scale_factor") -> np.ndarray:
        """Retrieve scale factors for the given snapshot IDs."""
        try:
            return np.asarray(
                self.moments.map_by_attribute(
                    key_attr="snapshot_id",
                    attr_value=snap_ids,
                    return_attr=wrt,
                ),
                dtype=float,
            )
        except Exception:
            return np.array(
                [getattr(self.moments[sid], wrt) for sid in snap_ids], 
                dtype=float
            )

    