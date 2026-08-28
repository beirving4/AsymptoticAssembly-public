from __future__ import annotations

import numpy as np
from attrs import define
from typing import Iterable
from collections.abc import Collection

@define(slots=True)
class ParticleIdentifier:
    ids: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> int:
        return self.ids[idx]
    
    def get_subset(self, ids: Collection[int]) -> ParticleIdentifier:
        return ParticleIdentifier(ids=self.get_contained_ids(ids))
    
    def check_if_particle_exists(self, id: int) -> bool:
        return id in self.ids

    # Check if list/array of ids exist in collection
    def check_if_particles_exist(self, ids: Iterable[int]) -> bool:
        # May use assume_unique=True for speed up
        return np.isin(np.array(ids), self.ids, assume_unique=True).all() # type: ignore
    
    def get_particle_index(self, id_number: int) -> int:
        if not self.check_if_particle_exists(id_number):
            raise ValueError(
                f"Particle with id {id_number} not found in collection"
            )
        return np.where(self.ids == id_number)[0][0]

    def get_contained_ids(self, ids: Collection[int]) -> np.ndarray:
        if not self.check_if_particles_exist(ids):
            # Only get particles that do exist in the collection
            all_ids = np.array(ids)
            unbound_ids = all_ids[~np.isin(all_ids, self.ids)]
            ids = all_ids[~np.isin(np.array(ids), unbound_ids)]
            pct_ids = (len(unbound_ids) / len(all_ids)) * 100
            print(
                f"Roughly {pct_ids:.2f}% of the {len(all_ids)} particles "
                f"don't exist in this collection."
            )
        return ids if isinstance(ids, np.ndarray) else np.asarray(ids)

    def get_index_locations(self, ids: Collection[int]) -> np.ndarray:
        particle_ids = self.get_contained_ids(ids)
        sorted_ids = np.argsort(self.ids)
        idxs = np.searchsorted(self.ids[sorted_ids], particle_ids)
        idxs[idxs == len(self.ids)] = 0
        matched_idxs = sorted_ids[idxs]
        return matched_idxs[np.isin(self.ids[matched_idxs], particle_ids)]