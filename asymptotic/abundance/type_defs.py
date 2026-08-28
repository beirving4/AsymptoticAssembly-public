from __future__ import annotations

from enum import StrEnum

class AbundanceType(StrEnum):
    NUMBER_DENSITY  = "number_density"    # ID: 1
    DIFFERENTIAL    = "differential"      # ID: 2
    NORMALIZED      = "normalized"        # ID: 3
    CUMULATIVE      = "cumulative"        # ID: 4
    MULTIPLICITY    = "multiplicity"      # ID: 5

    @property
    def id(self) -> int:
        # position in definition order, starting at 1
        return list(self.__class__).index(self) + 1
    

class RelationType(StrEnum):
    HALO_MASS_FUNCTION = "mass"      # ID: 1

    @property
    def id(self) -> int:
        # position in definition order, starting at 1
        return list(self.__class__).index(self) + 1
    
    @property
    def bin_name(self) -> str:
        return f"{self.value}_bins"
    
