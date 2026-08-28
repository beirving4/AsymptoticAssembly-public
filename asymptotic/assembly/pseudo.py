from __future__ import annotations


from attrs import define, field

from ..mass_def.base import MassDefinitionDependentType



@define(slots=True)
class PseudoEvolutionHistory: # similar to mass definition value
    pass 

@define(slots=True)
class PseudoEvolutionHistories:  # similar to mass definition
    histories: np.ndarray


@define(slots=True)
class PseudoAssemblyHistories(MassDefinitionDependentType): # similar to MassDefinitions
    fof: PseudoEvolutionHistories | None = field(default=None)
    subfind: PseudoEvolutionHistories | None = field(default=None)
    crit200: PseudoEvolutionHistories | None = field(default=None)
    crit500: PseudoEvolutionHistories | None = field(default=None)
    mean200: PseudoEvolutionHistories | None = field(default=None)
    virial: PseudoEvolutionHistories | None = field(default=None)
    splashback: PseudoEvolutionHistories | None = field(default=None)
    asymptotic: PseudoEvolutionHistories | None = field(default=None)