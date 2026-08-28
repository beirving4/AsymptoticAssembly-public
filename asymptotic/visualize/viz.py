from __future__ import annotations

from attrs import define, field 
from abc import abstractmethod, ABC


@define
class VisualizerConfig:
    '''
    Put general matplotlib controls in here like text sizes, tick size,
    dpi, font style, etc. 
    '''


@define
class Visualizer(ABC): 
    '''
    Let this handle all of the plotting methods for a class. That way we take it out 
    of the responsiblity of a given class like MassFunction, MassAssembly, Clustering, etc.
    '''
    cfg: VisualizerConfig