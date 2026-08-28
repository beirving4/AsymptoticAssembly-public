from __future__ import annotations

from attrs import define
from abc import ABC, abstractmethod


@define
class BoundedObjectTracker(ABC):
    ... 

@define
class FOFGroupTracker(BoundedObjectTracker):
    ...

@define
class SubhaloTracker(BoundedObjectTracker):
    ...