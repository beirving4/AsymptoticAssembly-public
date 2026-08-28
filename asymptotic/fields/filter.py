from __future__ import annotations

import numpy as np, pdb

from enum import StrEnum
from attrs import define, field


class FilterFunctionType(StrEnum): 
    TOPHAT = "tophat"
    GAUSSIAN = "gaussian"
    SHARP_K = "sharp_k"

def topHat(kR: float | np.ndarray) -> float | np.ndarray:
    return (3 * (np.sin(kR) - kR * np.cos(kR))) / (kR) ** 3

def gaussian(kR: float | np.ndarray) -> float | np.ndarray:
    return np.exp(-0.5 * (kR) ** 2)

def sharpK(kR: float | np.ndarray) -> float | np.ndarray:
    return np.heaviside(1.0 - kR, 1.0)


@define(slots=True)
class FilterFunction: 
    name: FilterFunctionType

    def __call__(self, kR: float | np.ndarray) -> float | np.ndarray:
        if self.name == FilterFunctionType.TOPHAT: 
            return topHat(kR)
        elif self.name == FilterFunctionType.GAUSSIAN: 
            return gaussian(kR)
        elif self.name == FilterFunctionType.SHARP_K: 
            return sharpK(kR)
        else: 
            raise ValueError(f"Invalid filter function type: {self.name}")


def get_filter_function(name: str) -> FilterFunction:
    try: 
        filter_type = FilterFunctionType(name)
    except ValueError as e: 
        raise ValueError(f"Invalid filter function name: {name}") from e
    return FilterFunction(name=filter_type)