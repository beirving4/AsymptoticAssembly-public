from __future__ import annotations

import numpy as np, pdb

from copy import deepcopy
from abc import ABC, abstractmethod
from attrs import define, field, fields
from scipy.optimize import curve_fit
from collections.abc import Collection
from collections import OrderedDict, defaultdict

from .fit import CurveFitter


@define(slots=True)
class AbundanceCurveFitter(CurveFitter):


    def __call__(
            self, 
            peak_heights: np.ndarray, 
            multiplicity: np.ndarray,
            multiplicity_errors: np.ndarray | None = None,
            verbose: bool = False
        ) -> np.ndarray:
        
        ...