from __future__ import annotations

import random
import numpy as np, pdb 
import matplotlib.pyplot as plt

from pathlib import Path
from typing import ClassVar, Type
from attrs import define, field
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Collection
from scipy.interpolate import interp1d

from ..cosmo.model import Cosmology
from .triangle import CoverageTriangle
from ..simulation.evo import EvolutionData
from ..simulation.moments import MomentsInTime
from .type_defs import AbundanceType, RelationType
from ..fields.peaks import DensityPeaks, DensityPeaksEvo
from .compute import get_abundance_function_data, get_jackknifed_abundance_data
from ..model.mass_function import (
    AsymptoticAbundanceFit, AsymptoticAbundanceFitEvo
)

def get_abundance_data_idx(abundance_type: str) -> int:
    try:
        return AbundanceType(abundance_type).id
    except ValueError as e:
        raise ValueError(f"Invalid abundance type: {abundance_type}") from e
    
def get_full_nan_array(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=float)

@define(slots=True)
class AbundanceValue:
    peak_height: float
    number_density: float
    differential: float
    normalized: float
    cumulative: float
    multiplicity: float

    @property
    def in_log10(self) -> dict[str, float]:
        return {
            "peak_height" : np.log10(self.peak_height),
            "number_density" : np.log10(self.number_density),
            "differential" : np.log10(self.differential),
            "normalized" : (
                np.log10(self.normalized)
                if np.isfinite(self.normalized) else
                np.nan
            ),
            "cumulative" : np.log10(self.cumulative),
            "multiplicity" : (
                np.log10(self.multiplicity) 
                if np.isfinite(self.multiplicity) else 
                np.nan
            ),
        }
    
    @classmethod
    def for_normalizing(
            cls, 
            peak_height: float,
            cutoff_scale_factor: float,
            accumulation_history_array: np.ndarray
        ) -> AbundanceValue:

        mask = accumulation_history_array[0] >= cutoff_scale_factor
        masked_accum = accumulation_history_array[:, mask]

        def avg(idx: int) -> float:
            if np.all(np.isnan(masked_accum[idx])):
                return np.nan
            return float(10.0**np.nanmean(np.log10(masked_accum[idx])))

        return cls(
            peak_height=peak_height,
            number_density=avg(1),
            differential=avg(2),
            normalized=avg(3),
            cumulative=avg(4),
            multiplicity=avg(5)
        )
    

            

@define(slots=True)
class Abundances: 
    histogram: np.ndarray
    histogram_density: np.ndarray
    number_density: np.ndarray
    differential: np.ndarray
    cumulative: np.ndarray
    normalized: np.ndarray = field(default=np.empty(0), repr=False)
    multiplicity: np.ndarray = field(default=np.empty(0), repr=False)


    def __attrs_post_init__(self) -> None:

        if self.normalized.size == 0:
            self.normalized = get_full_nan_array(self.histogram.shape)

        if self.multiplicity.size == 0:
            self.multiplicity = get_full_nan_array(self.histogram.shape)


    def __getitem__(self, idx: int) -> AbundanceValue:
        return AbundanceValue(
            peak_height=self.histogram[idx],
            number_density=self.number_density[idx],
            differential=self.differential[idx],
            normalized=self.normalized[idx],
            cumulative=self.cumulative[idx],
            multiplicity=self.multiplicity[idx]
        )

    @classmethod
    def from_data(cls, data: dict[str, np.ndarray]) -> Abundances:

        return cls(
            histogram=data["histogram"],
            histogram_density=data["histogram_density"],
            number_density=data["number_density"],
            differential=data["differential"],
            cumulative=data["cumulative"],
            normalized=(
                data["normalized"] if "normalized" in data else np.empty(0)
            ),
            multiplicity=(
                data["multiplicity"] if "multiplicity" in data else np.empty(0)
            )
        )
    
    @classmethod
    def from_error_data(cls, data: dict[str, np.ndarray]) -> Abundances | None:
        return cls.from_data(data) if (data.get("errors") is None) else None
    
    @classmethod
    def from_fit(
            cls, 
            fit: AsymptoticAbundanceFit, # will eventually change once other fits are added
            histogram: np.ndarray,
            histogram_density: np.ndarray,
            peak_heights: np.ndarray, 
            samples: np.ndarray, 
        ) -> Abundances:

        def fitted(mf_type: str) -> np.ndarray:
            return np.asarray(fit(peak_heights, samples, mf_type=mf_type))

        return cls(
            histogram=histogram,
            histogram_density=histogram_density,
            number_density=fitted("number_density"),
            differential=fitted("differential"),
            normalized=fitted("normalized"),
            cumulative=fitted("cumulative"),
            multiplicity=fitted("multiplicity")
        )
    
    @classmethod
    def for_residuals(cls, data: Abundances, fitted: Abundances) -> Abundances:

        def residual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            if (not np.isfinite(x).all()) or (not np.isfinite(y).all()):
                return get_full_nan_array(x.shape)
            return 10.0**(np.log10(x) - np.log10(y)) - 1.0

        return cls(
            histogram=data.histogram,
            histogram_density=data.histogram_density,
            number_density=residual(data.number_density, fitted.number_density),
            differential=residual(data.differential, fitted.differential),
            normalized=residual(data.normalized, fitted.normalized),
            cumulative=residual(data.cumulative, fitted.cumulative),
            multiplicity=residual(data.multiplicity, fitted.multiplicity)
        )
    
    @classmethod
    def join(cls, abundances: Collection[Abundances]) -> Abundances:
        
        def concatenate(prop: str) -> np.ndarray:
            return np.concatenate([
                getattr(abundance, prop) for abundance in abundances
            ])
        
        return cls(
            histogram=concatenate("histogram"),
            histogram_density=concatenate("histogram_density"),
            number_density=concatenate("number_density"),
            differential=concatenate("differential"),
            normalized=concatenate("normalized"),
            cumulative=concatenate("cumulative"),
            multiplicity=concatenate("multiplicity")
        )
    
    def masked(self, mask: np.ndarray) -> Abundances:
        return Abundances(
            histogram=self.histogram[mask],
            histogram_density=self.histogram_density[mask],
            number_density=self.number_density[mask],
            differential=self.differential[mask],
            normalized=self.normalized[mask],
            cumulative=self.cumulative[mask],
            multiplicity=self.multiplicity[mask]
        )
    
    def interpolants(self, samples: np.ndarray, is_logged: bool = False) -> dict[str, interp1d]:
        log_samples = samples if is_logged else np.log10(samples)
        interpolants = {
            "number_density" : interp1d(log_samples, np.log10(self.number_density)),
            "differential" : interp1d(log_samples, np.log10(self.differential)),
            "cumulative" : interp1d(log_samples, np.log10(self.cumulative)),
        }

        if np.isfinite(self.histogram).all():
            interpolants["normalized"] = interp1d(log_samples, np.log10(self.normalized))

        if np.isfinite(self.multiplicity).all():
            interpolants["multiplicity"] = interp1d(log_samples, np.log10(self.multiplicity))

        return interpolants
    
    @property
    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "histogram" : self.histogram,
            "histogram_density" : self.histogram_density,
            "number_density" : self.number_density,
            "differential" : self.differential,
            "normalized" : self.normalized,
            "cumulative" : self.cumulative,
            "multiplicity" : self.multiplicity
        }
    
    @property
    def to_numpy(self) -> np.ndarray:
        return np.column_stack([
            self.histogram, 
            self.histogram_density, 
            self.number_density, 
            self.differential, 
            self.normalized, 
            self.cumulative, 
            self.multiplicity
        ])

@define(slots=True)
class AbundancesEvo(EvolutionData):
    data: OrderedDict[int, Abundances]

    def __repr__(self) -> str:
        return super().__repr__() 
    

@define(slots=True)
class AbundanceRelation(ABC):
    sample_bins: dict[str, np.ndarray]
    estimate: Abundances
    errors: Abundances | None = field(default=None, repr=False)
    peak_bins: dict[str, np.ndarray] | None = field(default=None, repr=False)
    peak_jacobian: np.ndarray | None = field(default=None, repr=False)

    folds_info: dict[int, dict[str, np.ndarray]] | None = field(default=None, repr=False)

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @property
    def samples(self) -> np.ndarray:
        return self.sample_bins["centers"]
    
    @property
    def peak_heights(self) -> np.ndarray:
        return self.peak_bins["centers"] if self.peak_bins is not None else np.empty(0)
    
    @property
    def as_dict(self) -> dict[str, np.ndarray]:
        values = self.estimate.as_dict | {"samples" : self.samples}

        if self.peak_bins is not None:
            values["peak_heights"] = self.peak_heights

        if self.peak_jacobian is not None:
            values["jacobian"] = self.peak_jacobian

        if self.errors is not None:

            for key, value in self.errors.as_dict.items():
                values[f"{key}_error"] = value

        if self.folds_info is not None:
            values["fold_samples"] = self.folds_info

        return values
    
    @property
    def to_numpy(self) -> np.ndarray:
        array = self.estimate.to_numpy

        if self.peak_bins is not None:
            array = np.column_stack([array, self.peak_heights])

        if self.errors is not None:
            array = np.column_stack([array, self.errors.to_numpy])
        
        return array
    
    @property
    def has_jacobian(self) -> bool:
        return self.peak_jacobian is not None and self.peak_jacobian.size > 0
    
    @property
    def has_peak_bins(self) -> bool:
        return self.peak_bins is not None and self.peak_bins["centers"].size > 0
    
    @property
    def coefficient_of_variations(self) -> np.ndarray:
        if self.errors is None:
            return np.empty(0)
        return self.errors.histogram / self.estimate.histogram
    

    # Trapped mass by bin generalization ??? 


    @classmethod
    def from_data(
            cls, data: dict[str, dict[str, np.ndarray]], bin_name: str
        ) -> AbundanceRelation:
        
        if error_data := data.get("errors"):
            errors = Abundances.from_error_data(error_data)
        else:
            errors = None

        if data["estimate"]["histogram"] is None:
            raise ValueError("No mass function data")
        
        return cls(
            sample_bins=data[bin_name],
            estimate=Abundances.from_data(data["estimate"]),
            errors=errors,
            peak_bins={
                "edges" : data["peaks"]["edges"], 
                "centers" : data["peaks"]["centers"]
            },
            peak_jacobian=data["peaks"]["jacobian"],
            folds_info=data.get("fold_samples")
        )

    @classmethod
    def from_fold_data(
            cls, 
            data: dict[str, dict[str, np.ndarray]], 
            sample_bins: dict[str, np.ndarray],
        ) -> AbundanceRelation:

        if data.get("histogram") is None:
            raise ValueError("No mass function fold data")

        return cls(
            sample_bins=sample_bins,
            estimate=Abundances.from_data(data),
            errors=None,
            peak_bins={
                "edges" : data["peaks"]["edges"], 
                "centers" : data["peaks"]["centers"]
            },
            peak_jacobian=data["peaks"]["jacobian"],
            folds_info=None
        )
    

    @classmethod
    def join(cls, relations: Collection[AbundanceRelation]) -> AbundanceRelation:

        if len(relations) == 0:
            raise ValueError("No relations provided to join")

        if len(relations) == 1:
            # If only one relation, return it directly
            return list(relations)[0]

        sample_bin_edges, sample_bin_centers = [], []
        peak_bin_edges, peak_bin_centers = [], []
        estimate, errors, jacobians = [], [], []
        joint_fold_info = defaultdict(lambda: defaultdict(list))

        for relation in relations:
            sample_bin_edges.append(relation.sample_bins["edges"])
            sample_bin_centers.append(relation.sample_bins["centers"])
            if relation.peak_bins is not None:
                peak_bin_edges.append(relation.peak_bins["edges"])
                peak_bin_centers.append(relation.peak_bins["centers"])

            if relation.peak_jacobian is not None:
                jacobians.append(relation.peak_jacobian)

            estimate.append(relation.estimate)
            errors.append(relation.errors)

            if relation.folds_info is not None:
                for fold_id, fold_entry in relation.folds_info.items():
                    for variable_name, data in fold_entry.items():
                        if isinstance(data, np.ndarray):
                            joint_fold_info[fold_id][variable_name].append(data)


        if not sample_bin_edges:
            raise ValueError("No relations provided to join")

        if len(sample_bin_edges) == 1:
            # If only one relation, return it directly
            return list(relations)[0]
        
        sample_bins = {
            "edges" : np.concatenate(sample_bin_edges),
            "centers" : np.concatenate(sample_bin_centers)
        }

        if peak_bin_edges:
            peak_bins = {
                "edges" : np.concatenate(peak_bin_edges),
                "centers" : np.concatenate(peak_bin_centers)
            }
            peak_jacobian = np.concatenate(jacobians) if jacobians else None
        else:
            peak_bins, peak_jacobian = None, None

        if joint_fold_info:
            for fold_id in joint_fold_info:
                for variable_name, data in joint_fold_info[fold_id].items(): 
                    # May need to sort these later, just concatenating for now
                    joint_fold_info[fold_id][variable_name] = np.concatenate(data)

                joint_fold_info[fold_id]["peaks"] = peak_bins
                joint_fold_info[fold_id]["peaks"]["jacobian"] = peak_jacobian

        
        return cls(
            sample_bins=sample_bins,
            estimate=Abundances.join(estimate),
            errors=Abundances.join(errors) if errors else None,
            peak_bins=peak_bins,
            peak_jacobian=peak_jacobian,
            folds_info=joint_fold_info
        )
    
    def masked(self, mask: np.ndarray) -> AbundanceRelation:
        
        bin_centers = self.samples[mask]
        delta = np.diff(np.log10(self.samples)).mean()
        sample_bin_edges = np.log10(bin_centers) - 0.5 * delta
        bin_edges = np.append(sample_bin_edges, sample_bin_edges[-1] + delta)

        if self.peak_bins is None:
            peak_bins = None
            peak_jacobian = None
        else:
            peak_bin_centers = self.peak_bins["centers"][mask]
            peak_delta = np.diff(np.log10(self.peak_bins["centers"])).mean()
            peak_bin_edges = np.log10(peak_bin_centers) - 0.5 * peak_delta
            peak_edges = np.append(peak_bin_edges, peak_bin_edges[-1] + peak_delta)
            peak_bins = {"edges" : peak_edges, "centers" : peak_bin_centers}
            peak_jacobian = self.peak_jacobian[mask] if self.has_jacobian else None

        return self.__class__(
            sample_bins = {"edges" : bin_edges, "centers" : bin_centers},
            estimate=self.estimate.masked(mask),
            errors=self.errors.masked(mask) if self.errors is not None else None,
            peak_bins=peak_bins,
            peak_jacobian=peak_jacobian,
            folds_info=get_masked_folds_info(
                folds_info=self.folds_info, 
                mask=mask, 
                peak_bins=peak_bins,
                peak_jacobian=peak_jacobian
            )
        )
    
    @abstractmethod
    def display(self, *args, **kwargs) -> plt.Axes | None:
        ...

@define(slots=True)
class AbundanceRelationEvo(EvolutionData):
    
    data: OrderedDict[int, AbundanceRelation]
    relation_cls: ClassVar[Type[AbundanceRelation]] = AbundanceRelation

    def __repr__(self) -> str:
        return super().__repr__()


@define(slots=True)
class AbundanceData(ABC):

    comoving_volume: float
    relation_type: RelationType = field(repr=False)
    
    raw: AbundanceRelation
    cleaned: AbundanceRelation | None = field(default=None, repr=False)
    triangle: CoverageTriangle | None = field(default=None, repr=False)

    peaks: DensityPeaks | None = field(default=None, repr=False)

    data: Abundances = field(init=False, repr=False)
    fitted: Abundances | None = field(init=False, repr=False)
    residuals: Abundances | None = field(init=False, repr=False)

    run_fit: bool = field(default=False, repr=False)
    fit: AsymptoticAbundanceFit = field(init=False, repr=False)


    # ← this is the “hook” that determines what kind of AbundanceRelation
    # will be constructed by from_data
    relation_cls: ClassVar[Type[AbundanceRelation]] = AbundanceRelation

    def __attrs_post_init__(self) -> None:

        if self.cleaned is None:
            self.data = Abundances(**self.raw.estimate.as_dict)
        else:
            self.data = Abundances(**self.cleaned.estimate.as_dict)

        if self.run_fit:
            self.fit_model()
        else:
            self.initialize_fit() 

        if (self.peaks is not None) and self.has_optimized_fit:

            self.fitted = Abundances.from_fit(
                fit=self.fit, 
                histogram=self.data.histogram, 
                histogram_density=self.data.histogram_density,
                peak_heights=self.peak_heights, 
                samples=self.samples
            )

            self.set_residuals()

    
    @abstractmethod
    def __repr__(self) -> str:
        '''
        Return a string representation of the abundance data.
        This should include the type of abundance, the sample bins, and any other relevant information.
        '''
        ...

    @classmethod
    def from_data(
            cls, 
            samples: np.ndarray | list[np.ndarray],
            box_size: float,
            redshift: float,
            present_day_density: float,
            relation_type: str,
            bin_count: int = 50,
            transfer: str = "eisenstein98",
            corrections: bool = True,
            is_comoving_volume: bool = True,
            run_jackknife: bool = False,
            sample_bins: dict[str, np.ndarray] | None = None,
            run_fit: bool = False,
            use_ps_file: bool = False,
            ps_file_path: Path | None = None,
            cosmo: Cosmology | None = None,
            return_fold_samples: bool = False,
        ) -> AbundanceData:

        L = box_size if is_comoving_volume else box_size / (1.0 + redshift)
        comoving_volume = L**3

        peaks = DensityPeaks(
            redshift=redshift, 
            present_day_density=present_day_density,
            transfer=transfer, 
            corrections=corrections,
            use_ps_file=use_ps_file,
            ps_file_path=ps_file_path,
            cosmo=cosmo
        )

        abundance_data = setup_jackknifed_abundance_data(
            samples=samples,
            comoving_volume=comoving_volume,
            relation_type=relation_type,
            peaks=peaks,
            present_day_density=present_day_density,
            bin_count=bin_count,
            sample_bins=sample_bins,
            run_jackknife=run_jackknife,
            return_fold_samples=return_fold_samples
        )

        setup_multiplicity_data(
                abundance_data=abundance_data,
                redshift=redshift,
                peaks=peaks,
                relation_type=relation_type
        )

        # If return_fold_samples, need to also setup the multiplicity for each fold
        if return_fold_samples:
            setup_fold_multiplicity_data(
                abundance_data=abundance_data,
                redshift=redshift,
                peaks=peaks,
                relation_type=relation_type
            )



        return cls(
            comoving_volume=comoving_volume,
            relation_type=RelationType(relation_type),
            # ← use cls.relation_cls here, NOT the hard-coded AbundanceRelation
            raw=cls.relation_cls.from_data(
                data=abundance_data, 
                bin_name=RelationType(relation_type).bin_name,
            ),
            peaks=peaks,
            run_fit=run_fit,
        )

    @property
    def folds_info(self) -> dict[int, dict[str, np.ndarray]] | None:
        return self.raw.folds_info if self.not_cleaned else self.cleaned.folds_info

    @property
    def fold_instances(self) -> dict[int, AbundanceData]:
        if self.folds_info is None:
            raise ValueError("No fold information available")
        
        instances = {} 
        for fold_idx in self.folds_info:

            instances[fold_idx] = self.__class__(
                comoving_volume=self.comoving_volume / len(self.folds_info),
                relation_type=self.relation_type,
                raw=self.relation_cls.from_fold_data(
                    data=self.raw.folds_info[fold_idx], 
                    sample_bins=self.raw.sample_bins
                ),
                peaks=self.peaks,
                run_fit=False,
            )

            if self.not_cleaned and self.has_triangle:
                
                try:
                    instances[fold_idx].cleaned = instances[fold_idx].raw.masked(
                        self.triangle.abundance_bounds
                    )  
                    clean_dict = instances[fold_idx].cleaned.estimate.as_dict
                    instances[fold_idx].data = Abundances(**clean_dict)
                except (ValueError, IndexError):
                    print("Warning: Could not clean fold data, setting to None")
                    instances[fold_idx].cleaned = None

            if not self.not_cleaned:

                instances[fold_idx].cleaned = self.relation_cls.from_fold_data(
                    data=self.cleaned.folds_info[fold_idx], 
                    sample_bins=self.cleaned.sample_bins
                )

            
        return instances

    def get_masked_version(
            self, mask: np.ndarray, run_fit: bool = False
        ) -> AbundanceData:

        return self.__class__(
            comoving_volume=self.comoving_volume,
            relation_type=self.relation_type,
            raw=self.raw,
            cleaned=(
                self.cleaned.masked(mask)
                if self.cleaned is not None
                else None
            ),
            peaks=self.peaks,
            run_fit=run_fit,
            triangle=self.triangle,
        )
    
    @classmethod
    def join(
            cls, abundances: dict[int, AbundanceData] | Collection[AbundanceData]
        ) -> AbundanceData:

        if isinstance(abundances, dict):
            return join_abundances_by_dict(abundances)
        else: 
            return join_abundances_by_collection(abundances)
        
    
    @abstractmethod
    def initialize_fit(self) -> None:
        ...

    def fit_model(self, verbose: bool = False) -> None:    
        self.initialize_fit()
        if self.run_fit:
            self.fit.fit(
                peak_heights=self.peak_heights, 
                multiplicity=self.data.multiplicity,
                multiplicity_errors=(
                    self.errors.multiplicity
                    if self.errors is not None 
                    else None
                ),
                verbose=verbose
            )

    def set_residuals(self) -> None:
        if self.fitted is None:
            raise ValueError("Need fitted mass function to set residuals!")
        self.residuals = Abundances.for_residuals(self.data, self.fitted)


    def set_clean_abundance_relationship(
            self,
            min_sample_value: float, 
            min_above_max_mass_bin_count: int,
            from_histogram: bool = True,
            verbose: bool = False
        ) -> None:

        triangle_args = {
            "bin_centers" : self.samples,
            "comoving_volume" : self.comoving_volume,
            "min_count" : min_above_max_mass_bin_count,
            "min_sample_value" : min_sample_value,
            "min_peak_height" : (
                self.peak_heights.min() 
                if self.peaks is not None else 
                0.0
            ),
            "max_peak_height" : (
                self.peak_heights.max() 
                if self.peaks is not None else 
                np.inf
            ),
        }

        if from_histogram:

            self.triangle = CoverageTriangle.from_histogram(
                histogram=self.raw.estimate.histogram,
                **triangle_args
            )

        else:

            self.triangle = CoverageTriangle.from_abundance(
                log10_differential=np.log10(self.raw.estimate.differential),
                **triangle_args
            )

        self.cleaned = self.raw.masked(self.triangle.abundance_bounds)
        # self.data = Abundances(**self.cleaned.estimate.as_dict)
        self.data = self.cleaned.estimate
        self.fit_model(verbose=verbose)
        self.fitted = Abundances.from_fit(
            fit=self.fit,
            histogram=self.data.histogram,
            histogram_density=self.data.histogram_density,
            peak_heights=self.peak_heights,
            samples=self.samples,
        )
        
        self.set_residuals() 

    
    @property
    def not_cleaned(self) -> bool:
        return self.cleaned is None
    
    @property
    def no_peaks(self) -> bool:
        return self.peaks is None
    
    @property
    def no_errors(self) -> bool:
        return self.errors is None 
    

    @property
    def has_triangle(self) -> bool:
        return self.triangle is not None
    

    @property
    def samples(self) -> np.ndarray:
        return self.raw.samples if self.not_cleaned else self.cleaned.samples
    
    @property
    def sample_bins(self) -> dict[str, np.ndarray]:
        return (
            self.raw.sample_bins if self.not_cleaned else self.cleaned.sample_bins
        )
    
    @property
    def peak_bins(self) -> dict[str, np.ndarray] | None:
        return (
            self.raw.peak_bins if self.not_cleaned else self.cleaned.peak_bins
        )
    
    @property
    def peak_heights(self) -> np.ndarray:
        return (
            self.raw.peak_heights 
            if self.not_cleaned else 
            self.cleaned.peak_heights
        )
    
    @property
    def peak_jacobian(self) -> np.ndarray | None:
        return (
            self.raw.peak_jacobian 
            if self.not_cleaned else 
            self.cleaned.peak_jacobian
        )
    
    @property
    def estimate(self) -> Abundances:
        return (
            self.raw.estimate if self.not_cleaned else self.cleaned.estimate
        )
    
    @property
    def errors(self) -> Abundances | None:
        return (
            self.raw.errors if self.not_cleaned else self.cleaned.errors
        )
    
    @property
    def has_fit(self) -> bool:
        return self.fit is not None
    
    @property
    def has_optimized_fit(self) -> bool:
        return self.has_fit and self.fit.is_fitted
    
    @property
    def as_dict(self) -> dict[str, np.ndarray]:
        return self.estimate.as_dict | {"peak_heights" : self.peak_heights}
    
    @property
    def to_numpy(self) -> np.ndarray:
        return (
            np.column_stack([self.estimate.to_numpy, self.peak_heights]) 
            if self.peak_heights is not None else 
            self.estimate.to_numpy
        )
    
    @property
    def interped_abundances(self) -> dict[str, interp1d]:
        return self.estimate.interpolants(self.samples)
    

    def get_interpolated_abundance_float_value(
            self, 
            value: float, 
            abundance_type: str,
            is_logged: bool = True
        ) -> float:

        if self.no_peaks:
            raise ValueError("No peaks to get interped abundance value!")
        
        interpolator = self.interped_abundances.get(abundance_type, None)
        if interpolator is None:
            raise ValueError(f"Abundance type {abundance_type} not available for interpolation!")
        
        log10_sample_value = value if is_logged else np.log10(value)
        interped_value = interpolator(log10_sample_value)
        
        return (
            10.0**interped_value 
            if np.isfinite(interped_value) else 
            np.nan
        )
    
    
    def jacobian_interpolant(self, wrt_mass: bool=True, from_masses: bool=True) -> interp1d:
        if self.no_peaks:
            raise ValueError("No peaks to build interpolators!")
        return self.peaks.jacobian_interpolant(
            values=self.samples, 
            wrt_mass=wrt_mass, 
            from_masses=from_masses
        )
    
    def get_interped_jacobian_value(
            self, 
            value: float, 
            wrt_mass: bool = True, 
            from_masses: bool = True,
            is_logged: bool = True
        ) -> float:

        if self.no_peaks:
            raise ValueError("No peaks to get interped jacobian value!")
        
        interpolator = self.jacobian_interpolant(
            wrt_mass=wrt_mass, from_masses=from_masses
        )
        log10_sample_value = value if is_logged else np.log10(value)
        return interpolator(log10_sample_value)

    
    @property
    def fitted_abundance_relation(self) -> AbundanceRelation:
        if self.fitted is None:
            raise ValueError("Need fitted abundance relation!")
        return self.relation_cls(
            sample_bins=self.sample_bins,
            estimate=self.fitted,
            peak_bins=self.peak_bins,
            peak_jacobian=self.peak_jacobian,
        )
    
    def get_interped_abundance_value(
            self, 
            value: float, 
            wrt_mass: bool = True,
            from_masses: bool = True,
            is_logged: bool = True
        ) -> AbundanceValue:

        if self.no_peaks:
            raise ValueError("No peaks to get interped abundance value!")

        log10_sample_value = value if is_logged else np.log10(value)
        jacobian_interp = self.jacobian_interpolant(
            wrt_mass=wrt_mass, 
            from_masses=from_masses
        )
        jacobian = jacobian_interp(log10_sample_value)

        interped_values = {}
        for key in ("number_density", "differential", "cumulative", "normalized"):
            interped_values[key] = self.get_interpolated_abundance_float_value(
                value=log10_sample_value, 
                abundance_type=key, 
                is_logged=True
            )

        interped_values["multiplicity"] = interped_values["normalized"] / jacobian

        return AbundanceValue(
            peak_height=self.peaks.heights( #type: ignore
                values=10.0**log10_sample_value,
                from_masses=from_masses
            ),
            **interped_values,
        )
    
    def get_fitted_abundance_value(
            self, 
            value: float, 
            relation_type: str, 
            wrt_mass: bool = True,
            is_logged: bool = True
        ) -> AbundanceValue:

        if (self.fitted is None) or not self.has_optimized_fit:
            raise ValueError("Need fitted abundance relations to get value!")
        
        if self.peaks is None:
            raise ValueError("Need peaks to get fitted abundance value!")
        
        log10_sample_value = value if is_logged else np.log10(value)
        sample_value = 10.0**log10_sample_value
        nu = self.peaks.heights(values=sample_value)

        fitted_values = get_fitted_values(
            nu=float(nu),
            target_value=sample_value,
            samples=self.samples,
            fitted_cumulative=self.fitted.cumulative,
            jacobian_value=self.get_interped_jacobian_value(
                value=log10_sample_value,
                wrt_mass=wrt_mass,
                is_logged=True
            ),
            fit=self.fit,
            relation_type=relation_type,
            present_day_density=self.peaks.present_day_density
        )

        return AbundanceValue(peak_height=float(nu), **fitted_values)


    def get_abundance_value(
            self,
            value: float,
            relation_type: str,
            from_masses: bool = True,
            wrt_mass: bool = True,
            is_logged: bool = True,
            for_data: bool = True
        ) -> AbundanceValue:

        if for_data:
            return self.get_interped_abundance_value(value, is_logged=is_logged)
        
        return self.get_fitted_abundance_value(
            value,
            relation_type=relation_type,
            is_logged=is_logged
        )


    
    @abstractmethod
    def display_fit(self, *args, **kwargs) -> plt.Axes | None:
        ... 
    
    @abstractmethod
    def display(self, *args, **kwargs) -> plt.Axes | None:
        ... 
            
    

@define(slots=True)
class AbundanceDataEvo(EvolutionData):

    data: OrderedDict[int, AbundanceData]
    fits: AsymptoticAbundanceFitEvo = field(init=False, repr=False)

    color_palette: str = field(default="viridis", repr=False)
    colormap: plt.cm.ColorMap = field(init=False, repr=False)

    relation_cls: ClassVar[Type[AbundanceRelation]] = AbundanceRelation
    solo_evo_class: ClassVar[Type[AbundanceRelationEvo]] = AbundanceRelationEvo

    def __attrs_post_init__(self) -> None:
        self.colormap = plt.cm.get_cmap(self.color_palette, len(self.data))


        try:
            snaps_with_fits = [
                snap_id for snap_id, abundances in self.data.items() 
                if abundances.has_optimized_fit
            ]
        except AttributeError as error:
            print("Error checking for optimized fits:", error)
            # pdb.set_trace()

        if not snaps_with_fits:
            self.fits = AsymptoticAbundanceFitEvo(
                moments=self.moments, 
                fits=OrderedDict(
                    (snap_id, self.data[snap_id].fit) 
                    for snap_id in self.moments.snapshot_ids
                )
            )

        elif len(snaps_with_fits) == 1:
            snap_with_fit = snaps_with_fits[0]
            self.fits = AsymptoticAbundanceFitEvo(
                moments=self.moments.get_subset_by_attribute(
                    key_attr="snapshot_id",
                    attr_value=snap_with_fit
                ),
                fits=OrderedDict({snap_with_fit : self.data[snap_with_fit].fit})
            )

        else:
            self.fits = AsymptoticAbundanceFitEvo(
                moments=self.moments.get_subset_by_attribute(
                    key_attr="snapshot_id",
                    attr_value=snaps_with_fits
                ),
                fits=OrderedDict(
                    (snap_id, self.data[snap_id].fit) 
                    for snap_id in snaps_with_fits
                )
            )

    def __repr__(self) -> str:
        return super().__repr__()
    
    def get_data_evo(self, return_raw: bool = False) -> AbundanceRelationEvo:
        return self.solo_evo_class(
            moments=self.moments,
            data=OrderedDict(
                (snap_id, abundances.raw if return_raw else abundances.cleaned)
                for snap_id, abundances in self.data.items()
            )
        )
    
    @property
    def raw_data_evo(self) -> AbundanceRelationEvo:
        return self.get_data_evo(return_raw=True)
    
    @property
    def cleaned_data_evo(self) -> AbundanceRelationEvo:
        return self.get_data_evo(return_raw=False)
    
    @property
    def data_evo(self) -> AbundanceRelationEvo:
        return self.solo_evo_class(
            moments=self.moments,
            data=OrderedDict(
                (
                    snap_id, 
                    abundances.raw 
                    if abundances.not_cleaned else 
                    abundances.cleaned
                )
                for snap_id, abundances in sorted(self.data.items())
            )
        )

    @property
    def fold_instances(self) -> dict[int, AbundanceDataEvo]:
        instances = defaultdict(dict) # fold_id -> snap_id -> AbundanceData
        for snap_id, data in self.data.items():
            if data.folds_info is None: continue
            fold_instances = data.fold_instances
            for fold_id, fold_data in fold_instances.items():
                instances[fold_id][snap_id] = fold_data 

        fold_instances = {}
        for fold_id, fold_data_evo in instances.items():
            new_data = OrderedDict()
            new_moments = MomentsInTime()
            for snap_id, fold_data in sorted(fold_data_evo.items(), key=lambda x: x[0]):
                new_data[snap_id] = fold_data
                new_moments.add_moment(self.moments[snap_id])
            
            fold_instances[fold_id] = self.__class__(
                moments=new_moments,
                data=new_data
            )


        return fold_instances

    
    @property
    def peaks_evo(self) -> DensityPeaksEvo:
        return DensityPeaksEvo(
            moments=self.moments,
            data=OrderedDict(
                (k, v.peaks) for k, v in self.data.items()
                if v.peaks is not None
            )
        )
    
    @property
    def fitted_evo(self) -> AbundanceRelationEvo:
        data = OrderedDict()
        for snap_id, abundances in self.data.items():
            data[snap_id] = self.relation_cls(
                sample_bins=abundances.sample_bins,
                estimate=abundances.fitted,
                peak_bins=abundances.peak_bins,
                peak_jacobian=abundances.peak_jacobian
            )

        return self.solo_evo_class(moments=self.moments, data=data)

    def update_colormap(self, color_palette: str) -> None:
        self.color_palette = color_palette
        self.colormap = plt.cm.get_cmap(self.color_palette, len(self.data))


    @abstractmethod
    def display_fit_evolution(self, *args, **kwargs) -> plt.Axes | None:
        ...

    @abstractmethod
    def display(self, *args, **kwargs) -> tuple[plt.Axes | None, plt.Axes | None] | None:
        ... 

def setup_jackknifed_abundance_data(
        samples: np.ndarray | list[np.ndarray],
        comoving_volume: float,
        relation_type: str = "mass",
        peaks: DensityPeaks | None = None,
        present_day_density: float | None = None,
        bin_count: int | None = None,
        sample_bins: dict[str, np.ndarray] | None = None,
        run_jackknife: bool = True,
        return_fold_samples: bool = False,
    ) -> dict[str, dict[str, np.ndarray]]:


    bin_name = RelationType(relation_type).bin_name

    if run_jackknife:

        data_samples = samples if isinstance(samples, list) else [samples]

        abundance_data = get_jackknifed_abundance_data(
            samples=data_samples,
            comoving_volume=comoving_volume,
            relation_type=relation_type,
            present_day_density=present_day_density,
            bin_count=bin_count,
            samples_bins=sample_bins,
            return_fold_samples=return_fold_samples
        )

        if peaks is not None:
            abundance_data["errors"]["multiplicity"] = peaks.multiplicity(
                number_density=abundance_data["errors"]["histogram_density"],
                values=abundance_data[bin_name]["centers"],
            )

    else:
        
        data_samples = (
            samples if isinstance(samples, np.ndarray) else np.hstack(samples)
        )

        abundance_data = get_abundance_function_data(
            samples=data_samples,
            comoving_volume=comoving_volume,
            relation_type=relation_type,
            present_day_density=present_day_density,
            bin_count=bin_count,
            samples_bins=sample_bins,
        )

    return abundance_data


def setup_multiplicity_data(
        abundance_data: dict[str, dict[str, np.ndarray]],
        redshift: float,
        relation_type: str,
        peaks: DensityPeaks,
    ) -> None:

    bin_name = RelationType(relation_type).bin_name

    if redshift >= -0.99:
        abundance_data["estimate"]["multiplicity"] = peaks.multiplicity(
            number_density=abundance_data["estimate"]["histogram_density"],
            values=abundance_data[bin_name]["centers"],
        )

        abundance_data["peaks"] = {
            "edges" : peaks.heights(abundance_data[bin_name]["edges"]),
            "centers" : peaks.heights(abundance_data[bin_name]["centers"]),
            "jacobian" : peaks.jacobian(abundance_data[bin_name]["centers"])
        }
    else:
        abundance_data["estimate"]["multiplicity"] = np.empty_like(
            abundance_data["estimate"]["histogram_density"]
        )
        abundance_data["peaks"] = {
            "edges" : np.empty_like(abundance_data[bin_name]["edges"]),
            "centers" : np.empty_like(abundance_data[bin_name]["centers"]),
            "jacobian" : np.empty_like(abundance_data[bin_name]["centers"])
        }


# New function: setup_fold_multiplicity_data
def setup_fold_multiplicity_data(
        abundance_data: dict[str, dict[str, np.ndarray]],
        redshift: float,
        relation_type: str,
        peaks: DensityPeaks,
    ) -> None:
    """Add per-fold multiplicity and peaks blocks in-place.

    Expects abundance_data["fold_samples"] to be a dict[int, dict[str, np.ndarray]]
    where each value already contains keys produced by _compute_abundance_dict,
    e.g. "histogram_density", etc.
    """

    # If no per-fold outputs, nothing to do
    if "fold_samples" not in abundance_data:
        return

    bin_name = RelationType(relation_type).bin_name

    centers = abundance_data[bin_name]["centers"]
    edges = abundance_data[bin_name]["edges"]

    # Update each fold **in place** by writing through the nested dict
    for fold_id, fold_entry in list(abundance_data["fold_samples"].items()):

        if redshift >= -0.99:
            # multiplicity for this fold using its histogram_density
            abundance_data["fold_samples"][fold_id]["multiplicity"] = peaks.multiplicity(
                number_density=fold_entry["histogram_density"],
                values=centers,
            )

            # per-fold peaks block mirroring setup_multiplicity_data
            abundance_data["fold_samples"][fold_id]["peaks"] = {
                "edges":   peaks.heights(edges),
                "centers": peaks.heights(centers),
                "jacobian": peaks.jacobian(centers),
            }
        else:
            # Below threshold: empty arrays with the correct shapes
            abundance_data["fold_samples"][fold_id]["multiplicity"] = np.empty_like(
                fold_entry["histogram_density"]
            )
            abundance_data["fold_samples"][fold_id]["peaks"] = {
                "edges":   np.empty_like(edges),
                "centers": np.empty_like(centers),
                "jacobian": np.empty_like(centers),
            }
            

def join_abundances_by_dict(ds: dict[int, AbundanceData]) -> AbundanceData:
    comoving_volume = max(data.comoving_volume for data in ds.values())
    ref_data = random.choice(list(ds.values()))

    if any(data.peaks != ref_data.peaks for data in ds.values()):
        raise ValueError("Peaks must be the same!")
    
    if all(data.cleaned is None for data in ds.values()):
        cleaned = None
    else:
        cleaned = ref_data.raw.__class__.join([
            data.cleaned for data in ds.values() 
            if data.cleaned is not None
        ])

    return ref_data.__class__(
        comoving_volume=comoving_volume,
        relation_type=ref_data.relation_type,
        raw=ref_data.raw.__class__.join([data.raw for data in ds.values()]),
        cleaned=cleaned,
        peaks=ref_data.peaks,
        run_fit=True,
    )

def join_abundances_by_collection(ds: Collection[AbundanceData]) -> AbundanceData:
    comoving_volume = max(data.comoving_volume for data in ds)
    ref_data = random.sample(list(ds), 1)[0]

    if any(data.peaks != ref_data.peaks for data in ds):
        raise ValueError("Peaks must be the same!")
    
    if all(data.cleaned is None for data in ds):
        cleaned = None
    else:
        cleaned = ref_data.raw.__class__.join([
            data.cleaned for data in ds if data.cleaned is not None
        ])

    return ref_data.__class__(
        comoving_volume=comoving_volume,
        relation_type=ref_data.relation_type,
        raw=ref_data.raw.__class__.join([data.raw for data in ds]),
        cleaned=cleaned,
        peaks=ref_data.peaks,
        run_fit=True,
    )


def get_fitted_values(
        nu: float, 
        target_value: float,
        samples: np.ndarray, 
        fitted_cumulative: np.ndarray,
        jacobian_value: float,
        fit: AsymptoticAbundanceFit,
        relation_type: str,
        present_day_density: float
    ) -> dict[str, float]:
    '''
    Get the fitted values for a given nu and sample value.
    '''
    
    match relation_type:
        case RelationType.HALO_MASS_FUNCTION.value:
            return get_fitted_mass_function_value(
                nu=nu, 
                target_value=target_value,
                samples=samples, 
                fitted_cumulative=fitted_cumulative,
                jacobian_value=jacobian_value,
                fit=fit,
                present_day_density=present_day_density
            )
        case _:
            raise ValueError(
                f"Unknown relation type: {relation_type}"
                f" (expected 'mass')"
            )
    

def get_fitted_mass_function_value(
        nu: float, 
        target_value: float,
        samples: np.ndarray, 
        fitted_cumulative: np.ndarray,
        jacobian_value: float,
        fit: AsymptoticAbundanceFit,
        present_day_density: float
    ) -> dict[str, float]:

    f_nu = fit(nu, masses=target_value, mf_type="multiplicity")
    normalized_n = f_nu * jacobian_value
    dndlnM = (present_day_density / target_value) * normalized_n

    cumulative_interp = interp1d(
        np.log10(samples),
        np.log10(fitted_cumulative),
        kind="linear",
    )

    return {
        "number_density" : float(dndlnM / target_value),
        "differential" : float(dndlnM),
        "normalized" : float(normalized_n),
        "cumulative" : 10.0**cumulative_interp(np.log10(target_value)),
        "multiplicity" : float(f_nu)
    }
    
def get_masked_folds_info(
        folds_info: dict[int, dict[str, np.ndarray]] | None, 
        mask: np.ndarray,
        peak_bins: dict[str, np.ndarray], 
        peak_jacobian: np.ndarray | None = None
    ) -> dict[int, dict[str, np.ndarray]] | None:

    if folds_info is None:
        return None

    masked_folds_info = defaultdict(dict)
    for fold_id, fold_entry in folds_info.items():
        for variable_name, data in fold_entry.items():
            if isinstance(data, np.ndarray):
                masked_folds_info[fold_id][variable_name] = data[mask]

        masked_folds_info[fold_id]["peaks"] = peak_bins
        masked_folds_info[fold_id]["peaks"]["jacobian"] = peak_jacobian

    return masked_folds_info
        


# Now I just need to change the AbundanceRelation.masked method and the 
# AbundanceData.as_dict method ...