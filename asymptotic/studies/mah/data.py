from __future__ import annotations

import h5py
import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from attrs import define, field
from collections import OrderedDict, defaultdict
from scipy.stats import lognorm, norm, rv_continuous

from .helpers import *
from .stats import AssemblyStats, HistoryStats, Percentiles
from ...simulation.evo import EvolutionData


by_seed = lambda sim_name: int(sim_name.split('_')[-1].split('box')[-1])
by_size = lambda sim_name: int(sim_name.split('_')[0].split('L')[-1])
by_resolution = lambda sim_name: int(sim_name.split('_')[1].split('N')[-1])

@define
class AssemblyTimePrior:
    model: rv_continuous

    @classmethod
    def from_data(cls, times: np.ndarray) -> AssemblyTimePrior:
        fitted_model = lognorm(*lognorm.fit(np.log10(times)))
        return cls(model=fitted_model)

    @property
    def mean(self) -> float:
        return np.log(self.model.args[2])

    @property
    def standard_deviation(self) -> float: 
        return self.model.args[0]

    @property
    def log_base_shift(self) -> float: 
        return self.model.args[1]

    @property
    def visual_parameters(self) -> dict[str, float]: 
        return {r"$\mu$" : self.mean, r"$\sigma$" : self.standard_deviation}


@define
class FormationTimeData:
    times: np.ndarray
    halo_idxs: np.ndarray
    prior: AssemblyTimePrior = field(init=False)
    # prior: rv_continuous = field(init=False)

    def __attrs_post_init__(self) -> None: 
        self.prior = AssemblyTimePrior.from_data(self.times)
        # self.prior = fit_to_lognormal_distribution(self.times)

    @property
    def data(self) -> np.ndarray: 
        return np.column_stack((self.halo_idxs, self.times))

    @property 
    def min_time(self) -> float:
        return np.nanmin(self.times)

    @property 
    def max_time(self) -> float:
        return np.nanmax(self.times)

    def get_histogram_data(self, num_bins: int) -> dict[str, np.ndarray]:
        return get_formation_distribution(self.times, num_bins)

    def display(self, num_bins: int = 30, ax: Axes | None = None) -> Axes | None: 

        if ((orig_ax := ax) is None): 
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        a_range = np.linspace(
            np.log10(self.min_time), 
            np.log10(self.max_time), 
            200
        )

        # PDF 
        ax[0].hist(
            np.log10(self.times), bins=num_bins, 
            density=True, alpha=0.3, color='blue',
        )
        ax[0].plot(a_range, self.prior.model.pdf(a_range), color='k')

        # CDF
        ax[1].hist(
            np.log10(self.times), bins=num_bins, 
            density=True, alpha=0.3, color='blue', cumulative=True
        )
        ax[1].plot(a_range, self.prior.model.cdf(a_range), color='k')

        if orig_ax is not None:
            return ax 
        else: 
            plt.show()
    
@define
class JointFormationDistribution:
    times: np.ndarray
    present_halos_idxs: np.ndarray
    final_halos_idxs: np.ndarray

    @classmethod
    def from_formation_data(
            cls, present: FormationTimeData, final: FormationTimeData
        ) -> JointFormationDistribution:

        common = get_common_formation_assemblies(present.data, final.data)

        return cls(
            times = common["times"],
            present_halos_idxs = common["present_form_idxs"],
            final_halos_idxs = common["final_form_idxs"]
        )

    # Add a function to get a 2D histogram .. 


@define
class AssemblyTimes:
    present: FormationTimeData
    final: FormationTimeData
    joint: JointFormationDistribution

    @classmethod
    def from_histories(cls, histories: np.ndarray) -> AssemblyTimes:

        present_times = get_present_formation_times(histories)
        final_times = get_formation_times(histories)

        present = FormationTimeData(
            times = present_times[:, 1],
            halo_idxs = present_times[:, 0]
        )
        final = FormationTimeData(
            times = final_times[:, 1],
            halo_idxs = final_times[:, 0]
        )

        return cls(
            present = present,
            final = final,
            joint = JointFormationDistribution.from_formation_data(present, final)
        )

    def get_present_distribution(self, num_bins: int = 30) -> dict[str, np.ndarray]:
        return self.present.get_histogram_data(num_bins)

    def get_final_distribution(self, num_bins: int = 30) -> dict[str, np.ndarray]:
        return self.final.get_histogram_data(num_bins)

    def get_time_distributions(self, num_bins: int = 30) -> dict[str, dict[str, np.ndarray]]:
        return {
            "present" : self.get_present_distribution(num_bins), 
            "final" : self.get_final_distribution(num_bins), 
        }
    

@define 
class MAHData(ABC):
    mass_def: str
    masses: np.ndarray  
    histories: np.ndarray
    
    fraction_cutoff: float 

    rates: np.ndarray = field(init=False)
    accretion_rate: np.ndarray = field(init=False)
    freeze_out_time: float = field(init=False)
    mass_accumulation_time: float = field(init=False)

    # stats: MAHStats = field(init=False)
    stats: AssemblyStats = field(init=False)
    assembly_times: AssemblyTimes = field(init=False)

    def __attrs_post_init__(self) -> None:

        self.rates = np.zeros_like(self.histories)
        for i, mah in enumerate(self.histories):
            self.rates[i, :, 0] = mah[:, 0]
            self.rates[i, :, 1] = get_accretion_rate(mah[:, 0], mah[:, 1])

        # self.stats = MAHStats.from_histories(
        #     self.histories, fraction_cutoff=self.fraction_cutoff
        # )
        self.stats = AssemblyStats.from_data(
            histories=self.histories, 
            rates=self.rates,
            fraction_cutoff=self.fraction_cutoff
        )
        self.histories = self.histories[:, self.stats.above_fraction, :]
        self.rates = self.rates[:, self.stats.above_fraction, :]
        self.assembly_times = AssemblyTimes.from_histories(self.histories)

        try: 
            self.accretion_rate = get_accretion_rate(
                self.stats.mah.median[:, 0] , self.stats.mah.median[:, 1]
            )
        except (ValueError, TypeError) as error:
            print(error)
            self.accretion_rate = np.empty_like(self.stats.mah.median[:, 0])

        mah_nan_mask = ~np.isnan(self.stats.mah.median[:, 1])

        try:
            self.mass_accumulation_time = get_mass_accumulation_time(
                self.stats.mah.median[:, 0][mah_nan_mask],
                self.stats.mah.median[:, 1][mah_nan_mask],

            )
        except (ValueError, TypeError) as error:
            print(error)
            self.mass_accumulation_time = 0.0

        try: 
            accretion_nan_mask = ~np.isnan(self.accretion_rate) 
            
            self.freeze_out_time = get_freeze_out_time(
                self.accretion_rate[accretion_nan_mask], 
                self.stats.mah.median[:, 0][accretion_nan_mask]
            )
        except (ValueError, TypeError) as error:
            print(error)
            self.freeze_out_time = 0.0


    def __len__(self) -> int:
        return len(self.masses)
    
    @property
    def formation_time_normalized_histories(self) -> np.ndarray:
        valid_history_idxs = self.assembly_times.joint.final_halos_idxs
        valid_histories = self.histories[valid_history_idxs, :, :]
        for i, mah in enumerate(valid_histories):
            mah[:, 0] /= self.assembly_times.joint.times[i, 1]

        return valid_histories
    
    def get_median_mass_accumulation_time(self, pct: float) -> float:
        mah_nan_mask = ~np.isnan(self.stats.mah.median[:, 1])
        return get_mass_accumulation_time(
            self.stats.mah.median[:, 0][mah_nan_mask],
            self.stats.mah.median[:, 1][mah_nan_mask], 
            pct
        )
    
    def get_mass_accumulation_distribution(self, pct: float) -> np.ndarray:
        # Get the mass accumulation time for each halo
        times = []
        for mah in self.histories:
            mask = ~np.isnan(mah[:, 1])
            scale_factors, valid_history = mah[:, 0][mask], mah[:, 1][mask]
            if np.nanmin(valid_history) > pct: continue
            times.append(
                get_mass_accumulation_time(scale_factors, valid_history, pct)
            )

        return np.asarray(times)

    @property
    def peak_mass_fraction(self) -> np.ndarray: 
        return np.nanmax(self.histories[:, :, 1], axis=1)

    @property
    def peak_masses(self) -> np.ndarray: 
        return self.masses + np.log10(self.peak_mass_fraction)
        # return np.log10(10.0**self.masses * self.peak_mass_fraction)


    def display_mah(
            self, 
            num_histories: int = -1, 
            ax: Axes | None = None, 
            show_xlabels: bool = True,
            show_ylabels: bool = True,
            show_legend: bool = True, 
            text_size: int = 24,
            save_fig: bool = False
        ) -> Axes | None: 

        if (num_histories == -1) or (num_histories > len(self)): 
            num_histories = len(self)

        ax = plot_mass_accretion_history(
            histories=self.histories[:num_histories], 
            median=self.stats.mah.median,
            five_pct=self.stats.mah.pct.five,
            twenty_five_pct=self.stats.mah.pct.twenty_five,
            seventy_five_pct=self.stats.mah.pct.seventy_five,
            ninety_five_pct=self.stats.mah.pct.ninety_five,
            mass_def = self.mass_def[1:], 
            is_group = (self.mass_def != "subhalo"),
            ax = ax, 
            show_xlabel = show_xlabels,
            show_ylabel = show_ylabels,
            show_legend = show_legend, 
            text_size = text_size
        )

        return ax


    # def display_accretion(
    #         self, ax: Axes | None = None, show_freeze_out: bool = True
    #     ) -> Axes | None: 

    #     plot_accretion_rate(
    #         scale_factors = self.stats.mah.median[:, 0],
    #         accretion_rate = self.accretion_rate,
    #         # freeze_out_time = self.freeze_out_time, 
    #         freeze_out_time = self.mass_accumulation_time, 
    #         mass_def = self.mass_def, 
    #         show_freeze_out = show_freeze_out,
    #         ax = ax
    #     )

    # @abstractmethod
    # def display(self, *args, **kwargs) -> None:
    #     ...


@define 
class MAHDataset(ABC):
    scale_factors: np.ndarray
    M200c: MAHData
    M200m: MAHData
    Mfof: MAHData
    subhalo: MAHData

    
    @classmethod
    @abstractmethod
    def from_directory(cls, data_dir: Path, *args, **kwargs) -> MAHDataset:
        ... 

    @property
    def freeze_out_times(self) -> dict[str, float]:
        return {
            "M200c" : self.M200c.freeze_out_time,
            "M200m" : self.M200m.freeze_out_time,
            "Mfof" : self.Mfof.freeze_out_time,
            "subhalo" : self.subhalo.freeze_out_time,
        }
    
    @property
    def mass_accumulation_times(self) -> dict[str, float]:
        return {
            "M200c" : self.M200c.mass_accumulation_time,
            "M200m" : self.M200m.mass_accumulation_time,
            "Mfof" : self.Mfof.mass_accumulation_time,
            "subhalo" : self.subhalo.mass_accumulation_time,
        }

    def display_fraction_around(self, fraction_cutoff: float) -> None: 

        fig, ax = plt.subplots()

        

        ax.semilogx(
            self.scale_factors, 
            self.M200c.stats.fraction_around, 
            label=r"$M_{\rm 200c}$",
            zorder=0
        )

        ax.semilogx(
            self.scale_factors, 
            self.M200m.stats.fraction_around, 
            label=r"$M_{\rm 200m}$",
            zorder=0
        )

        ax.semilogx(
            self.scale_factors, 
            self.Mfof.stats.fraction_around, 
            label=r"$M_{\rm fof}$",
            zorder=0
        )

        ax.semilogx(
            self.scale_factors, 
            self.subhalo.stats.fraction_around, 
            label=r"$M_{\rm sub}$",
            zorder=0
        )

        ax.axhline(fraction_cutoff, color="grey", linestyle="--", zorder=1)

        ax.set_xlabel(r"${\rm Scale}$ ${\rm Factor}$, $a$")
        ax.set_ylabel(r"${\rm Fraction}$ ${\rm of} ${\rm Halos}$ ${\rm Around}$")
        ax.legend()
        plt.show()

    def display_mahs(self, num_histories: int = -1) -> None:
        _, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        if (num_histories == -1) or (num_histories > len(self.M200c)): 
            num_histories = len(self.M200c)
         
        axes[0] = self.M200c.display_mah(ax=axes[0], num_histories=num_histories)
        axes[1] = self.M200m.display_mah(ax=axes[1], num_histories=num_histories)
        axes[2] = self.Mfof.display_mah(ax=axes[2], num_histories=num_histories)
        axes[3] = self.subhalo.display_mah(ax=axes[3], num_histories=num_histories)

        plt.show()

    def display_accretions(self) -> None: 
        _, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        axes[0] = self.M200c.display_accretion(ax=axes[0])
        axes[1] = self.M200m.display_accretion(ax=axes[1])
        axes[2] = self.Mfof.display_accretion(ax=axes[2])
        axes[3] = self.subhalo.display_accretion(ax=axes[3])

        plt.show()

    
    def display_group(self) -> None:

        cmap = plt.get_cmap("tab10", 3)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))


        axes[0].loglog(
            self.M200c.stats.mah.median[:, 0],
            self.M200c.stats.mah.median[:, 1],
            '-', color=cmap(0), label=r"$M_{\rm 200c}$"
        )

        axes[0].loglog(
            self.M200m.stats.mah.median[:, 0],
            self.M200m.stats.mah.median[:, 1],
            '--', color=cmap(1), label=r"$M_{\rm 200m}$"
        )

        axes[0].loglog(
            self.Mfof.stats.mah.median[:, 0],
            self.Mfof.stats.mah.median[:, 1],
            '-.', color=cmap(2), label=r"$M_{\rm fof}$"
        )

        axes[0].set_xlabel(r"${\rm Scale}$ ${\rm Factor}$, $a$")
        # axes[0].set_ylabel(r"Mass Accretion History, $\frac{M(a)}{M(a=100)}$")
        axes[0].set_ylabel(r"$\Psi_{\Delta}(a)$")
        axes[0].legend()
        

        axes[1].semilogx(
            self.M200c.stats.mah.median[:, 0],
            self.M200c.accretion_rate,
            '-', color=cmap(0), 
            label=(
                r"$a_{{\rm freeze}}^{\rm 200c}$"
                rf" $\simeq$ {self.M200c.freeze_out_time:.2f}"
            )
        )

        axes[1].semilogx(
            self.M200m.stats.mah.median[:, 0],
            self.M200m.accretion_rate,
            '--', color=cmap(1), 
            label=(
                r"$a_{{\rm freeze}}^{\rm 200m}$"
                rf" $\simeq$ {self.M200m.freeze_out_time:.2f}"
            )
        )

        axes[1].semilogx(
            self.Mfof.stats.mah.median[:, 0],
            self.Mfof.accretion_rate,
            '-', color=cmap(2), 
            label=(
                r"$a_{{\rm freeze}}^{\rm fof}$"
                rf" $\simeq$ {self.Mfof.freeze_out_time:.2f}"
            )
        )

        axes[1].set_xlabel(r"${\rm Scale}$ ${\rm Factor}$, $a$")
        axes[1].set_ylabel(r"$\Gamma_{\Delta}(a)$")
        # axes[1].set_ylabel(r"Mass Accretion Rate, $\frac{d \ln M}{d \ln a}$")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def display_group_mahs(
            self, num_histories: int = -1,
            text_size: int = 24,
            save_fig: bool = False, 
        ) -> None:

        fig, axes = plt.subplots(
            1, 3, figsize=(18, 6), 
            sharex=True, sharey=True,
            gridspec_kw={'hspace': 0, 'wspace': 0}
        )
        axes = axes.flatten()

        if (num_histories == -1) or (num_histories > len(self.M200c)): 
            num_histories = len(self.M200c)
         
        axes[0] = self.M200c.display_mah(
            ax=axes[0], 
            num_histories=num_histories, 
            show_ylabels=False, 
            show_legend=True,
            text_size=text_size
        )
        axes[1] = self.M200m.display_mah(
            ax=axes[1], 
            num_histories=num_histories, 
            show_ylabels=False, 
            show_legend=False,
            text_size=text_size
        )
        axes[2] = self.Mfof.display_mah(
            ax=axes[2], 
            num_histories=num_histories, 
            show_ylabels=False, 
            show_legend=False,
            text_size=text_size
        )

        # Hide the y-labels
        axes[1].label_outer()
        axes[2].label_outer()

        # axes[0].set_ylabel(
        #     (
        #         # f"Mass Accretion Histories, "
        #         # r"$\log_{10}\left[\frac{M_{\Delta}(a)}{M_{\Delta}(a=100)}\right]$"
        #         r"$\log_{10}\Psi_{\Delta}(a)$"
        #     ),
        #     # r"$\log_{10}\left[\frac{M_{\Delta}(a)}{M_{\Delta}(a=100)}\right]$",
        #     fontsize=12
        # )
        axes[0].set_ylabel(r"$\log_{10}\Psi_{\Delta}(a)$", fontsize=text_size)
        axes[0].set_xlabel(r"${\rm Scale}$ ${\rm Factor}$, $a$", fontsize=text_size)
        axes[1].set_xlabel(r"${\rm Scale}$ ${\rm Factor}$, $a$", fontsize=text_size)
        axes[2].set_xlabel(r"${\rm Scale}$ ${\rm Factor}$, $a$", fontsize=text_size)


        # Set y-limits to be the same
        axes[0].set_ylim(-3.25, 0.5)
        axes[0].yaxis.set_tick_params(labelsize=text_size)
        

        axes[0].text(
            0.05, 0.95, r"$M_{\rm 200c}$", 
            transform=axes[0].transAxes, 
            fontsize=text_size, verticalalignment='top'
        )

        axes[1].text(
            0.05, 0.95, r"$M_{\rm 200m}$", 
            transform=axes[1].transAxes, 
            fontsize=text_size, verticalalignment='top'
        )

        axes[2].text(
            0.05, 0.95, r"$M_{\rm fof}$", 
            transform=axes[2].transAxes, 
            fontsize=text_size, verticalalignment='top'
        )

        if save_fig:
            plt.savefig("dataset_group_mahs.pdf", dpi=600)
        else:
            plt.show()
    


@define
class BinnedMAHData(MAHData):
    mass_bin: str | int
    formation_time: float | None = field(init=False)
    fraction_cutoff: float = field(default=0.0)

    def __attrs_post_init__(self) -> None:
        
        super().__attrs_post_init__()

        try: 
            nan_mask = ~np.isnan(self.stats.mah.median[:, 1]) 
            mah = self.stats.mah.median[:, 1][nan_mask]
            scale_factors = self.stats.mah.median[:, 0][nan_mask]
            self.formation_time = get_formation_time(scale_factors , mah)
        except (ValueError, TypeError) as error: 
            self.formation_time = None 

    def __repr__(self) -> str:  
        return (
            f"BinnedMAHData(mass_def={self.mass_def}, N={len(self)}, "
            f"a=({np.nanmin(self.stats.mah.median[:, 0]):.2f}, "
            f"{np.nanmax(self.stats.mah.median[:, 0]):.2f}), "
            f"mass_bin = {self.mass_bin_txt})"
        ) 

    @property
    def mass_bin_txt(self) -> str:
        return (
            self.mass_bin.capitalize() 
            if isinstance(self.mass_bin, str) else
            rf"$M$=$10^{{{self.mass_bin}}}$"
        )

    # Update this method to have histories and rates in 1 x 2 figure 
    def display(
            self, axes: Axes | None = None, 
            text_size: int = 24,
            display_ax_labels: bool = True, 
            save_fig: bool = False
        ) -> Axes | None:
        
        if (orig_ax := axes) is None: 
            fig, axes = plt.subplots(
                1, 2, figsize=(12, 6), sharex=True,
                # gridspec_kw={'hspace': , 'wspace': 0}
            )
            axes[0].set_xlim(0.1, 100)
            axes[1].set_xlim(0.1, 100)

        # if color is None: 
        #     color = "tab10:blue"

        axes[0] = plot_mass_accretion_history(
            histories=self.histories, 
            median=self.stats.mah.median,
            five_pct=self.stats.mah.pct.five,
            twenty_five_pct=self.stats.mah.pct.twenty_five,
            seventy_five_pct=self.stats.mah.pct.seventy_five,
            ninety_five_pct=self.stats.mah.pct.ninety_five,
            mass_def = self.mass_def[1:], 
            is_group = (self.mass_def != "subhalo"),
            ax = axes[0], 
            show_xlabel = display_ax_labels,
            show_ylabel = display_ax_labels,
            show_legend = True, 
            text_size = text_size
            # color=color
        )

        axes[1] = plot_mass_accretion_rates(
            rates=self.rates, 
            median=self.stats.rates.median,
            five_pct=self.stats.rates.pct.five,
            twenty_five_pct=self.stats.rates.pct.twenty_five,
            seventy_five_pct=self.stats.rates.pct.seventy_five,
            ninety_five_pct=self.stats.rates.pct.ninety_five,
            mass_def = self.mass_def[1:], 
            # is_group = (self.mass_def != "subhalo"),
            ax = axes[1], 
            show_xlabel = display_ax_labels,
            show_ylabel = display_ax_labels,
            show_legend = False, 
            text_size = text_size
            # color=color
        )

        if save_fig:
            plt.savefig(
                f"{self.mass_def}_{self.mass_bin}_history.pdf", 
                bbox_inches='tight',
                dpi=600
            )

        if orig_ax is not None:
            return axes
        else:
            # plt.tight_layout()
            plt.show()



@define
class AsymptoticMAHData(MAHData):

    mass_bins: np.ndarray = field(init=False)
    mass_bin_locs: np.ndarray = field(init=False)

    def __attrs_post_init__(self) -> None:

        super().__attrs_post_init__() 
        
        self.mass_bins = np.unique(self.masses.astype(int))
        self.mass_bin_locs = np.digitize(self.masses.astype(int), self.mass_bins)


    def __repr__(self) -> str: 
        mass_bins = list(self.mass_bins)
        return (
            f"AsymptoticMAHData(mass_def={self.mass_def}, N={len(self)}, "
            f"a=({np.nanmin(self.stats.mah.median[:, 0]):.2f}, "
            f"{np.nanmax(self.stats.mah.median[:, 0]):.2f}), "
            f"mass_bins = {mass_bins})"
        )

    @classmethod
    def from_hdf5(
            cls, filepath: Path, 
            fraction_cutoff: float = 0.5,
            for_group: bool=True, 
            min_mass: float | None = None
        ) -> AsymptoticMAHData:

        with h5py.File(filepath, "r") as data_file:
            mass_def = data_file.attrs["mass_def"] if for_group else "subhalo"
            data = data_file["group"] if for_group else data_file["subhalo"]

            masses = data["masses"][()]
            histories = data["histories"][()]

            # pdb.set_trace()

            above_cutoff = np.asarray(masses >= min_mass)

            return cls(
                mass_def = mass_def,
                fraction_cutoff = fraction_cutoff, 
                masses = masses[above_cutoff],
                histories = histories[above_cutoff]
            )

    @classmethod
    def from_mah_datasets(
            cls, mass_def: str,
            dataset: dict[int, dict[int, MAHDataset]], 
            fraction_cutoff: float = 0.5,
            above_fraction: np.ndarray | None = None
            # Add the abiltiy to use the history indicies that correspond to the 
            # same M200c values that made the 200c mass cut
        ) -> AsymptoticMAHData: 

        masses, formation_times, histories = [], [], []

        for box in dataset.values(): 
            for data in box.values():
                if mass_def not in data.__annotations__:
                    raise ValueError(f"{mass_def} not in {data}")

                masses.append(getattr(data, mass_def).masses)
                histories.append(getattr(data, mass_def).histories)

        # Histories Need to be concatenated at the correct scale factor
        unique_scale_factors = np.unique(
            np.concatenate(
                [np.nanmax(history[:, :, 0], axis=0) for history in histories], 
                axis=0
            )
        )

        concat_histories = []
        for history in histories: 
            current_scale_factors = np.nanmax(history[:, :, 0], axis=0)

            missing_scale_factors = np.isin(unique_scale_factors, current_scale_factors)
            missing_idxs = np.asarray(np.where(~missing_scale_factors)[0])

            if len(missing_idxs) == 0:
                concat_histories.append(history)
            else:

                current_idxs = np.asarray(np.where(missing_scale_factors)[0])

                new_history = np.empty((history.shape[0], len(unique_scale_factors), 2))
                new_history[:, :, 0] = unique_scale_factors
                new_history[:, :, 1] = np.nan
                new_history[:, current_idxs, :] = history
                concat_histories.append(new_history)

        histories = np.concatenate(concat_histories, axis=0)
        
        if above_fraction is not None: 
            histories = histories[:, above_fraction, :]
            fraction_cutoff = 0.0

        return cls(
            mass_def = mass_def,
            fraction_cutoff = fraction_cutoff, 
            masses = np.concatenate(masses, axis=0),
            histories = histories
        )


    def get_binned_data(self, mass_bin: str | int) -> BinnedMAHData: 
        if mass_bin == "all": 
            return BinnedMAHData(
                mass_def = self.mass_def,
                mass_bin = "all",
                masses = self.masses,
                histories = self.histories,
            )

        mass_bin_idx = np.where(mass_bin == self.mass_bins)[0] + 1
        
        if len(mass_bin_idx) == 0: 
            raise ValueError("This is not a valid mass bin for this data")

        return BinnedMAHData(
            mass_def = self.mass_def,
            mass_bin = mass_bin, 
            masses = self.masses[self.mass_bin_locs == mass_bin_idx], 
            histories = self.histories[self.mass_bin_locs == mass_bin_idx],
        )

    @property
    def binned_dataset(self) -> dict[int, BinnedMAHData]:
        return {
            mass_bin: self.get_binned_data(mass_bin) 
            for mass_bin in self.mass_bins
        }

    def display_binned(self, ax: Axes | None = None) -> None:

        binned = self.binned_dataset

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        bin_cmap = plt.get_cmap("tab20", len(binned))
        for i, (mass_bin, data) in enumerate(binned.items()):

            axes = data.display(
                mass_def = self.mass_def, 
                axes = axes,
                color = bin_cmap(i),
                display_ax_labels = False
            )

        axes[0].set_xlabel(r"Scale Factor, $a$")
        axes[0].set_ylabel(
            f"Mass Accretion History, "
            rf" $\frac{{M_{{\rm {self.mass_def}}}(a)}}{{M_{{\rm {self.mass_def}}}(a=100)}}$"
        )
        axes[0].legend()

        axes[1].set_xlabel(r"Scale Factor, $a$")
        axes[1].set_ylabel(
            rf"Mass Accretion Rate,"
            rf" $\frac{{d \ln M_{{\rm {self.mass_def}}}}}{{d \ln a}}$"
        )
        axes[1].legend()

        plt.tight_layout()
        plt.show()


    def display(self, num_histories: int = -1) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0] = self.display_mah(ax=axes[0], num_histories=num_histories)
        axes[1] = self.display_accretion(ax=axes[1])

        plt.show()

@define 
class BinnedMAHDataset(MAHDataset):

    def __attrs_post_init__(self) -> None: 

        assert(self.M200c.mass_def ==  "M200c")
        assert(self.M200m.mass_def ==  "M200m")
        assert(self.Mfof.mass_def ==  "Mfof")
        assert(self.subhalo.mass_def ==  "subhalo")

    @classmethod
    def from_directory(cls, data_dir: Path, mass_bin: int) -> BinnedMAHDataset:
        raise NotImplementedError
    
@define 
class BoxMAHDataset(MAHDataset):
    M200c: AsymptoticMAHData
    M200m: AsymptoticMAHData
    Mfof: AsymptoticMAHData
    subhalo: AsymptoticMAHData

    box_size: int
    num_particles: int 
    lower_mass_limit: float
    upper_mass_limit: float 


    def __repr__(self) -> str:
        mass_range = (self.lower_mass_limit, self.upper_mass_limit)
        mass_bins = list(self.M200c.mass_bins)
        return (
            f"BoxMAHDataset(L={self.box_size}, N={self.num_particles}^3, "
            f"mass_range=({mass_range[0]:.2f}, {mass_range[1]:.2f}), "
            f"{mass_bins= })"
        )

    @classmethod
    def from_directory(
            cls, data_dir: Path, 
            min_mass: float | None = None,
            max_mass: float | None = None,
            fraction_cutoff: float = 0.0
        ) -> BoxMAHDataset:
        
        box_size = by_size(data_dir.name)
        num_particles = by_resolution(data_dir.name)

        path200c =Path(data_dir, "mah_M200c.h5")

        if (min_mass is None) or (max_mass is None):
            with h5py.File(path200c, "r") as data_file:
                min_mass = min_mass or data_file.attrs["lower_mass_limit"]
                max_mass = max_mass or data_file.attrs["upper_mass_limit"]

        M200c = AsymptoticMAHData.from_hdf5(
            path200c, 
            for_group=True, 
            min_mass=min_mass, 
            fraction_cutoff=fraction_cutoff
        )
        M200m = AsymptoticMAHData.from_hdf5(
            Path(data_dir, "mah_M200m.h5"), 
            for_group=True, 
            min_mass=min_mass,
            fraction_cutoff=fraction_cutoff
        )
        Mfof = AsymptoticMAHData.from_hdf5(
            Path(data_dir, "mah_Mfof.h5"), 
            for_group=True, 
            min_mass=min_mass,
            fraction_cutoff=fraction_cutoff
        )
        subhalo = AsymptoticMAHData.from_hdf5(
            path200c, 
            for_group=False, 
            min_mass=min_mass,
            fraction_cutoff=fraction_cutoff
        )
        
        scale_factors = np.nanmax(M200c.histories[:, :, 0], axis=0)

        return cls(
            box_size = box_size,
            num_particles = num_particles,
            lower_mass_limit = min_mass,
            upper_mass_limit = max_mass, 
            scale_factors = scale_factors,
            M200c = M200c,
            M200m = M200m,
            Mfof = Mfof,
            subhalo = subhalo
        )

    @property
    def binned_dataset(self) -> dict[int, BinnedMAHDataset]:
        return {
            mass_bin: BinnedMAHDataset(
                scale_factors = self.scale_factors,
                M200c = self.M200c.get_binned_data(mass_bin),
                M200m = self.M200m.get_binned_data(mass_bin),
                Mfof = self.Mfof.get_binned_data(mass_bin),
                subhalo = self.subhalo.get_binned_data(mass_bin),
            )
            for mass_bin in self.M200c.mass_bins
        }

    def display_binned(self) -> None:
        
        binned = self.binned_dataset

        for data in binned.values(): 
            data.display_group(display_ax_labels=True)

    def display_mass_def_binned(self, mass_def: str) -> None:

        binned = self.binned_dataset 
        cmap = plt.get_cmap("tab20", len(binned))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        for i, mass_bin in enumerate(sorted(binned)): 

            data = getattr(binned[mass_bin], mass_def)

            axes[0].loglog(
                data.stats.mah.median[:, 0], 
                data.stats.mah.median[:, 1], 
                color=cmap(i),
                label=rf"$M_{{\rm {mass_def[1:]}}}=10^{{{mass_bin}}}$"
            )

            axes[1].semilogx(
                data.stats.mah.median[:, 0], 
                data.accretion_rate, 
                color=cmap(i),
                label=(
                    rf"$a_{{\rm freeze}}\left(M_{{\rm {mass_def[1:]}}} \right)$"
                    rf" $\simeq$ {data.mass_accumulation_time:.2f}"
                    # rf" $\simeq$ {data.freeze_out_time:.2f}"
                )
            )

        axes[0].set_xlabel(r"Scale Factor, $a$")
        axes[0].set_ylabel(rf"Mass Accretion History, $\frac{{M(a)}}{{M(a=100)}}$")
        # axes[0].set_ylabel(
        #     rf"Mass Accretion History, $\frac{{M_{{\rm {mass_def[1:]}}(a)}}{M_{{\rm {mass_def[1:]}}(a=100)}}}}$"
        # )

        axes[1].set_xlabel(r"Scale Factor, $a$")
        axes[1].set_ylabel(
            rf"Mass Accretion Rate, $\frac{{d \ln M}}{{d \ln a}}$"
        )
        # axes[1].set_ylabel(
        #     rf"Mass Accretion Rate, $\frac{{d \ln M_{{\rm {mass_def[1:]}}}}{{d \ln a}}}$"
        # )
        
        axes[0].legend()
        axes[1].legend()

        plt.tight_layout()
        plt.show()

@define 
class JointMAHDataset(MAHDataset):
    M200c: AsymptoticMAHData
    M200m: AsymptoticMAHData
    Mfof: AsymptoticMAHData
    subhalo: AsymptoticMAHData 

    data: dict[int, dict[int, BoxMAHDataset]]

    def __repr__(self) -> str:
        return repr(self.data)
    
    @classmethod
    def from_directory(
            cls, data_dir: Path, 
            fraction_cutoff: float = 0.5,
            min_particles: int = 100, 
            seed_num: int = 1,
            boxes_to_skip: list[tuple[int, int, int]] = []
        ) -> JointMAHDataset: 

        # May have to update this to get scale factor cutoff, then past it
        # into the box datasets. Add keyword to trigger fractional cutoff
        # in MAHData, MAHDataset, and MAHStats. This should resolve that weird
        # jagged issue. 
        
        data = defaultdict(dict)

        for mah_dir in data_dir.glob("L*_N*_box*"):

            seed = by_seed(mah_dir.name)

            if seed != seed_num: continue 
            
            box_size = by_size(mah_dir.name)
            num_particles = by_resolution(mah_dir.name)

            if (box_size, num_particles, seed_num) in boxes_to_skip: continue
            
            if seed_num == 1:
                particle_mass = box_particle_masses[num_particles][box_size]
            else:
                particle_mass = wmap_cosmo_particle_mass
            min_mass = np.log10(min_particles * particle_mass)

            if not any(mah_dir.iterdir()): continue
            # print(f"Storing L={box_size}, N={num_particles}^3")

            try:
                data[num_particles][box_size] = BoxMAHDataset.from_directory(
                    mah_dir, min_mass=min_mass
                )
            except (OSError, ValueError) as error:
                print(
                    f"L={box_size}, N={num_particles}^3 box will not be " 
                    f"included due to  {error}"
                )
                continue

        dataM200c = AsymptoticMAHData.from_mah_datasets(
            mass_def="M200c", 
            dataset=data,
            fraction_cutoff=fraction_cutoff
        )
        scale_factors = np.nanmax(dataM200c.histories[:, :, 0], axis=0)

        return cls(
            data=data,
            scale_factors=scale_factors,
            M200c=dataM200c,
            M200m=AsymptoticMAHData.from_mah_datasets(
                mass_def="M200m", 
                dataset=data,
                # fraction_cutoff=fraction_cutoff
                above_fraction=dataM200c.stats.above_fraction
            ),
            Mfof=AsymptoticMAHData.from_mah_datasets(
                mass_def="Mfof", 
                dataset=data,
                # fraction_cutoff=fraction_cutoff
                above_fraction=dataM200c.stats.above_fraction
            ),
            subhalo=AsymptoticMAHData.from_mah_datasets(
                mass_def="subhalo", 
                dataset=data,
                # fraction_cutoff=fraction_cutoff
                above_fraction=dataM200c.stats.above_fraction
            )
        )

    # @classmethod
    # def from_directory(
    #         cls, data_dir: Path, min_particles: int = 100, seed_num: int = 1
    #     ) -> JointMAHDataset: 

    #     # May have to update this to get scale factor cutoff, then past it
    #     # into the box datasets. Add keyword to trigger fractional cutoff
    #     # in MAHData, MAHDataset, and MAHStats. This should resolve that weird
    #     # jagged issue. 
        
    #     data = defaultdict(dict)

    #     for mah_dir in data_dir.glob("L*_N*_box*"):

    #         seed = by_seed(mah_dir.name)

    #         if seed != seed_num: continue 
            
    #         box_size = by_size(mah_dir.name)
    #         num_particles = by_resolution(mah_dir.name)
            
    #         if seed_num == 1:
    #             particle_mass = box_particle_masses[num_particles][box_size]
    #         else:
    #             particle_mass = wmap_cosmo_particle_mass
    #         min_mass = np.log10(min_particles * particle_mass)

    #         if not any(mah_dir.iterdir()): continue
    #         # print(f"Storing L={box_size}, N={num_particles}^3")

    #         try:
    #             data[num_particles][box_size] = BoxMAHDataset.from_directory(mah_dir, min_mass=min_mass)
    #         except (OSError, ValueError) as error:
    #             print(
    #                 f"L={box_size}, N={num_particles}^3 box will not be " 
    #                 f"included due to  {error}"
    #             )
    #             continue

    #     dataM200c = AsymptoticMAHData.from_mah_datasets(data, "M200c")
    #     scale_factors = np.nanmax(dataM200c.histories[:, :, 0], axis=0)

    #     return cls(
    #         data=data,
    #         scale_factors=scale_factors,
    #         M200c=dataM200c,
    #         M200m=AsymptoticMAHData.from_mah_datasets(data, "M200m"),
    #         Mfof=AsymptoticMAHData.from_mah_datasets(data, "Mfof"),
    #         subhalo=AsymptoticMAHData.from_mah_datasets(data, "subhalo")
    #     )

    @property
    def binned_dataset(self) -> dict[int, BinnedMAHDataset]:
        return {
            mass_bin: BinnedMAHDataset(
                scale_factors=self.scale_factors,
                M200c = self.M200c.get_binned_data(mass_bin),
                M200m = self.M200m.get_binned_data(mass_bin),
                Mfof = self.Mfof.get_binned_data(mass_bin),
                subhalo = self.subhalo.get_binned_data(mass_bin),
            )
            for mass_bin in self.M200c.mass_bins
        }

    def display_binned(self) -> None:
        
        binned = self.binned_dataset

        for data in binned.values(): 
            data.display_group()

    def display_mass_def_binned(self, mass_def: str) -> None:

        binned = self.binned_dataset 
        cmap = plt.get_cmap("tab20", len(binned))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        for i, mass_bin in enumerate(sorted(binned)): 

            data = getattr(binned[mass_bin], mass_def)

            axes[0].loglog(
                data.stats.mah.median[:, 0], 
                data.stats.mah.median[:, 1], 
                color=cmap(i),
                label=rf"$M_{{\rm {mass_def[1:]}}}=10^{{{mass_bin}}}$"
            )

            axes[1].semilogx(
                data.stats.median[:, 0], 
                data.accretion_rate, 
                color=cmap(i),
                label=(
                    rf"$a_{{\rm freeze}}\left(M_{{\rm {mass_def[1:]}}} \right)$"
                    rf" $\simeq$ {data.freeze_out_time:.2f}"
                )
            )

        axes[0].set_xlabel(r"Scale Factor, $a$")
        axes[0].set_ylabel(rf"Mass Accretion History, $\frac{{M(a)}}{{M(a=100)}}$")
        # axes[0].set_ylabel(
        #     rf"Mass Accretion History, $\frac{{M_{{\rm {mass_def[1:]}}(a)}}{M_{{\rm {mass_def[1:]}}(a=100)}}}}$"
        # )

        axes[1].set_xlabel(r"Scale Factor, $a$")
        axes[1].set_ylabel(
            rf"Mass Accretion Rate, $\frac{{d \ln M}}{{d \ln a}}$"
        )
        # axes[1].set_ylabel(
        #     rf"Mass Accretion Rate, $\frac{{d \ln M_{{\rm {mass_def[1:]}}}}{{d \ln a}}}$"
        # )
        
        axes[0].legend()
        axes[1].legend()

        plt.tight_layout()
        plt.show()