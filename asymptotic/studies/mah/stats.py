from __future__ import annotations

import numpy as np

from attrs import define, field
from collections import OrderedDict

from ...simulation.evo import EvolutionData


by_seed = lambda sim_name: int(sim_name.split('_')[-1].split('box')[-1])
by_size = lambda sim_name: int(sim_name.split('_')[0].split('L')[-1])
by_resolution = lambda sim_name: int(sim_name.split('_')[1].split('N')[-1])


@define
class Percentiles:
    five: np.ndarray
    twenty_five: np.ndarray
    fifty: np.ndarray
    seventy_five: np.ndarray
    ninety_five: np.ndarray

@define
class HistoryStats:
    mean: np.ndarray
    std: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    pct: Percentiles

    @classmethod
    def from_data(cls, data: np.ndarray,  above: np.ndarray) -> HistoryStats:
        return cls(
            mean=np.nanmean(data[:, above, :], axis=0),
            std=np.nanstd(data[:, above, :], axis=0),
            minimum=np.nanmin(data[:, above, :], axis=0),
            maximum=np.nanmax(data[:, above, :], axis=0),
            pct=Percentiles(
                five=np.nanpercentile(data[:, above, :], 5, axis=0),
                twenty_five=np.nanpercentile(data[:, above, :], 25, axis=0),
                fifty=np.nanpercentile(data[:, above, :], 50, axis=0),
                seventy_five=np.nanpercentile(data[:, above, :], 75, axis=0),
                ninety_five=np.nanpercentile(data[:, above, :], 95, axis=0)
            )
        )

    @property
    def median(self) -> np.ndarray:
        return self.pct.fifty


@define
class AssemblyStats:
    """MAH + accretion-rate stats over the snapshots that clear ``fraction_cutoff``.

    ``above_fraction`` is the boolean snapshot mask kept (those present in at least
    ``fraction_cutoff`` of halos); ``mah`` and ``rates`` are the per-snapshot
    :class:`HistoryStats` on the (n_halos, n_snaps, 2) history and rate arrays.
    """

    above_fraction: np.ndarray
    mah: HistoryStats
    rates: HistoryStats

    @classmethod
    def from_data(
        cls, histories: np.ndarray, rates: np.ndarray, fraction_cutoff: float
    ) -> AssemblyStats:
        num_halos = np.sum(~np.isnan(histories[:, :, 1]), axis=0)
        above_fraction = np.asarray((num_halos / histories.shape[0]) >= fraction_cutoff)
        return cls(
            above_fraction=above_fraction,
            mah=HistoryStats.from_data(histories, above_fraction),
            rates=HistoryStats.from_data(rates, above_fraction),
        )