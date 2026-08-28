from __future__ import annotations

import h5py
import numpy as np

from pathlib import Path
from typing import Any, Mapping
from collections import OrderedDict

from ..simulation.moments import Moment, MomentsInTime
from .power import (
    PowerSpectrum,
    MatterPowerSpectrum,
    MatterPowerSpectrumData,
    MatterPowerSpectrumEvoData,
)

'''
HDF5 I/O for the matter power spectrum containers defined in `power.py`.

Everything written here is plain arrays + scalar attributes (no pickles), so
the files are readable by any HDF5 tool while still round-tripping exactly
back into the nested attrs objects.

Layout of a single `MatterPowerSpectrumEvoData` file
----------------------------------------------------
/
  attrs:
    class          = "MatterPowerSpectrumEvoData"
    schema_version = 1
    in_comoving    = 0/1
    colormap_name  = "viridis"
  /moments/                        <- MomentsInTime
      snapshot_ids, scale_factors, redshifts,
      ages, lookback_times, hubble_epochs, dynamical_times
      /initial/                    <- MomentsInTime.initial (a single Moment)
          attrs: snapshot_id, scale_factor, redshift, age,
                 lookback_time, hubble_epoch, dynamical_time
  /snapshots/
      /<snapshot_id>/              <- MatterPowerSpectrumData
          attrs: in_comoving, is_null
          /estimate/               <- MatterPowerSpectrum
              attrs: in_comoving, has_shot_noise
              /linear/             <- PowerSpectrum
                  wavenumbers, amplitudes, normalized
                  attrs: is_linear, in_comoving, growth_factor, is_null
              /nonlinear/          <- PowerSpectrum
              /shot_noise/         <- PowerSpectrum (only if not None)
          /errors/                 <- MatterPowerSpectrum (same shape as above)

A dict of instances keyed by resolution (e.g. `{512: ..., 1024: ...}`) is
written with the same per-instance layout nested one level down, under a
group named after the key.

The derived `spline` (rebuilt in `PowerSpectrum.__attrs_post_init__`) and
`colormap` (rebuilt from `colormap_name`) fields are intentionally not
stored; they are regenerated on load.
'''

SCHEMA_VERSION = 1

EVO_DATA_CLASS_TAG = "MatterPowerSpectrumEvoData"
EVO_DATA_DICT_CLASS_TAG = "MatterPowerSpectrumEvoDataDict"

# Scalar Moment attributes stored alongside `snapshot_id`.
_MOMENT_SCALAR_FIELDS = (
    "scale_factor",
    "redshift",
    "age",
    "lookback_time",
    "hubble_epoch",
    "dynamical_time",
)

# Only compress arrays big enough for chunking to be worth it.
_COMPRESSION_THRESHOLD = 64

_HDF5_SUFFIXES = (".h5", ".hdf5")


# ---------------------------------------------------------------------------
# Low level helpers
# ---------------------------------------------------------------------------
def resolve_hdf5_path(
        file_path: Path | str,
        directory: Path | str | None = None,
        default_suffix: str = ".h5",
    ) -> Path:
    """
    Turn a bare file name (plus optional directory) into a full HDF5 path.

    Parameters
    ----------
    file_path : Path | str
        File name or full path. A missing `.h5`/`.hdf5` suffix is appended.
    directory : Path | str | None
        Directory to place the file in. When given, only the *name* of
        `file_path` is used, so both `save("ps.h5", directory=d)` and
        `save(d / "ps.h5")` land in the same place.
    default_suffix : str
        Suffix appended when `file_path` has neither `.h5` nor `.hdf5`.
    """
    file_path = Path(file_path)

    if directory is not None:
        file_path = Path(directory) / file_path.name

    if file_path.suffix.lower() not in _HDF5_SUFFIXES:
        file_path = file_path.parent / (file_path.name + default_suffix)

    return file_path


def _write_array(h5py_group: h5py.Group, name: str, array: np.ndarray) -> None:
    """Write a float array, replacing any dataset already sitting there."""
    data = np.asarray(array, dtype=float)

    if name in h5py_group:
        del h5py_group[name]

    if data.size >= _COMPRESSION_THRESHOLD:
        h5py_group.create_dataset(
            name, data=data, compression="gzip", shuffle=True
        )
    else:
        h5py_group.create_dataset(name, data=data)


def _read_array(h5py_group: h5py.Group, name: str) -> np.ndarray:
    if name not in h5py_group:
        return np.empty(0, dtype=float)
    return np.asarray(h5py_group[name][()], dtype=float)


def _as_str(value: Any) -> str:
    """h5py returns `bytes` for string attributes on some versions."""
    return value.decode() if isinstance(value, bytes) else str(value)


def _read_bool_attr(
        h5py_group: h5py.Group, name: str, default: bool = True
    ) -> bool:
    if name not in h5py_group.attrs:
        return default
    return bool(h5py_group.attrs[name])


def _read_float_attr(
        h5py_group: h5py.Group, name: str, default: float = np.nan
    ) -> float:
    if name not in h5py_group.attrs:
        return default
    return float(h5py_group.attrs[name])


def _replace_group(h5py_group: h5py.Group, name: str) -> h5py.Group:
    """Create `name` under `h5py_group`, clearing any previous contents."""
    if name in h5py_group:
        del h5py_group[name]
    return h5py_group.create_group(name)


# ---------------------------------------------------------------------------
# Moment / MomentsInTime
# ---------------------------------------------------------------------------
def write_moment(h5py_group: h5py.Group, moment: Moment) -> None:
    """Write a single `Moment` as scalar attributes."""
    h5py_group.attrs["class"] = "Moment"
    h5py_group.attrs["snapshot_id"] = int(moment.snapshot_id)

    for name in _MOMENT_SCALAR_FIELDS:
        h5py_group.attrs[name] = float(getattr(moment, name))


def read_moment(h5py_group: h5py.Group) -> Moment:
    return Moment(
        snapshot_id=int(h5py_group.attrs.get("snapshot_id", -1)),
        **{
            name: _read_float_attr(h5py_group, name)
            for name in _MOMENT_SCALAR_FIELDS
        },
    )


def write_moments(h5py_group: h5py.Group, moments: MomentsInTime) -> None:
    """
    Write a `MomentsInTime`, including the `initial` moment.

    `MomentsInTime.to_hdf5` covers the time-axis arrays; the `initial`
    Moment is not part of that payload, so it gets its own subgroup here.
    """
    h5py_group.attrs["class"] = moments.__class__.__name__
    h5py_group.attrs["schema_version"] = SCHEMA_VERSION

    moments.to_hdf5(h5py_group)
    write_moment(_replace_group(h5py_group, "initial"), moments.initial)


def read_moments(h5py_group: h5py.Group) -> MomentsInTime:
    moments = MomentsInTime.from_hdf5(h5py_group)

    if "initial" in h5py_group:
        moments.initial = read_moment(h5py_group["initial"])

    return moments


# ---------------------------------------------------------------------------
# PowerSpectrum
# ---------------------------------------------------------------------------
def write_power_spectrum(
        h5py_group: h5py.Group, spectrum: PowerSpectrum
    ) -> None:
    """Write the three arrays plus the scalar state of a `PowerSpectrum`."""
    h5py_group.attrs["class"] = spectrum.__class__.__name__
    h5py_group.attrs["is_linear"] = int(bool(spectrum.is_linear))
    h5py_group.attrs["in_comoving"] = int(bool(spectrum.in_comoving))
    h5py_group.attrs["growth_factor"] = float(spectrum.growth_factor)
    # Informational only; `is_null` is re-derived from the arrays on load.
    h5py_group.attrs["is_null"] = int(bool(spectrum.is_null))

    _write_array(h5py_group, "wavenumbers", spectrum.wavenumbers)
    _write_array(h5py_group, "amplitudes", spectrum.amplitudes)
    _write_array(h5py_group, "normalized", spectrum.normalized)


def read_power_spectrum(h5py_group: h5py.Group) -> PowerSpectrum:
    return PowerSpectrum(
        wavenumbers=_read_array(h5py_group, "wavenumbers"),
        amplitudes=_read_array(h5py_group, "amplitudes"),
        normalized=_read_array(h5py_group, "normalized"),
        is_linear=_read_bool_attr(h5py_group, "is_linear", default=False),
        in_comoving=_read_bool_attr(h5py_group, "in_comoving", default=True),
        growth_factor=_read_float_attr(h5py_group, "growth_factor", default=1.0),
    )


# ---------------------------------------------------------------------------
# MatterPowerSpectrum
# ---------------------------------------------------------------------------
def write_matter_power_spectrum(
        h5py_group: h5py.Group, spectrum: MatterPowerSpectrum
    ) -> None:
    """Write the linear / nonlinear / (optional) shot-noise triple."""
    h5py_group.attrs["class"] = spectrum.__class__.__name__
    h5py_group.attrs["in_comoving"] = int(bool(spectrum.in_comoving))
    h5py_group.attrs["has_shot_noise"] = int(spectrum.shot_noise is not None)

    write_power_spectrum(_replace_group(h5py_group, "linear"), spectrum.linear)
    write_power_spectrum(
        _replace_group(h5py_group, "nonlinear"), spectrum.nonlinear
    )

    # `shot_noise` is genuinely optional (None for ratio-built spectra), so
    # its absence is recorded by omitting the group entirely.
    if spectrum.shot_noise is not None:
        write_power_spectrum(
            _replace_group(h5py_group, "shot_noise"), spectrum.shot_noise
        )


def read_matter_power_spectrum(h5py_group: h5py.Group) -> MatterPowerSpectrum:
    return MatterPowerSpectrum(
        linear=read_power_spectrum(h5py_group["linear"]),
        nonlinear=read_power_spectrum(h5py_group["nonlinear"]),
        shot_noise=(
            read_power_spectrum(h5py_group["shot_noise"])
            if "shot_noise" in h5py_group else
            None
        ),
        in_comoving=_read_bool_attr(h5py_group, "in_comoving", default=True),
    )


# ---------------------------------------------------------------------------
# MatterPowerSpectrumData
# ---------------------------------------------------------------------------
def write_matter_power_spectrum_data(
        h5py_group: h5py.Group, ps_data: MatterPowerSpectrumData
    ) -> None:
    """Write the estimate / errors pair for a single snapshot."""
    h5py_group.attrs["class"] = ps_data.__class__.__name__
    h5py_group.attrs["in_comoving"] = int(bool(ps_data.in_comoving))
    h5py_group.attrs["is_null"] = int(bool(ps_data.is_null))

    write_matter_power_spectrum(
        _replace_group(h5py_group, "estimate"), ps_data.estimate
    )
    write_matter_power_spectrum(
        _replace_group(h5py_group, "errors"), ps_data.errors
    )


def read_matter_power_spectrum_data(
        h5py_group: h5py.Group
    ) -> MatterPowerSpectrumData:
    return MatterPowerSpectrumData(
        estimate=read_matter_power_spectrum(h5py_group["estimate"]),
        errors=read_matter_power_spectrum(h5py_group["errors"]),
        in_comoving=_read_bool_attr(h5py_group, "in_comoving", default=True),
    )


# ---------------------------------------------------------------------------
# MatterPowerSpectrumEvoData
# ---------------------------------------------------------------------------
def write_matter_power_spectrum_evo_data(
        h5py_group: h5py.Group, evo_data: MatterPowerSpectrumEvoData
    ) -> None:
    """
    Write a full `MatterPowerSpectrumEvoData` into an open group.

    `h5py_group` may be the root of a file (single instance) or a subgroup
    (one entry of a resolution-keyed dict).
    """
    h5py_group.attrs["class"] = EVO_DATA_CLASS_TAG
    h5py_group.attrs["schema_version"] = SCHEMA_VERSION
    h5py_group.attrs["in_comoving"] = int(bool(evo_data.in_comoving))
    h5py_group.attrs["colormap_name"] = str(evo_data.colormap_name)
    h5py_group.attrs["n_snapshots"] = int(len(evo_data.data))

    write_moments(_replace_group(h5py_group, "moments"), evo_data.moments)

    snapshots_group = _replace_group(h5py_group, "snapshots")
    for snapshot_id, ps_data in evo_data.data.items():
        write_matter_power_spectrum_data(
            snapshots_group.create_group(str(int(snapshot_id))), ps_data
        )


def read_matter_power_spectrum_evo_data(
        h5py_group: h5py.Group
    ) -> MatterPowerSpectrumEvoData:

    if "snapshots" not in h5py_group or "moments" not in h5py_group:
        raise KeyError(
            "Group is missing the 'moments'/'snapshots' subgroups written by "
            "save_matter_power_spectrum_evo_data(...)"
        )

    snapshots_group = h5py_group["snapshots"]

    data = OrderedDict(
        (int(name), read_matter_power_spectrum_data(snapshots_group[name]))
        for name in sorted(snapshots_group.keys(), key=int)
    )

    return MatterPowerSpectrumEvoData(
        moments=read_moments(h5py_group["moments"]),
        data=data,
        in_comoving=_read_bool_attr(h5py_group, "in_comoving", default=True),
        colormap_name=_as_str(h5py_group.attrs.get("colormap_name", "viridis")),
    )


def save_matter_power_spectrum_evo_data(
        evo_data: MatterPowerSpectrumEvoData,
        file_path: Path | str,
        directory: Path | str | None = None,
    ) -> Path:
    """
    Save a `MatterPowerSpectrumEvoData` instance to an HDF5 file.

    Parameters
    ----------
    evo_data : MatterPowerSpectrumEvoData
        The instance to write.
    file_path : Path | str
        Output file name or full path (`.h5` appended when missing).
    directory : Path | str | None
        Optional directory; when given, only the name of `file_path` is used.

    Returns
    -------
    Path
        The path actually written.
    """
    file_path = resolve_hdf5_path(file_path, directory)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, "w") as h5_file:
        write_matter_power_spectrum_evo_data(h5_file, evo_data)

    return file_path


def load_matter_power_spectrum_evo_data(
        file_path: Path | str,
        directory: Path | str | None = None,
    ) -> MatterPowerSpectrumEvoData:
    """
    Load a `MatterPowerSpectrumEvoData` written by
    `save_matter_power_spectrum_evo_data`.
    """
    file_path = _resolve_existing_path(file_path, directory)

    with h5py.File(file_path, "r") as h5_file:
        file_class = _as_str(h5_file.attrs.get("class", ""))

        if file_class == EVO_DATA_DICT_CLASS_TAG:
            raise ValueError(
                f"{file_path} holds a dict of MatterPowerSpectrumEvoData "
                "instances; use load_matter_power_spectrum_evo_data_dict(...) "
                "(or MatterPowerSpectrumEvoData.load_dict(...)) instead."
            )

        if file_class != EVO_DATA_CLASS_TAG:
            raise ValueError(
                f"{file_path} is not a {EVO_DATA_CLASS_TAG} file "
                f"(found class={file_class!r})"
            )

        return read_matter_power_spectrum_evo_data(h5_file)


# ---------------------------------------------------------------------------
# dict[key, MatterPowerSpectrumEvoData]  (e.g. keyed by resolution N)
# ---------------------------------------------------------------------------
def save_matter_power_spectrum_evo_data_dict(
        evo_data_dict: Mapping[Any, MatterPowerSpectrumEvoData],
        file_path: Path | str,
        directory: Path | str | None = None,
    ) -> Path:
    """
    Save a mapping of `MatterPowerSpectrumEvoData` instances to one file.

    Written for the `{N: MatterPowerSpectrumEvoData}` dicts produced by
    `get_joint_ps_evo_data`, where `N` is the particle-grid resolution.
    """
    file_path = resolve_hdf5_path(file_path, directory)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, "w") as h5_file:
        h5_file.attrs["class"] = EVO_DATA_DICT_CLASS_TAG
        h5_file.attrs["schema_version"] = SCHEMA_VERSION
        h5_file.attrs["n_entries"] = int(len(evo_data_dict))

        for key, evo_data in evo_data_dict.items():
            key_group = h5_file.create_group(str(key))
            key_group.attrs["key"] = str(key)
            key_group.attrs["key_type"] = (
                "int" if isinstance(key, (int, np.integer)) else "str"
            )
            write_matter_power_spectrum_evo_data(key_group, evo_data)

    return file_path


def load_matter_power_spectrum_evo_data_dict(
        file_path: Path | str,
        directory: Path | str | None = None,
    ) -> dict[Any, MatterPowerSpectrumEvoData]:
    """
    Load a mapping written by `save_matter_power_spectrum_evo_data_dict`.

    A file holding a single instance loads as a one-entry dict keyed by the
    file stem, so either flavour can be read back through this function.
    """
    file_path = _resolve_existing_path(file_path, directory)

    with h5py.File(file_path, "r") as h5_file:
        file_class = _as_str(h5_file.attrs.get("class", ""))

        if file_class == EVO_DATA_CLASS_TAG:
            return {
                file_path.stem: read_matter_power_spectrum_evo_data(h5_file)
            }

        if file_class != EVO_DATA_DICT_CLASS_TAG:
            raise ValueError(
                f"{file_path} is not a {EVO_DATA_DICT_CLASS_TAG} file "
                f"(found class={file_class!r})"
            )

        evo_data_dict = {}
        for name in h5_file.keys():
            key_group = h5_file[name]
            key_type = _as_str(key_group.attrs.get("key_type", "str"))
            key = int(name) if key_type == "int" else name
            evo_data_dict[key] = read_matter_power_spectrum_evo_data(key_group)

    return dict(sorted(evo_data_dict.items(), key=lambda item: str(item[0])))


def _resolve_existing_path(
        file_path: Path | str, directory: Path | str | None = None
    ) -> Path:
    """Resolve a save path for reading, raising if nothing is there."""
    file_path = resolve_hdf5_path(file_path, directory)

    if file_path.is_file():
        return file_path

    # Tolerate the other HDF5 suffix so `.h5`/`.hdf5` are interchangeable.
    for suffix in _HDF5_SUFFIXES:
        candidate = file_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"No such HDF5 file: {file_path}")
