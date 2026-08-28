from __future__ import annotations

import sys
from enum import StrEnum
from attrs import define, field

# ---- Enums -------------------------------------------------------------------

class CosmoType(StrEnum):
    PRIMARY = "primary"
    TOY_MODEL_A = "toy_model_a"
    TOY_MODEL_B = "toy_model_b"
    PLANCK = "planck"
    DUMMY = "sim_cosmo"


class TransferFn(StrEnum):
    EISENSTEIN98 = "eisenstein98"
    EFSTATHIOU = "efstathiou"
    CAMB = "camb"
    CLASS = "class"
    SIM_EFSTATHIOU = "simEfstathiou"   # used by your default case


# ---- Alias handling (replaces TRANSFER_FN_NAMES) -----------------------------

_TRANSFER_FN_ALIASES: dict[str, TransferFn] = {
    # canonical names
    "eisenstein98": TransferFn.EISENSTEIN98,
    "efstathiou": TransferFn.EFSTATHIOU,
    "camb": TransferFn.CAMB,
    "class": TransferFn.CLASS,
    "simefstathiou": TransferFn.SIM_EFSTATHIOU,
    "sim_efstathiou": TransferFn.SIM_EFSTATHIOU,

    # friendly aliases
    "eisenstein": TransferFn.EISENSTEIN98,
}

def parse_transfer_fn(name: str | TransferFn) -> TransferFn:
    """Normalize a TF name (e.g., 'eisenstein', 'eisenstein98', 'simEfstathiou') to TransferFn."""
    if isinstance(name, TransferFn):
        return name
    key = name.replace("-", "").replace("_", "").replace(" ", "").lower()
    try:
        return _TRANSFER_FN_ALIASES[key]
    except KeyError as e:
        raise ValueError(f"Unknown transfer function '{name}'") from e


# ---- Seed synthesis (collapses both of your original functions) --------------
@define(slots=True)
class CosmoConfig:
    cosmo: CosmoType
    transfer_fn: TransferFn

    seed: int = field(default=sys.maxsize)  # use maxsize as a sentinel for "not set"

    def __attrs_post_init__(self) -> None:
        if (self.seed == sys.maxsize):
            self.seed = get_seed_number(self.cosmo, self.transfer_fn)

    def to_dict(self) -> dict[str, int | str]:
        return {
            "seed": self.seed,
            "cosmo": self.cosmo.value,
            "transfer_fn": self.transfer_fn.value,
        }
    
    @classmethod
    def from_str(cls, name: str, transfer: str) -> CosmoConfig:
        return cls(
            cosmo=CosmoType(name.lower()),
            transfer_fn=parse_transfer_fn(transfer),
        )


# single source of truth for seed -> (cosmo, transfer)
_SEED_CONFIG: dict[int, tuple[CosmoType, TransferFn]] = {
    -2: (CosmoType.PLANCK,     TransferFn.CAMB),
    -1: (CosmoType.PRIMARY,    TransferFn.CLASS),
    0: (CosmoType.TOY_MODEL_B, TransferFn.EISENSTEIN98),
}
for seed in range(1, 9):
    _SEED_CONFIG[seed] = (CosmoType.TOY_MODEL_A, TransferFn.SIM_EFSTATHIOU)
    _SEED_CONFIG[seed + 10] = (CosmoType.TOY_MODEL_A, TransferFn.SIM_EFSTATHIOU)

def config_from_seed(seed: int = -1) -> CosmoConfig:
    """Return both the CosmoType and TransferFn for a given seed."""
    cosmo, tf = _SEED_CONFIG.get(seed, (CosmoType.TOY_MODEL_A, TransferFn.SIM_EFSTATHIOU))
    return CosmoConfig(seed=seed, cosmo=cosmo, transfer_fn=tf)


def get_cosmo_type(seed: int = -1) -> CosmoType:
    return config_from_seed(seed).cosmo

def get_cosmo_name(seed: int = -1) -> str:
    return config_from_seed(seed).cosmo.value

def get_transfer_model_type(seed: int = -1) -> str:
    return config_from_seed(seed).transfer_fn.value


def get_seed_number(cosmo_type: CosmoType, transfer_fn: TransferFn) -> int:
    return next(
        (
            seed
            for seed, (c, tf) in _SEED_CONFIG.items()
            if c == cosmo_type and tf == transfer_fn
        ),
        sys.maxsize,  # not found (sentinel
    )