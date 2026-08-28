import pdb
from functools import partial

from ..cosmo.config import CosmoType, TransferFn # Integrate in here as well...


PRIMARY_COSMO_PARAMETERS = {
    "H0" : 68.0,
    "Omega_m" : 0.306,
    "Omega_b" : 0.0487,
    "Omega_L" : 0.694,
    "sigma_8" : 0.804,
    "nspec" : 0.969,
    "w0" : -1.0,
    "wa" : 0.0,
    "cfg" : {
        "cosmo" : CosmoType.PRIMARY.value,
        "transfer_fn" : TransferFn.CLASS.value,
        "seed" : -1
    }
}

PLANCK18_COSMO_PARAMETERS = {
    "H0" : 67.66,
    "Omega_m" : 0.3111,
    "Omega_b" : 0.049,
    "Omega_L" : 0.6889,
    "sigma_8" : 0.8102,
    "nspec" : 0.9665,
    "w0" : -1.0,
    "wa" : 0.0,
    "cfg" : {
        "cosmo" : CosmoType.PLANCK.value,
        "transfer_fn" : TransferFn.CAMB.value,
        "seed" : -2
    },
}


# This way we only need to get the seed number for the variance study
TOY_MODEL_A_COSMO_PARAMETERS = {
    "H0" : 67.8,
    "Omega_m" : 0.308,
    "Omega_b" : 0.0482,
    "Omega_L" : 0.692,
    "sigma_8" : 0.9,
    "nspec" : 1.0,
    "w0" : -1.0,
    "wa" : 0.0,
    "cfg" : {
        "cosmo" : CosmoType.TOY_MODEL_A.value,
        "transfer_fn" : TransferFn.SIM_EFSTATHIOU.value,
        "seed" : 1
    },# this is the default value, but it can ve different 
}

TOY_MODEL_B_COSMO_PARAMETERS = {
    "H0" : 70.3,
    "Omega_m" : 0.276,
    "Omega_b" : 0.045,
    "Omega_L" : 0.724,
    "sigma_8" : 0.811,
    "nspec" : 0.961,
    "w0" : -1.0,
    "wa" : 0.0,
    "cfg" : {
        "cosmo" : CosmoType.TOY_MODEL_B.value,
        "transfer_fn" : TransferFn.EISENSTEIN98.value,
        "seed" : 0
    },
}


def get_cosmo_parameters(cosmo_name: str) -> dict[str, float | str | int]:
    match cosmo_name.lower():
        case CosmoType.PRIMARY.value:
            return PRIMARY_COSMO_PARAMETERS
        case CosmoType.PLANCK.value:
            return PLANCK18_COSMO_PARAMETERS
        case CosmoType.TOY_MODEL_A.value:
            return TOY_MODEL_A_COSMO_PARAMETERS
        case CosmoType.TOY_MODEL_B.value:
            return TOY_MODEL_B_COSMO_PARAMETERS
        case _:
            raise ValueError(f"Unknown cosmology model: {cosmo_name}")