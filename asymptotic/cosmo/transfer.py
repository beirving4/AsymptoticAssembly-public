import numpy as np
from colossus.cosmology.power_spectrum import modelEisenstein98, modelCamb

def modelEfstathiou86(k: np.ndarray, shape: float = 0.21) -> np.ndarray:
    q = k / shape
    return (1.0 + ((6.4 * q) + (3 * q)**1.5 + (1.7 * q)**2)**1.13)**(-1.77)

# def toy_model_b_Tk(k: np.ndarray) -> np.ndarray:
#     return modelEisenstein98(
#         k, 
#         h=0.7, 
#         Om0=0.3, 
#         Ob0=0.05, 
#         Tcmb0=2.725
#     )