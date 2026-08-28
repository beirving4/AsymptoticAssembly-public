import numpy as np 

# For generating composite halo particles 

DEFAULT_NUM_BINS = 30

def make_6D_histogram(
        phase_space_data: list[np.ndarray],
        masses: np.ndarray, 
        num_bins: int = DEFAULT_NUM_BINS
    ) -> dict[str, np.ndarray| list[np.ndarray]]: 

    denisty, edges = np.histogramdd(
        sample=np.vstack(phase_space_data),
        bins=num_bins,
        density=True,
        weights=masses
    )

    return {"edges": edges, "density": denisty}