import numpy as np, pdb 

from astropy import units as u
from collections.abc import Collection
from scipy.interpolate import interp1d
from astropy import constants as const
from scipy.signal import savgol_filter, argrelmin, argrelmax

from ..cosmo.model import Cosmology

DEFAULT_MIN_RADIUS = 0.0
DEFAULT_MAX_RADIUS = np.inf
DEFAULT_WINDOW_SIZE = 11
DEFAULT_POLY_ORDER = 4

NEWTONS_GRAVITATIONAL_CONSTANT = const.G.to("Mpc**3 / (Msun * s**2)")

def get_circular_velocity(mass: float, radius: float) -> float:
    G = NEWTONS_GRAVITATIONAL_CONSTANT
    return np.sqrt(G * (mass * u.Msun) / (radius * u.Mpc)).to("km/s").value
def get_rotation_curve(radii: np.ndarray, masses: np.ndarray) -> np.ndarray:
    G = NEWTONS_GRAVITATIONAL_CONSTANT
    return np.sqrt(G * (masses * u.Msun) / (radii * u.Mpc)).to("km/s").value
def get_radial_mask(radii: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    return np.logical_and(radii >= lower_bound, radii <= upper_bound)

def filtered_profile(
        x_values: np.ndarray,
        y_values: np.ndarray,
        window_size: int = DEFAULT_WINDOW_SIZE,
        poly_order: int = DEFAULT_POLY_ORDER,
        derivative: int = 0,
        mode: str = "interp",
    ) -> np.ndarray:

    return savgol_filter(
        y_values,
        window_length=window_size,
        polyorder=poly_order,
        deriv=derivative,
        mode=mode,
        delta=np.diff(x_values).mean(),
    )


def get_slope(
        x_values : np.ndarray, 
        y_values: np.ndarray,
        window_size: int = DEFAULT_WINDOW_SIZE,
        poly_order: int = DEFAULT_POLY_ORDER,
        mode: str = "interp",
    ) -> np.ndarray:
    # slope = np.gradient(np.log(y_values), np.log(x_values))
    # filtered_slope = savgol_filter(slope, window_size, poly_order)

    filtered_slope = 10.0**filtered_profile(
        np.log10(x_values),
        np.log10(y_values),
        window_size=window_size,
        poly_order=poly_order,
        derivative=1,
        mode=mode,
    )

    return np.array([x_values, filtered_slope]).T

def get_density_slope(
        profiles: np.ndarray, 
        min_radius: float = DEFAULT_MIN_RADIUS, 
        max_radius: float = DEFAULT_MAX_RADIUS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        poly_order: int = DEFAULT_POLY_ORDER
    ) -> np.ndarray:

    mask = get_radial_mask(profiles[:,0], min_radius, max_radius)
    return get_slope(profiles[mask,0], profiles[mask,3], window_size, poly_order)

def get_enclosed_density_slope(
        profiles: np.ndarray, 
        min_radius: float = DEFAULT_MIN_RADIUS, 
        max_radius: float = DEFAULT_MAX_RADIUS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        poly_order: int = DEFAULT_POLY_ORDER
    ) -> np.ndarray:

    mask = get_radial_mask(profiles[:,0], min_radius, max_radius)
    slope = get_slope(profiles[mask,1], profiles[mask,2], window_size, poly_order)
    return np.array([profiles[mask, 0], slope[:,1]]).T

def get_scale_radius(
        profiles: np.ndarray, 
        min_radius: float = DEFAULT_MIN_RADIUS, 
        max_radius: float = DEFAULT_MAX_RADIUS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        poly_order: int = DEFAULT_POLY_ORDER
    ) -> float:
    slope = get_density_slope(
        profiles, 
        min_radius=min_radius, 
        max_radius=max_radius,
        window_size=window_size,
        poly_order=poly_order
    )
    loc_steepest_slope = np.nanargmin(slope[:,1])

    try: 
        slope_interp = interp1d(
            slope[:loc_steepest_slope,1], 
            np.log10(slope[:loc_steepest_slope,0]), 
            fill_value="extrapolate"
        )
        return float(10.0**np.asarray(slope_interp(-2.0)))
    except ValueError:
        return np.nan
    
def find_radial_minimum(radii: np.ndarray, values: np.ndarray) -> float:
    minima_idxs = argrelmin(values)[0]
    if len(minima_idxs) == 0: return np.nan
    minima_values, minima_radii = values[minima_idxs], radii[minima_idxs]
    return minima_radii[np.argmin(minima_values)]

def find_enclosed_mass_minimum(enc_mass_profile: np.ndarray, values: np.ndarray) -> float:
    minima_idxs = argrelmin(values)[0]
    if len(minima_idxs) == 0: return np.nan
    minima_values, minima_radii = values[minima_idxs], enc_mass_profile[minima_idxs]
    return minima_radii[np.argmin(minima_values)]


def find_radial_maximum(radii: np.ndarray, values: np.ndarray) -> float:
    maxima_idxs = argrelmax(values)[0]
    if len(maxima_idxs) == 0: return np.nan
    maxima_values, maxima_radii = values[maxima_idxs], radii[maxima_idxs]
    return maxima_radii[np.argmax(maxima_values)]

def find_rotation_curve_maximum(radii: np.ndarray, masses: np.ndarray) -> tuple[float, float]:
    rotation_curve = get_rotation_curve(masses, radii)
    maxima_idxs = argrelmax(rotation_curve)[0]
    if len(maxima_idxs) == 0: return np.nan, np.nan
    maxima_velocities, maxima_radii = rotation_curve[maxima_idxs], radii[maxima_idxs]
    max_loc = np.argmax(maxima_velocities)
    return maxima_radii[max_loc], maxima_velocities[max_loc]



def get_splashback_radius(
        profiles: np.ndarray, 
        min_radius: float = DEFAULT_MIN_RADIUS, 
        max_radius: float = DEFAULT_MAX_RADIUS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        poly_order: int = DEFAULT_POLY_ORDER
    ) -> float:

    slope = get_density_slope(
        profiles, 
        min_radius=min_radius, 
        max_radius=max_radius,
        window_size=window_size,
        poly_order=poly_order
    )
    return find_radial_minimum(slope[:,0], slope[:,1])

def get_last_bound_radius(
        profiles: np.ndarray, 
        min_radius: float = DEFAULT_MIN_RADIUS, 
        max_radius: float = DEFAULT_MAX_RADIUS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        poly_order: int = DEFAULT_POLY_ORDER
    ) -> float:
    slope = get_enclosed_density_slope(
        profiles, 
        min_radius=min_radius, 
        max_radius=max_radius,
        window_size=window_size,
        poly_order=poly_order
    )
    return find_radial_minimum(slope[:,0], slope[:,1])  

def get_splashback_info(
        profiles: np.ndarray, 
        min_radius: float = DEFAULT_MIN_RADIUS, 
        max_radius: float = DEFAULT_MAX_RADIUS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        poly_order: int = DEFAULT_POLY_ORDER
    ) -> tuple[float, float]:

    r_splashback = get_splashback_radius(
        profiles, 
        min_radius=min_radius, 
        max_radius=max_radius,
        window_size=window_size,
        poly_order=poly_order
    )

    mass_interp = interp1d(
        np.log10(profiles[:,0]), 
        np.log10(profiles[:,1]),
        fill_value="extrapolate"
    )

    splashback_mass = 10.0**mass_interp(np.log10(r_splashback))

    return (r_splashback, splashback_mass)

def get_v_max(
        profiles: np.ndarray, 
        min_radius: float = DEFAULT_MIN_RADIUS, 
        max_radius: float = DEFAULT_MAX_RADIUS
    ) -> tuple[float, float, float]:

    
    mask = np.logical_and.reduce([
        profiles[:, 0] >= min_radius, 
        profiles[:, 0] <= max_radius,
        np.isfinite(profiles[:, 0]),
        np.isfinite(profiles[:, 1])
    ])


    # Ensure arrays are numpy arrays and sorted by r
    r = profiles[:, 0][mask]
    M = profiles[:, 1][mask]
    v_circ = get_rotation_curve(M, r)

    # Compute derivative dv/dr using finite differences
    dv = np.gradient(v_circ, r)

    # Identify indices where dv changes from positive to negative:
    # This indicates a local maximum candidate.
    candidates = []
    for i in range(len(r) - 1):
        if dv[i] > 0 and dv[i + 1] < 0:
            # We have a sign change from + to - around r[i], r[i+1]
            
            # For more accurate location, linearly interpolate dv around this region
            # dv changes from dv[i] at r[i] to dv[i+1] at r[i+1]
            slope = (dv[i+1] - dv[i]) / (r[i+1] - r[i])
            # We want dv = 0: r_zero = r[i] - dv[i]/slope
            r_zero = r[i] - dv[i] / slope

            # Interpolate v_circ at r_zero for a more accurate v_max
            v_interp = interp1d(r, v_circ, kind='cubic', fill_value="extrapolate")
            v0 = v_interp(r_zero)

            candidates.append((r_zero, v0))

    # If no sign change found, fallback to global maximum
    if not candidates:
        idx = np.argmax(v_circ)
        r_max = r[idx]
        v_max = v_circ[idx]
    else:
        # If multiple candidates, pick the one with the largest v_circ
        candidates.sort(key=lambda x: x[1], reverse=True)
        r_max, v_max = candidates[0]

    # Interpolate M to find M_max at r_max
    mass_interp = interp1d(
        np.log10(r),
        np.log10(M),
        fill_value="extrapolate"
    )
    M_max = 10.0 ** mass_interp(np.log10(r_max))

    return r_max, v_max, M_max



def get_mass_distribution_milestones(
        profiles: np.ndarray, 
        min_radius: float = DEFAULT_MIN_RADIUS, 
        max_radius: float = DEFAULT_MAX_RADIUS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        poly_order: int = DEFAULT_POLY_ORDER
    ) -> dict[str, np.ndarray | float]:

    r_scale = get_scale_radius(
        profiles, 
        min_radius=min_radius, 
        max_radius=max_radius,
        window_size=window_size,
        poly_order=poly_order
    )
    four_scale = 4.0 * r_scale
    r_splashback = get_splashback_radius(
        profiles, 
        min_radius=min_radius, 
        max_radius=max_radius,
        window_size=window_size,
        poly_order=poly_order
    )
    r_last_bound = get_last_bound_radius(
        profiles, 
        min_radius=min_radius, 
        max_radius=max_radius,
        window_size=window_size,
        poly_order=poly_order
    )
    r_max, v_max, M_max = get_v_max(
        profiles, min_radius=min_radius, max_radius=max_radius
    )

    mass_interp = interp1d(
        np.log10(profiles[:,0]), 
        np.log10(profiles[:,1]),
        fill_value="extrapolate"
    )

    enclosed_density_interp = interp1d(
        np.log10(profiles[:,0]), 
        np.log10(profiles[:,2]),
        fill_value="extrapolate"
    )

    local_density_interp = interp1d(
        np.log10(profiles[:,0]),
        np.log10(profiles[:,3]),
        fill_value="extrapolate"
    )

    scale_mass = 10.0**mass_interp(np.log10(r_scale))
    M4rs = 10.0**mass_interp(np.log10(four_scale))
    splashback_mass = 10.0**mass_interp(np.log10(r_splashback))
    last_bound_mass = 10.0**mass_interp(np.log10(r_last_bound))

    enclosed_scale = 10.0**enclosed_density_interp(np.log10(r_scale))
    enclosed_4rs = 10.0**enclosed_density_interp(np.log10(four_scale))
    enclosed_splashback = 10.0**enclosed_density_interp(np.log10(r_splashback))
    enclosed_last_bound = 10.0**enclosed_density_interp(np.log10(r_last_bound))
    enclosed_max = 10.0**enclosed_density_interp(np.log10(r_max))

    local_scale = 10.0**local_density_interp(np.log10(r_scale))
    local_4rs = 10.0**local_density_interp(np.log10(four_scale))
    local_splashback = 10.0**local_density_interp(np.log10(r_splashback))
    local_last_bound = 10.0**local_density_interp(np.log10(r_last_bound))
    local_max = 10.0**local_density_interp(np.log10(r_max))

    v_scale = get_circular_velocity(scale_mass, r_scale)
    v_4rs = get_circular_velocity(M4rs, four_scale)
    v_splash = get_circular_velocity(splashback_mass, r_splashback)
    v_last = get_circular_velocity(last_bound_mass, r_last_bound)

    gamma = get_density_slope(
        profiles=profiles, 
        min_radius=min_radius, 
        max_radius=max_radius,
        window_size=window_size, 
        poly_order=poly_order
    )
    kappa = get_enclosed_density_slope(
        profiles=profiles, 
        min_radius=min_radius, 
        max_radius=max_radius,
        window_size=window_size, 
        poly_order=poly_order
    )


    return {
        "slope" : gamma,
        "enclosed_slope" : kappa,
        "scale_radius" : r_scale,
        "scale_mass" : scale_mass,
        "enclosed_scale_rho" : enclosed_scale,
        "M4rs" : M4rs,
        "enclosed_4rs_rho" : enclosed_4rs,
        "four_scale" : four_scale,
        "splashback" : r_splashback,
        "last_bound" : r_last_bound,
        "splashback_mass" : splashback_mass,
        "last_bound_mass" : last_bound_mass,
        "v_splash" : v_splash,
        "v_last" : v_last,
        "v_scale" : v_scale,
        "v_4rs" : v_4rs,
        "enclosed_splashback_rho" : enclosed_splashback,
        "enclosed_last_bound_rho" : enclosed_last_bound,
        "enclosed_max_rho" : enclosed_max,
        "v_max" : v_max,
        "r_max" : r_max,
        "M_max" : M_max,
        "local_scale_rho" : local_scale,
        "local_4rs_rho" : local_4rs,
        "local_splashback_rho" : local_splashback,
        "local_last_bound_rho" : local_last_bound,
        "local_max_rho" : local_max
    }

def get_filtered_scale_info(
        radii: np.ndarray, 
        gamma: np.ndarray, 
        enclosed_mass_interp: interp1d,
        enclosed_density_interp: interp1d,
        local_density_interp: interp1d,
    ) -> dict[str, float]:

    loc_steepest_slope = np.nanargmin(gamma)

    try:

        slope_interp = interp1d(
            gamma[:loc_steepest_slope], 
            np.log10(radii[:loc_steepest_slope]), 
            fill_value="extrapolate"
        )

        r_scale = float(10.0**np.asarray(slope_interp(-2.0)))
        M_scale = 10.0**enclosed_mass_interp(np.log10(r_scale))

        r_4s = 4.0 * r_scale
        M4rs = 10.0**enclosed_mass_interp(np.log10(r_4s))

        return {
            "r_scale" : r_scale,
            "M_scale" : M_scale,
            "v_scale" : get_circular_velocity(M_scale, r_scale),
            "rho_enc_scale" : 10.0**enclosed_density_interp(np.log10(r_scale)),
            "rho_local_scale" : 10.0**local_density_interp(np.log10(r_scale)),
            "r_4s" : r_4s,
            "M4rs" : M4rs,
            "v_4rs" : get_circular_velocity(M4rs, r_4s),
            "rho_enc_4s" : 10.0**enclosed_density_interp(np.log10(r_4s)),
            "rho_local_4s" : 10.0**local_density_interp(np.log10(r_4s))
        }
    
    except ValueError:
        return {
            "r_scale" : np.nan,
            "M_scale" : np.nan, 
            "v_scale" : np.nan,
            "rho_enc_scale" : np.nan,
            "rho_local_scale" : np.nan,
            "r_4s" : np.nan,
            "M4rs" : np.nan,
            "v_4rs" : np.nan,
            "rho_enc_4s" : np.nan,
            "rho_local_4s" : np.nan,
        }
    
# def get_filtered_splashback_info(
#         enclosed_mass_profile: np.ndarray, 
#         gamma: np.ndarray, 
#         enclosed_mass_to_radii_interp: interp1d,
#         enclosed_density_interp: interp1d,
#         local_density_interp: interp1d,
#     ) -> dict[str, float]:

#     M_sp = find_enclosed_mass_minimum(enclosed_mass_profile, gamma)
#     r_sp = 10.0**enclosed_mass_to_radii_interp(np.log10(M_sp))

#     return {
#         "r_splashback" : r_sp,
#         "M_splashback" : M_sp,
#         "v_splashback" : get_circular_velocity(M_sp, r_sp),
#         "rho_enc_splashback" : 10.0**enclosed_density_interp(np.log10(r_sp)),
#         "rho_local_splashback" : 10.0**local_density_interp(np.log10(r_sp)),
#     }

def get_filtered_bound_info(
        enclosed_mass_profile: np.ndarray, 
        kappa: np.ndarray,
        enclosed_mass_to_radii_interp: interp1d,
        enclosed_density_interp: interp1d,
        local_density_interp: interp1d,
    ) -> dict[str, float]:

    M_bound = find_enclosed_mass_minimum(enclosed_mass_profile, kappa)
    r_bound = 10.0**enclosed_mass_to_radii_interp(np.log10(M_bound))

    return {
        "r_bound" : r_bound, 
        "M_bound" : M_bound, 
        "v_bound" : get_circular_velocity(M_bound, r_bound),
        "rho_enc_bound" : 10.0**enclosed_density_interp(np.log10(r_bound)),
        "rho_local_bound" : 10.0**local_density_interp(np.log10(r_bound)),
    }
    
def get_filtered_splashback_info(
        radii: np.ndarray, 
        gamma: np.ndarray, 
        enclosed_mass_interp: interp1d,
        enclosed_density_interp: interp1d,
        local_density_interp: interp1d,
    ) -> dict[str, float]:
    
    r_sp = find_radial_minimum(radii, gamma)
    M_sp = 10.0**enclosed_mass_interp(np.log10(r_sp))

    return {
        "r_splashback" : r_sp,
        "M_splashback" : M_sp,
        "v_splashback" : get_circular_velocity(M_sp, r_sp),
        "rho_enc_splashback" : 10.0**enclosed_density_interp(np.log10(r_sp)),
        "rho_local_splashback" : 10.0**local_density_interp(np.log10(r_sp)),
    }

# def get_filtered_bound_info(
#         radii: np.ndarray,
#         kappa: np.ndarray,
#         enclosed_mass_interp: interp1d,
#         enclosed_density_interp: interp1d,
#         local_density_interp: interp1d,
#     ) -> dict[str, float]:

#     r_bound = find_radial_minimum(radii, kappa)
#     M_bound = 10.0**enclosed_mass_interp(np.log10(r_bound))

#     return {
#         "r_bound" : r_bound, 
#         "M_bound" : M_bound, 
#         "v_bound" : get_circular_velocity(M_bound, r_bound),
#         "rho_enc_bound" : 10.0**enclosed_density_interp(np.log10(r_bound)),
#         "rho_local_bound" : 10.0**local_density_interp(np.log10(r_bound)),
#     }

def get_filtered_vmax_info(
        radii: np.ndarray,
        enclosed_mass_interp: interp1d,
        enclosed_density_interp: interp1d,
        local_density_interp: interp1d,
    ) -> dict[str, float]:

    enclosed_masses = 10.0**enclosed_mass_interp(np.log10(radii))
    r_v_max, v_max = find_rotation_curve_maximum(radii, enclosed_masses)

    return {
        "r_v_max" : r_v_max,
        "M_v_max" : 10.0**enclosed_mass_interp(np.log10(r_v_max)),
        "v_max" : v_max,
        "rho_enc_v_max" : 10.0**enclosed_density_interp(np.log10(r_v_max)),
        "rho_local_v_max" : 10.0**local_density_interp(np.log10(r_v_max)),
    }

    

def get_filtered_mass_distribution_milestones(
        filtered_profiles: np.ndarray,
        min_radius: float = DEFAULT_MIN_RADIUS,
        max_radius: float = DEFAULT_MAX_RADIUS,
    ) -> dict[str, float]:

    radii_mask = np.logical_and(
        filtered_profiles[:, 0] >= min_radius,
        filtered_profiles[:, 0] <= max_radius
    )

    enclosed_mass_interp = interp1d(
        np.log10(filtered_profiles[:, 0]),
        np.log10(filtered_profiles[:, 1]),
        fill_value="extrapolate"
    )

    enclosed_mass_to_radii_interp = interp1d(
        np.log10(filtered_profiles[:, 1]),
        np.log10(filtered_profiles[:, 0]),
        fill_value="extrapolate"
    )

    enclosed_density_interp = interp1d(
        np.log10(filtered_profiles[:, 0]),
        np.log10(filtered_profiles[:, 2]),
        fill_value="extrapolate"
    )

    local_density_interp = interp1d(
        np.log10(filtered_profiles[:, 0]),
        np.log10(filtered_profiles[:, 3]),
        fill_value="extrapolate"
    )

    scale_info = get_filtered_scale_info(
        radii=filtered_profiles[radii_mask, 0],
        gamma=filtered_profiles[radii_mask, 5],
        enclosed_mass_interp=enclosed_mass_interp,
        enclosed_density_interp=enclosed_density_interp,
        local_density_interp=local_density_interp,
    )

    splashback_info = get_filtered_splashback_info(
        radii=filtered_profiles[radii_mask, 0],
        gamma=filtered_profiles[radii_mask, 5],
        enclosed_mass_interp=enclosed_mass_interp,
        enclosed_density_interp=enclosed_density_interp,
        local_density_interp=local_density_interp,
    )

    bound_info = get_filtered_bound_info(
        enclosed_mass_profile=filtered_profiles[radii_mask, 1],
        kappa=filtered_profiles[radii_mask, 6],
        enclosed_mass_to_radii_interp=enclosed_mass_to_radii_interp,
        enclosed_density_interp=enclosed_density_interp,
        local_density_interp=local_density_interp,
    )

    vmax_info = get_filtered_vmax_info(
        radii=filtered_profiles[radii_mask, 0],
        enclosed_mass_interp=enclosed_mass_interp,
        enclosed_density_interp=enclosed_density_interp,
        local_density_interp=local_density_interp,
    )

    return {**scale_info, **splashback_info, **bound_info, **vmax_info}

    

def get_enc_slope_interp(
        slope_array: np.ndarray,
        x_new: float | Collection[float] | np.ndarray
    ) -> float | np.ndarray:
    """
    Interpolates y-values for given x-values using slope_array data,
    accounting for negative logarithmically spaced y-values.

    Parameters:
        slope_array (np.ndarray): An array of shape (n, 2) where:
            - slope_array[:, 0] are the x-values (must be positive).
            - slope_array[:, 1] are the y-values (may be negative and logarithmically spaced).
        x_new (float | Sequence[float] | np.ndarray): The x-values at which to interpolate.

    Returns:
        float | np.ndarray: The interpolated y-values at x_new.

    Raises:
        ValueError: If any x_values are non-positive or y_values are zero.
    """
    # Extract x_values and y_values
    x_values = slope_array[:, 0]
    y_values = slope_array[:, 1]

    # Check for valid x_values
    if np.any(x_values <= 0):
        raise ValueError("All x_values must be positive to compute log10.")

    # Check for zero y_values
    if np.any(y_values == 0):
        raise ValueError("y_values contain zero; cannot compute logarithm of zero.")

    # Compute the logarithm of x_values
    log_x = np.log10(x_values)

    # Compute the logarithm of the absolute y_values
    log_abs_y = np.log10(np.abs(y_values))

    # Get the sign of y_values
    sign_y = np.sign(y_values)

    # Create the interpolator for log_abs_y
    interp_log_abs_y = interp1d(
        log_x,
        log_abs_y,
        fill_value="extrapolate"
    )

    # Check if signs vary
    if np.all(sign_y == sign_y[0]):
        # All signs are the same
        constant_sign = sign_y[0]

        def interpolator(x_vals):
            x_vals = np.asarray(x_vals)
            log_x_new = np.log10(x_vals)
            log_abs_y_new = interp_log_abs_y(log_x_new)
            y_new = constant_sign * 10 ** log_abs_y_new
            return y_new

    else:
        # Signs vary; interpolate sign separately
        sign_interp = interp1d(
            log_x,
            sign_y,
            kind='nearest',  # Nearest interpolation for discrete signs
            fill_value="extrapolate"
        )

        def interpolator(x_vals):
            x_vals = np.asarray(x_vals)
            log_x_new = np.log10(x_vals)
            log_abs_y_new = interp_log_abs_y(log_x_new)
            sign_y_new = sign_interp(log_x_new)
            y_new = sign_y_new * 10 ** log_abs_y_new
            return y_new

    # Perform the interpolation and return the result
    y_new = interpolator(x_new)
    return y_new


def get_so_values(profiles: np.ndarray, z: float, cosmo: Cosmology) -> dict[str, dict[str, float]]:
    
    interp_logM = interp1d(np.log10(profiles[:,2]), np.log10(profiles[:,1]), fill_value="extrapolate")
    interp_logR = interp1d(np.log10(profiles[:,1]), np.log10(profiles[:,0]), fill_value="extrapolate")

    rho200c = cosmo.get_density_threshold(200, z=z, base="critical")
    rho200m = cosmo.get_density_threshold(200, z=z, base="mean")
    rho500c = cosmo.get_density_threshold(500, z=z, base="critical")
    rho_vir = cosmo.virial_density(z)
    
    M200c = 10.0**np.asarray(interp_logM(np.log10(rho200c)))
    R200c = 10.0**np.asarray(interp_logR(np.log10(M200c)))
    V200c = get_circular_velocity(M200c, R200c)
    
    M200m = 10.0**np.asarray(interp_logM(np.log10(rho200m)))
    R200m = 10.0**np.asarray(interp_logR(np.log10(M200m)))
    V200m = get_circular_velocity(M200m, R200m)
    
    M500c = 10.0**np.asarray(interp_logM(np.log10(rho500c)))
    R500c = 10.0**np.asarray(interp_logR(np.log10(M500c)))
    V500c = get_circular_velocity(M500c, R500c)

    Mvir = 10.0**np.asarray(interp_logM(np.log10(rho_vir)))
    Rvir = 10.0**np.asarray(interp_logR(np.log10(Mvir)))
    Vvir = get_circular_velocity(Mvir, Rvir)
    
    return {
        "rho" : {"200c": rho200c, "200m": rho200m, "500c": rho500c, "vir": rho_vir},
        "M" : {"200c": M200c, "200m": M200m, "500c": M500c, "vir": Mvir},
        "R" : {"200c": R200c, "200m": R200m, "500c": R500c, "vir": Rvir},
        "V" : {"200c": V200c, "200m": V200m, "500c": V500c, "vir": Vvir},
    }

def get_filtered_enclosed_masses(
        radii: np.ndarray,
        filtered_radii: np.ndarray,
        enclosed_masses: np.ndarray,
        window_size: int = DEFAULT_WINDOW_SIZE,
        poly_order: int = DEFAULT_POLY_ORDER,
        mode: str = "interp",
    ) -> np.ndarray:

    enclosed_mass_interp = interp1d(
        np.log10(radii), 
        np.log10(enclosed_masses),
        fill_value="extrapolate"
    )
    interped_enclosed_masses = enclosed_mass_interp(np.log10(filtered_radii))

    return 10.0**filtered_profile(
        x_values=np.log10(filtered_radii),
        y_values=interped_enclosed_masses,
        window_size=window_size,
        poly_order=poly_order,
        derivative=0,
        mode=mode,
    )

def get_filtered_mass_distribution(
        particle_masses: np.ndarray, 
        particle_radii: np.ndarray,
        size_in_dex: float = 1.0,
        poly_order: int = 4,
        mode: str = "interp",
        num_bins: int = 100,
        radii_are_pre_sorted: bool = True,
    ) -> np.ndarray:

    if not radii_are_pre_sorted:
        order = np.argsort(particle_radii)
        radii, masses = particle_radii[order], particle_masses[order]
    else:
        radii, masses = particle_radii, particle_masses

    min_num_particles = 0
    enclosed_masses = np.cumsum(masses)[min_num_particles:]
    radii = radii[min_num_particles:]
    min_radius, max_radius = np.nanmin(radii[radii > 0.0]), np.nanmax(radii)
    
    filtered_radii = np.logspace(
        np.log10(min_radius), 
        np.log10(max_radius), 
        num_bins,
    )

    n_dex = np.log10(max_radius / min_radius)
    bins_per_dex = num_bins / n_dex

    window_size = max(DEFAULT_WINDOW_SIZE, int(size_in_dex * bins_per_dex))
    if window_size % 2 == 0:
        window_size += 1

    filtered_enclosed_masses = get_filtered_enclosed_masses(
        radii=radii[radii > 0.0],
        filtered_radii=filtered_radii,
        enclosed_masses=enclosed_masses[radii > 0.0],
        window_size=window_size,
        poly_order=poly_order,
        mode=mode,
    )


    filtered_enclosed_densities = (
        (3.0 * filtered_enclosed_masses) / (4.0 * np.pi * filtered_radii**3)
    )

    mass_slope = filtered_profile(
        x_values=np.log10(filtered_radii),
        y_values=np.log10(filtered_enclosed_masses),
        window_size=window_size,
        poly_order=2, # this was found to work best
        derivative=1,
        mode=mode,
    )

    local_density = (filtered_enclosed_densities / 3.0) * mass_slope

    gamma = filtered_profile(
        x_values=np.log10(filtered_radii),
        y_values=np.log10(local_density),
        window_size=window_size,
        poly_order=poly_order,
        derivative=1,
        mode=mode,
    )

    kappa = 1.0 - (filtered_enclosed_densities / local_density)

    return np.column_stack([
        filtered_radii,
        filtered_enclosed_masses,
        filtered_enclosed_densities,
        local_density,
        mass_slope,
        gamma,
        kappa,
    ])


def get_toms_filtered_mass_distribution(
        particle_masses: np.ndarray, 
        particle_radii: np.ndarray,
        num_particles_in_dex: int = 60,
        radii_are_pre_sorted: bool = True,
    ) -> np.ndarray:

    if not radii_are_pre_sorted:
        order = np.argsort(particle_radii)
        radii, masses = particle_radii[order], particle_masses[order]
    else:
        radii, masses = particle_radii, particle_masses

    min_num_particles = 0
    enclosed_masses = np.cumsum(masses)[min_num_particles:]
    radii = radii[min_num_particles:]
    min_radius, max_radius = np.nanmin(radii[radii > 0.0]), np.nanmax(radii)

    num_bins = int (np.log10(max_radius / min_radius) * num_particles_in_dex)
    
    filtered_radii = np.logspace(
        np.log10(min_radius), 
        np.log10(max_radius), 
        num_bins,
    )

    enclosed_mass_interp = interp1d(
        np.log10(radii), 
        np.log10(enclosed_masses),
        kind="linear",
        fill_value="extrapolate"
    )
    filtered_enclosed_masses = 10.0**enclosed_mass_interp(np.log10(filtered_radii))
    filtered_enclosed_density = (
        (3.0 * filtered_enclosed_masses) / (4.0 * np.pi * filtered_radii**3)
    )

    mass_slope = np.gradient(
        np.log(filtered_enclosed_masses), 
        np.log(filtered_radii)
    )

    radii_slope = np.gradient(
        np.log(filtered_radii), 
        np.log(filtered_enclosed_masses)
    )

    local_density = (filtered_enclosed_density / 3.0) * mass_slope

    gamma = np.gradient(np.log(local_density), np.log(filtered_radii))  
    kappa = 1.0 - (filtered_enclosed_density / local_density)

    return np.column_stack([
        filtered_radii,
        filtered_enclosed_masses,
        filtered_enclosed_density,
        local_density,
        mass_slope,
        gamma,
        kappa,
        radii_slope,
    ])