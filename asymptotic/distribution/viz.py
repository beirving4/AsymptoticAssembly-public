import numpy as np, pdb
import matplotlib.pylab as plt


from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter

from ..mass_def.base import MASS_DEF_COLORS
from ..model.distribution import AsymptoticProfile
from ..mass_def.mapping import global_mass_def_mapping


PROFILE_Y_LABELS = {
    "density": {
        1: r"Density, $\frac{\rho(r)}{\rho_{200c}}$",
        2: r"Density, $\rho(r)$ $\left[h^{2} M_{\odot} {\rm Mpc}^{-3}\right]$",
    },
    "scaled_density": {
        1: r"Scaled Density, $\frac{\rho(r)}{\rho_{200c}}\left(\frac{r}{r_{200c}}\right)^{2}$",
        2: r"Scaled Density, $\rho(r) r^{2}$ $\left[M_{\odot} {\rm Mpc}^{-1}\right]$",
    },
    "mass_enclosed": {
        1: r"Mass Enclosed, $\frac{M(<r)}{M_{200c}}$",
        2: r"Mass Enclosed, $M(<r)$ $\left[h^{-1} M_{\odot}\right]$",
    },
    "enclosed_overdensity": {
        1: r"Enclosed Overdensity, $\delta = \frac{\bar{\rho}(< R)}{\rho_{M}(z)} - 1$",
        2: r"Enclosed Overdensity, $\delta = \frac{\bar{\rho}(< R)}{\rho_{M}(z)} - 1$",
    },
    "mean_density": {
        1: r"Mean Density, $\frac{\rho(r)}{\rho_{200c}}$",
        2: r"Mean Density, $\rho(r)$ $\left[h^{2} M_{\odot} {\rm Mpc}^{-3}\right]$",
    },
}


def get_profile_xlabel(radius_units: float) -> str:
    if radius_units == 1:
        return r"Radial Distance, $\frac{r}{R_{200c}}$"
    else:
        return r"Radial Distance, $r$ $\left[h^{-1} {\rm Mpc}\right]$"


def get_profile_ylabel(relation: str, units: float) -> str:
    return PROFILE_Y_LABELS[relation][1 if units == 1 else 2]


def get_profile_labels(
    radius_units: float, y_relation: str, yval_units: float
) -> tuple[str, str]:
    return (
        get_profile_xlabel(radius_units),
        get_profile_ylabel(y_relation, yval_units),
    )


def get_fitted_profiles(
    density_profile: np.ndarray,
    scaled_density_profile: np.ndarray,
    mass_enclosed_profile: np.ndarray,
    enclosed_overdensity_profile: np.ndarray,
    rho_mean: float,
    mass_units: float = 1,
    density_units: float = 1,
    scaled_density_units: float = 1,
    profile_fit: AsymptoticProfile | None = None,
) -> dict[str, np.ndarray | dict[str, np.ndarray] | None]:
    if profile_fit is None:
        return {
            "density": None,
            "scaled_density": None,
            "mass_enclosed": None,
            "enclosed_overdensity": None,
            "fit_params": None,
        }

    if rho_mean is None:
        raise ValueError("rho_mean must be provided if profile_fit is not None")

    fit_params = {k: v for k, v in profile_fit.parameters.items() if k != r"$a$"}

    fitted_density_profile = (
        profile_fit.density(density_profile[0], logged=False) * density_units
    )
    fitted_scaled_density_profile = (
        profile_fit.scaled_density(scaled_density_profile[0])[1] * scaled_density_units
    )
    fitted_mass_enclosed_profile = (
        profile_fit.mass_enclosed(mass_enclosed_profile[0]) * mass_units
    )
    fitted_enclosed_overdensity_profile = profile_fit.enclosed_overdensity(
        enclosed_overdensity_profile[0], rho_mean=rho_mean
    )[1]

    return {
        "density": fitted_density_profile,
        "scaled_density": fitted_scaled_density_profile,
        "mass_enclosed": fitted_mass_enclosed_profile,
        "enclosed_overdensity": fitted_enclosed_overdensity_profile,
        "fit_params": fit_params,
    }


def get_mean_enclosed_mass_profile(
    mass_enclosed_profile: np.ndarray,
    rho_mean: float | None = None,
    dimensionalized: bool = True,
) -> float:
    if rho_mean is None:
        raise ValueError("rho_mean must be provided to calculate mean mass")

    radii = mass_enclosed_profile[0]
    mean_volume = radii**3 * (1 if dimensionalized else 4 / 3 * np.pi)
    mean_mass = rho_mean * mean_volume

    return np.array([radii, mean_mass])


def mean_mass_difference(
    plot_mean_mass_difference: bool,
    mass_enclosed_profile: np.ndarray,
    rho_mean: float | None = None,
    dimensionalized: bool = True,
) -> np.ndarray | None:
    if not plot_mean_mass_difference:
        return None

    mean_mass_profile = get_mean_enclosed_mass_profile(
        mass_enclosed_profile, rho_mean, dimensionalized
    )

    return np.array(
        [mean_mass_profile[0], mass_enclosed_profile[1] - mean_mass_profile[1]]
    )


def plot_fitted_profile_and_residuals(
    actual_profile: np.ndarray,
    fitted_profile: np.ndarray,
    ax_main: Axes,
    ax_res: Axes,
    label: str,
    marker: str,
    x_label: str,
    fit_color: str = "black",
    show_params: bool = True,
    fit_params: dict[str, float] | None = None,
) -> None:


    # Plot the fitted profile on the main axis
    ax_main.semilogx(
        actual_profile[0], 
        fitted_profile, 
        linestyle="--",
        color=fit_color,
        alpha=0.8,
        zorder=1,
        label=f"{label} Fit"
    )

    # Calculate residuals
    residuals = (fitted_profile / actual_profile[1]) - 1.0

    # Plot residuals
    fit_params_str = (
        "".join([rf"{key} = {val:.3f} " for key, val in fit_params.items()])
        if fit_params is not None
        else ""
    )
    ax_res.semilogx(
        actual_profile[0], 
        residuals, marker, 
        label=fit_params_str
    )
    ax_res.axhline(
        0, color="gray", linestyle="--"
    )  # Add a horizontal line at y=0 for reference

    # Set labels and maybe some formatting
    ax_res.set_xlabel(x_label)
    ax_res.set_ylabel(r"Residuals")

    # Put legend in bottom right corner
    if show_params:
        ax_res.legend(
            loc="center",
            bbox_to_anchor=(0.5, -0.7),
        )


def plot_density_profile(
    profile: np.ndarray,
    fitted_profile: np.ndarray | None = None,
    rho_mean: float | None = None,
    ax: tuple[Axes, Axes] | None = None,
    label: str | None = None,
    marker: str = ".",
    radius_units: float = 1,
    density_units: float = 1,
    show_params: bool = True,
    fit_params: dict[str, float] | None = None,
    use_the_same_color: bool = False,
) -> None:
    input_ax = ax
    ax_res = None  # Initialize ax_res to None

    # Define labels for the main plot
    x_label, y_label = get_profile_labels(radius_units, "density", density_units)

    # Create subplots for the main plot and residuals if fitted_profile is not None
    if fitted_profile is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        # ax_res remains None

    elif ax is None:
        fig, axs = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.subplots_adjust(hspace=0)
        ax, ax_res = axs  # Unpack the axes
    else:
        # If ax is provided, it should be a tuple with two Axes
        ax, ax_res = ax
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Plot the provided density profile
    ax.loglog(*profile, marker, label=label)

    # Plot the fitted profile and residuals if provided
    if fitted_profile is not None:

        plot_fitted_profile_and_residuals(
            actual_profile=profile,
            fitted_profile=fitted_profile,
            ax_main=ax,
            ax_res=ax_res,
            label=label,
            marker=marker,
            x_label=x_label,
            show_params=show_params,
            fit_params=fit_params,
            fit_color=(
                plt.gca().lines[-1].get_color() 
                if use_the_same_color else "black" 
            )
        )

    if rho_mean is not None:
        ax.axhline(rho_mean, color="grey", linestyle="--")
        ax.text(
            1.0,
            1.5 * rho_mean,
            PROFILE_Y_LABELS["mean_density"][1 if (density_units == 1) else 2],
            color="grey",
            horizontalalignment="left",
        )

    # Show the plot if ax was not provided externally
    if input_ax is None:
        ax.legend()
        plt.show()


def plot_enclosed_overdensity_profile(
    profile: np.ndarray,
    fitted_profile: np.ndarray
    | None = None,  # Add fitted_profile as an optional argument
    ax: Axes | None = None,
    label: str | None = None,
    marker: str = ".",
    radius_units: float = 1,
    show_vertical_line: bool = True,
    show_horizontal_line: bool = True,
    show_params: bool = True,
    fit_params: dict[str, float] | None = None,
    use_the_same_color: bool = False,
) -> None:
    input_ax = ax
    ax_res = None  # Initialize ax_res to None

    # Create subplots for the main plot and residuals if fitted_profile is not None
    if fitted_profile is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        # ax_res remains None

    elif ax is None:
        fig, axs = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.subplots_adjust(hspace=0)
        ax, ax_res = axs  # Unpack the axes
    else:
        # If ax is provided, it should be a tuple with two Axes
        ax, ax_res = ax
    # Set x-axis label
    x_label = get_profile_xlabel(radius_units)
    y_label = PROFILE_Y_LABELS["enclosed_overdensity"][1]
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Plot the enclosed overdensity profile
    goes_negative = np.where(profile[1] < 0)[0]
    if len(goes_negative) > 0:
        # Split the plot at the point where the value goes negative
        plot_color = ax._get_lines.get_next_color()
        ax.semilogx(
            *profile[:, : goes_negative[0]], marker, color=plot_color, label=label
        )
        ax.semilogx(*profile[:, goes_negative[0] :], "--", color=plot_color)
        ax.set_yscale("symlog")

        # Plot the vertical line where the profile goes negative
        if show_vertical_line:
            ax.axvline(profile[0][goes_negative[0]], color="k", linestyle="--")
    else:
        ax.loglog(*profile, marker, label=label)

    # Plot the horizontal line at zero
    has_negative_vals: bool = np.any(profile[1] < 0)
    if show_horizontal_line and has_negative_vals:
        ax.axhline(0, color="k", linestyle="--")
        min_finite_non_zero_val = np.min(profile[1][profile[1] > 0])
        ax.text(
            10 ** (1 + int(np.log10(min_finite_non_zero_val))),
            0.1,
            r"$\delta = 0$",
            color="k",
            horizontalalignment="left",
        )

    # Plot the fitted profile and residuals if provided
    if fitted_profile is not None:
        plot_fitted_profile_and_residuals(
            actual_profile=profile,
            fitted_profile=fitted_profile,
            ax_main=ax,
            ax_res=ax_res,
            label=label,
            marker=marker,
            x_label=x_label,
            show_params=show_params,
            fit_params=fit_params,
            fit_color=(
                plt.gca().lines[-1].get_color() 
                if use_the_same_color else "black" 
            )
        )

    # Show the plot if ax was not provided externally
    if input_ax is None:
        ax.legend()
        plt.show()


def plot_scaled_density_profile(
    profile: np.ndarray,
    fitted_profile: np.ndarray | None = None,
    ax: tuple[Axes, Axes] | None = None,
    label: str | None = None,
    marker: str = ".",
    radius_units: float = 1,
    scaled_density_units: float = 1,
    show_params: bool = True,
    fit_params: dict[str, float] | None = None,
    use_the_same_color: bool = False,
) -> None:
    input_ax = ax
    ax_res = None  # Initialize ax_res to None

    # Set labels for the main plot
    x_label, y_label = get_profile_labels(
        radius_units, "scaled_density", scaled_density_units
    )

    # Create subplots with 2 rows if fitted_profile is not None
    if fitted_profile is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        # ax_res remains None

    elif ax is None:
        fig, axs = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.subplots_adjust(hspace=0)
        ax, ax_res = axs  # Unpack the axes
    else:
        # If ax is provided, it should be a tuple with two Axes
        ax, ax_res = ax
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Plot the provided scaled density profile
    ax.loglog(*profile, marker, label=label)

    # Plot the fitted profile and residuals if provided
    if fitted_profile is not None:
        plot_fitted_profile_and_residuals(
            actual_profile=profile,
            fitted_profile=fitted_profile,
            ax_main=ax,
            ax_res=ax_res,
            label=label,
            marker=marker,
            x_label=x_label,
            show_params=show_params,
            fit_params=fit_params,
            fit_color=(
                plt.gca().lines[-1].get_color() 
                if use_the_same_color else "black" 
            )
        )

    # Show the plot if ax was not provided externally
    if input_ax is None:
        ax.legend()
        plt.show()


def plot_mass_enclosed_profile(
    profile: np.ndarray,
    fitted_profile: np.ndarray | None = None,
    ax: tuple[Axes, Axes] | None = None,
    label: str | None = None,
    marker: str = ".",
    mass_units: float = 1,
    radius_units: float = 1,
    show200c: bool = True,
    show_params: bool = True,
    mean_enclosed_mass_profile: np.ndarray | None = None,
    mass_difference: np.ndarray | None = None,  # Difference b/t M(<r) and M_mean(<r)
    fit_params: dict[str, float] | None = None,
    use_the_same_color: bool = False,
) -> None:
    input_ax = ax
    ax_res = None  # Initialize ax_res to None

    # Set up labels based on units
    x_label, y_label = get_profile_labels(radius_units, "mass_enclosed", mass_units)

    # Create subplots with 2 rows if fitted_profile is not None
    if fitted_profile is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        # ax_res remains None

    elif ax is None:
        fig, axs = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.subplots_adjust(hspace=0)
        ax, ax_res = axs  # Unpack the axes
    else:
        # If ax is provided, it should be a tuple with two Axes
        ax, ax_res = ax
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Plot the provided mass enclosed profile
    ax.loglog(*profile, marker, label=label)

    # Plot the fitted profile and residuals if provided
    if fitted_profile is not None:
        plot_fitted_profile_and_residuals(
            actual_profile=profile,
            fitted_profile=fitted_profile,
            ax_main=ax,
            ax_res=ax_res,
            label=label,
            marker=marker,
            x_label=x_label,
            show_params=show_params,
            fit_params=fit_params,
            fit_color=(
                plt.gca().lines[-1].get_color() 
                if use_the_same_color else "black" 
            )
        )

    if mean_enclosed_mass_profile is not None:
        ax.loglog(
            *mean_enclosed_mass_profile, "--", color="black", label=r"$M_{m}(<r)$"
        )

    if mass_difference is not None:
        ax.loglog(*mass_difference, "-.", color="grey", label=r"$M(<r) - M_{m}(<r)$")

    # Plot R200c and M200c lines if required
    if show200c:
        ax.axvline(radius_units, color="k", linestyle="--")
        ax.axhline(mass_units, color="k", linestyle="--")
        ax.text(
            1.1 * radius_units,
            0.1 * mass_units,
            r"$R_{200c}$",
            color="k",
            verticalalignment="bottom",
        )
        ax.text(
            0.1 * radius_units,
            1.2 * mass_units,
            r"$M_{200c}$",
            color="k",
            horizontalalignment="left",
        )

    # Show the plot if ax was not provided externally
    if input_ax is None:
        ax.legend()
        plt.show()


def plot_mass_radial_profiles(
    density_profile: np.ndarray,
    scaled_density_profile: np.ndarray,
    mass_enclosed_profile: np.ndarray,
    enclosed_overdensity_profile: np.ndarray,
    profile_fit: AsymptoticProfile | None = None,
    rho_mean: float | None = None,
    fig: Figure | None = None,
    ax: list[Axes] | None = None,
    label: str | None = None,
    marker: str = ".",
    mass_units: float = 1,
    radius_units: float = 1,
    density_units: float = 1,
    scaled_density_units: float = 1,
    show200c: bool = True,
    show_overdensity_vertical_line: bool = True,
    show_overdensity_horizontal_line: bool = True,
    plot_all: bool = False,
    show_params_on_each: bool = True,
    plot_mean_mass_difference: bool = False,
    return_plot: bool = False,
    use_the_same_color: bool = False,
) -> tuple[Figure, list[Axes]] | None:
    # Check if fitted profiles are provided and update accordingly
    has_fitted_profiles = profile_fit is not None
    fitted_profiles = get_fitted_profiles(
        density_profile=density_profile,
        scaled_density_profile=scaled_density_profile,
        mass_enclosed_profile=mass_enclosed_profile,
        enclosed_overdensity_profile=enclosed_overdensity_profile,
        rho_mean=rho_mean,
        mass_units=mass_units,
        density_units=density_units,
        scaled_density_units=scaled_density_units,
        profile_fit=profile_fit,
    )

    mass_difference = mean_mass_difference(
        plot_mean_mass_difference,
        mass_enclosed_profile,
        rho_mean=rho_mean,
        # profile_fit=profile_fit,
        dimensionalized=(mass_units == 1),
    )

    if ax is None:
        if has_fitted_profiles:
            # Create a 4x2 grid of plots, where each profile gets a main plot and a residual plot
            fig, axs = plt.subplots(
                4, 2, figsize=(15, 15), gridspec_kw={"height_ratios": [3, 1, 3, 1]}
            )
            fig.subplots_adjust(hspace=0)  # Remove vertical space between plots

        else:
            # If no fitted profiles are to be plotted, create a 2x2 grid for the main plots only
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        # Flatten the grid into a 1D array for easier access
        axs = axs.flatten()
    else:
        if has_fitted_profiles and len(ax) != 8:
            raise ValueError(
                "When providing axes for fitted profiles, eight axes instances are expected."
            )
        elif not has_fitted_profiles and len(ax) != 4:
            raise ValueError(
                "When providing axes without fitted profiles, four axes instances are expected."
            )
        axs = ax

    # The plots now expect two Axes each: one for the main plot, one for the residuals
    plot_density_profile(
        density_profile,
        fitted_profile=fitted_profiles["density"],
        rho_mean=rho_mean,
        ax=(axs[0], axs[2]) if has_fitted_profiles else axs[0],
        label=label,
        marker=marker,
        radius_units=radius_units,
        density_units=density_units,
        show_params=(show_params_on_each and has_fitted_profiles),
        fit_params=fitted_profiles["fit_params"],
        use_the_same_color=use_the_same_color,
    )
    plot_scaled_density_profile(
        scaled_density_profile,
        fitted_profile=fitted_profiles["scaled_density"],
        ax=(axs[1], axs[3]) if has_fitted_profiles else axs[1],
        label=label,
        marker=marker,
        radius_units=radius_units,
        scaled_density_units=scaled_density_units,
        show_params=(show_params_on_each and has_fitted_profiles),
        fit_params=fitted_profiles["fit_params"],
    )

    if plot_all:
        plot_mass_enclosed_profile(
            mass_enclosed_profile,
            fitted_profile=fitted_profiles["mass_enclosed"],
            ax=(axs[4], axs[6]) if has_fitted_profiles else axs[2],
            label=label,
            marker=marker,
            mass_units=mass_units,
            radius_units=radius_units,
            show200c=show200c,
            # mean_enclosed_mass_profile = get_mean_enclosed_mass_profile(
            #     mass_enclosed_profile,
            #     rho_mean=rho_mean,
            #     dimensionalized=(mass_units == 1)
            # ),
            mass_difference=mass_difference,
            show_params=(show_params_on_each and has_fitted_profiles),
            fit_params=fitted_profiles["fit_params"],
            use_the_same_color=use_the_same_color,
        )

        plot_enclosed_overdensity_profile(
            enclosed_overdensity_profile,
            fitted_profile=fitted_profiles["enclosed_overdensity"],
            ax=(axs[5], axs[7]) if has_fitted_profiles else axs[3],
            label=label,
            marker=marker,
            radius_units=radius_units,
            show_horizontal_line=show_overdensity_horizontal_line,
            show_vertical_line=show_overdensity_vertical_line,
            show_params=(show_params_on_each and has_fitted_profiles),
            fit_params=fitted_profiles["fit_params"],
            use_the_same_color=use_the_same_color,
        )

    # If fitted profiles were provided, each plotting function should handle the residuals
    if has_fitted_profiles:
        # Adjust the layout of the main plots and residual plots
        for i, ax in enumerate(axs):
            if i % 2 == 0:  # Only the main plots (not the residual plots)
                pdb.set_trace()
                ax.get_shared_x_axes().join(ax, axs[i + 1]) # --> problem here?
                ax.set_xticklabels([])  # Hide x tick labels on the main plot
                axs[i + 1].set_xlabel(
                    ax.get_xlabel()
                )  # Move the xlabel to the residual plot
            if i < 4:
                # Remove the xlabel
                ax.set_xlabel("")  # Remove xlabel from the main plot
                ax.set_xticklabels([])  # Remove x tick labels from the main plot

    if return_plot:
        return fig, axs

    # If ax was not provided, show the plot
    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_radial_velocity_distribution(
    radii: np.ndarray,
    velocity: np.ndarray,
    radial_bins: np.ndarray,
    velocity_medians: np.ndarray,
    velocity_25pct: np.ndarray,
    velocity_75pct: np.ndarray,
    density_label: str | None = None,
    density: np.ndarray | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    label: str | None = None,
    marker: str = ".",
    x_scale: bool = False,
    y_scale: bool = False,
    radius_units: float = 1,
    velocity_units: float = 1,
    density_units: float = 1,
) -> AxesImage | None:
    input_ax = ax
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    plot_bins = np.sqrt(radial_bins[1:] * radial_bins[:-1])

    # pdb.set_trace()

    # ax.plot(plot_bins, velocity_medians, "r-", label="Median", zorder=2)
    ax.plot(radial_bins, velocity_medians, "r-", label="Median", zorder=2)
    ax.fill_between(
        radial_bins,
        velocity_25pct,
        velocity_75pct,
        color="r",
        alpha=0.3,
        label="25th-75th Percentile",
        zorder=1,
    )

    if x_scale:
        ax.set_xscale("log")
    if y_scale:
        ax.set_yscale("symlog")

    if radius_units == 1:
        ax.set_xlabel(r"Radial Distance, $\frac{r}{R_{200c}}$")
    else:
        ax.set_xlabel(r"Radial Distance, $r$ $\left[h^{-1} {\rm Mpc}\right]$")

    if velocity_units == 1:
        ax.set_ylabel(r"Radial Velocity, $\frac{v_r}{v_{200c}}$")
    else:
        ax.set_ylabel(r"Radial Velocity, $v_r$ $\left[{\rm km/s}\right]$")
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if density is None:
        ax.plot(
            radii, velocity, marker, alpha=0.75, markersize=1.0, label=label, zorder=0
        )
    else:
        RADII, VELOCITIES = np.meshgrid(radii, velocity)
        plot_density = density.T
        non_zero_values = np.nonzero(plot_density)
        min_density = np.min(plot_density[non_zero_values])
        plot_density[plot_density == 0.0] += min_density / 1000
        im = ax.pcolormesh(
            RADII, VELOCITIES, plot_density, norm=LogNorm(), cmap="viridis", zorder=0
        )

        if input_ax is not None:
            return im

        fig.colorbar(im, ax=ax, label=density_label)



    ax.set_xlim(left=0.1)

    ax.legend()

    return None


def plot_internal_structure(
        distribution_data: dict[str, np.ndarray | dict[str, np.ndarray]],
        boundaries: dict[str, float],
        scale_factor: float,
        v_norm: float = 1.0,
        use_density_distribution: bool = False,
        text_size: int = 14,
        tick_size: int = 14,
        legend_size: int = 14,
        label_size: int = 14,
        fig: Figure | None = None,
        axes: np.ndarray | None = None,
        return_plot: bool = False,
        wrt_mass: bool = False,
        phase_space_ylims: tuple[float, float] | None = None,
        kappa_ylims: tuple[float, float] | None = None,
        use_radii_slope: bool = False,
    ) -> tuple[Figure, np.ndarray, list[plt.Line2D]] | None: 

    if fig is None or axes is None:

        fig, axes = plt.subplots(
            2, 1, figsize=(6, 4),
            sharex=True, 
            gridspec_kw={'hspace': 0.0}
        )

    interpolator = get_enclosed_masses_interpolator(
        distribution_data["radii"],
        distribution_data["enclosed_mass"],
    )

    if use_density_distribution:
        phase_space_dist = distribution_data["phase_space"]
        x_bins = (
            interpolate_radii_to_mass(phase_space_dist["radii"], interpolator)
            if wrt_mass else 
            phase_space_dist["radii"]
        )
        pcm = axes[0].pcolormesh(
            x_bins,
            phase_space_dist["velocities"] / v_norm,
            phase_space_dist["density"],
            shading="auto",
            cmap="blues",
        )

        cbar = fig.colorbar(
            pcm, ax=axes[0], 
            orientation='horizontal', 
            location='top',
            pad=0.0
        )
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.set_label(
            r"$p(r, v_{r})$", 
            fontsize=label_size, 
            labelpad=10
        )
    else:
        
        x_vals = (
            interpolate_radii_to_mass(distribution_data["phase_space"][0], interpolator)
            if wrt_mass else 
            distribution_data["phase_space"][0]
        )


        axes[0].scatter(
            x_vals, 
            distribution_data["phase_space"][1] / v_norm,
            s=0.1, 
            color="tab:blue",
        )

    axes[0].text(
        0.05, 0.95, rf"$a = {scale_factor:.2f}$", 
        transform=axes[0].transAxes,
        fontsize=label_size,
        va='top', ha='left'
    )

    axes[0].set_xscale("log")

    x_vals = (
        distribution_data["enclosed_mass"]
        if wrt_mass else 
        distribution_data["radii"]
    )

    y_vals = (
        distribution_data["radii_slope"]
        if use_radii_slope else
        distribution_data["kappa"]
    )

    axes[1].semilogx(x_vals, y_vals)
    axes[1].set_yscale("symlog")

    plt_handles = []

    for boundary_mdef, radius in boundaries.items():

        mapping = global_mass_def_mapping.use_key(boundary_mdef)
        color = MASS_DEF_COLORS[mapping.name]

        if wrt_mass:
            mass = interpolate_radii_to_mass(radius, interpolator)
            axes[0].axvline(mass, color=color, linestyle="--")
            axes[1].axvline(mass, color=color, linestyle="--")
        else:
            axes[0].axvline(radius, color=color, linestyle="--")
            axes[1].axvline(radius, color=color, linestyle="--")

        plt_handles.append(
            plt.Line2D(
                [0], [0], 
                color=color, 
                linestyle="--", 
                label=mapping.radius.eqn
            )
        )

    if return_plot:
        return fig, axes, plt_handles

    if np.isclose(v_norm, 1.0):
        axes[0].set_ylabel(r"$v_{r}/v_{\rm max}$", fontsize=label_size)
    else:
        axes[0].set_ylabel(r"$v_{r}$ $[{\rm km}/{\rm s}]$", fontsize=label_size)

    if phase_space_ylims is not None:
        axes[0].set_ylim(phase_space_ylims)

    if kappa_ylims is not None:
        axes[1].set_ylim(kappa_ylims)

    if use_radii_slope:
        axes[1].set_ylabel(r"$\frac{d \log R}{d \log M}$", fontsize=label_size)
    else:
        axes[1].set_ylabel(r"$\kappa(<r)$", fontsize=label_size)
        axes[1].set_ylim(top=0.5)

    fig.legend(
        handles=plt_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(boundaries),
        fontsize=legend_size,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.15)


def get_enclosed_masses_interpolator(
        radii: np.ndarray,
        enclosed_masses: np.ndarray,
    ) -> interp1d:
    
    return interp1d(
        np.log10(radii),
        np.log10(enclosed_masses),
        fill_value="extrapolate",
    )

def interpolate_radii_to_mass(
        radii: float | np.ndarray, 
        interpolator: interp1d
    ) -> float | np.ndarray:
    return 10.0**interpolator(np.log10(radii))

