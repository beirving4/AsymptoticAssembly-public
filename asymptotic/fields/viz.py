import numpy as np
import matplotlib.pyplot as plt

RADII_LABELS = {
    "comoving" : r"${\rm Radii}$, $r$ [$h^{-1}$ ${\rm cMpc}$]",
    "physical" : r"${\rm Radii}$, $r$ [$h^{-1}$ ${\rm Mpc}$]",
}
WAVENUMBER_LABELS = {
    "comoving" : r"{\rm Wavenumbers}, $k$ [$h$ ${\rm cMpc}^{-1}$]",
    "physical" : r"{\rm Wavenumbers}, $k$ [$h$ ${\rm Mpc}^{-1}$]",
}

PS_LABELS = {
    "comoving" : {
            "linear" : {
            "amplitude" : r"$P_{\rm lin}(k)$ [$h^{-3}$ ${\rm cMpc}^3$]",
            "normalized" : r"$\Delta^2_{\rm lin}(k)$"
        },
        "nonlinear" : {
            "amplitude" : r"$P(k)$ [$h^{-3}$ ${\rm cMpc}^3$]",
            "normalized" : r"$\Delta^2(k)$",
        }
    },
    "physical" : {
            "linear" : {
            "amplitude" : r"$P_{\rm lin}(k)$ [$h^{-3}$ ${\rm Mpc}^3$]",
            "normalized" : r"$\Delta^2_{\rm lin}(k)$"
        },
        "nonlinear" : {
            "amplitude" : r"$P(k)$ [$h^{-3}$ ${\rm Mpc}^3$]",
            "normalized" : r"$\Delta^2(k)$",
        }
    }
}

CORRELATION_LABELS = {
    "comoving" : {
        "linear" : {
                "scaled" : r"$r^{2}\xi_{\rm lin}(r)$ [$h$ ${\rm cMpc}^{-1}$]",
                "unscaled" : r"$\xi_{\rm lin}(r)$ [$h^{3}$ ${\rm cMpc}^{-3}$]"
        },
        "nonlinear" : {
                "scaled" : r"$r^{2}\xi(r)$ [$h$ ${\rm cMpc}^{-1}$]",
                "unscaled" : r"$\xi(r)$ [$h^{3}$ ${\rm cMpc}^{-3}$]"
        },
    },
    "physical" : {
        "linear" : {
                "scaled" : r"$r^{2}\xi_{\rm lin}(r)$ [$h$ ${\rm Mpc}^{-1}$]",
                "unscaled" : r"$\xi_{\rm lin}(r)$ [$h^{3}$ ${\rm Mpc}^{-3}$]"
        },
        "nonlinear" : {
                "scaled" : r"$r^{2}\xi(r)$ [$h$ ${\rm Mpc}^{-1}$]",
                "unscaled" : r"$\xi(r)$ [$h^{3}$ ${\rm Mpc}^{-3}$]"
        },
    }
}

def get_ps_label(is_linear: bool, is_normalized: bool, in_comoving: bool) -> str:
    units_key = "comoving" if in_comoving else "physical"
    lin_key = "linear" if is_linear else "nonlinear"
    norm_key = "normalized" if is_normalized else "amplitude"
    return PS_LABELS[units_key][lin_key][norm_key]

def get_correlation_label(is_linear: bool, is_scaled: bool, in_comoving: bool) -> str:
    units_key = "comoving" if in_comoving else "physical"
    lin_key = "linear" if is_linear else "nonlinear"
    scale_key = "scaled" if is_scaled else "unscaled"
    return CORRELATION_LABELS[units_key][lin_key][scale_key]

def get_default_color(ax: plt.Axes) -> str | tuple:
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx = len(ax.get_lines()) % len(colors)
    return colors[color_idx]


def display_power_spectrum(
        wavenumbers: np.ndarray, 
        amplitudes: np.ndarray, 
        is_normalized: bool = True,
        is_linear: bool = True, 
        text_size: int = 16,
        ax: plt.Axes | None = None,
        return_fig: bool = False,
        color: str | tuple | None = None, 
        in_comoving: bool = True
    ) -> plt.Axes | None:

    if ax is None:
        _, ax = plt.subplots()

    color = get_default_color(ax) if color is None else color
    
    ax.loglog(wavenumbers, amplitudes, color=color)

    x_label = WAVENUMBER_LABELS["comoving" if in_comoving else "physical"]
    y_label = get_ps_label(is_linear, is_normalized, in_comoving)

    ax.set_xlabel(x_label, fontsize=text_size)
    ax.set_ylabel(y_label, fontsize=text_size)

    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.yaxis.set_tick_params(labelsize=text_size)

    if return_fig:
        return ax
    
    plt.show()


def display_correlation(
        radii: np.ndarray,
        correlations: np.ndarray,
        is_scaled: bool = True,
        is_linear: bool = True,
        text_size: int = 16,
        ax: plt.Axes | None = None,
        return_fig: bool = False,
        color: str | tuple | None = None,
        in_comoving: bool = True
    ) -> plt.Axes | None:

    if ax is None:
        _, ax = plt.subplots()

    color = get_default_color(ax) if color is None else color

    ax.loglog(radii, correlations, color=color)

    x_label = RADII_LABELS["comoving" if in_comoving else "physical"]
    y_label = get_correlation_label(is_linear, is_scaled, in_comoving)

    ax.set_xlabel(x_label, fontsize=text_size)
    ax.set_ylabel(y_label, fontsize=text_size)

    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.yaxis.set_tick_params(labelsize=text_size)

    if return_fig:
        return ax
    
    plt.show()

def display_clustering(
        wavenumbers: np.ndarray,
        amplitudes: np.ndarray,
        radii: np.ndarray,
        correlations: np.ndarray,
        is_scaled: bool = True,
        is_linear: bool = True,
        is_normalized: bool = True,
        text_size: int = 16,
        ax_ps: plt.Axes | None = None,
        ax_corr: plt.Axes | None = None,
        return_fig: bool = False,
        wspace: float = 0.3, 
        color: str | tuple | None = None,
    ) -> tuple[plt.Axes | None, plt.Axes | None] | None:

    if (ax_ps is None) or (ax_corr is None):
        _, (ax_ps, ax_corr) = plt.subplots(
            1, 2, figsize=(12, 6), gridspec_kw={"wspace" : wspace}
        )

    color = get_default_color(ax_ps) if color is None else color

    ax_ps = display_power_spectrum(
        wavenumbers=wavenumbers,
        amplitudes=amplitudes,
        is_normalized=is_normalized,
        is_linear=is_linear,
        text_size=text_size,
        ax=ax_ps,
        return_fig=True,
        color=color
    )
    ax_corr = display_correlation(
        radii=radii,
        correlations=correlations,
        is_scaled=is_scaled,
        is_linear=is_linear,
        text_size=text_size,
        ax=ax_corr,
        return_fig=True,
        color=color
    )

    if return_fig:
        return ax_ps, ax_corr

    plt.show()

def display_matter_power_spectrum(
        linear_wavenumbers: np.ndarray | None, 
        linear_amplitudes: np.ndarray | None, 
        nonlinear_wavenumbers: np.ndarray | None, 
        nonlinear_amplitudes: np.ndarray | None, 
        shot_noise_wavenumbers: np.ndarray | None,
        shot_noise_amplitudes: np.ndarray | None,
        is_normalized: bool = True,
        text_size: int = 16,
        ax: plt.Axes | None = None,
        return_fig: bool = False,
        color: str | tuple | None = None,
        show_legend: bool = True, 
        in_comoving: bool = True
    ) -> plt.Axes | None:

    if ax is None:
        _, ax = plt.subplots() 

    color = get_default_color(ax) if color is None else color

    if nonlinear_wavenumbers is not None:
        ax.loglog(
            nonlinear_wavenumbers, 
            nonlinear_amplitudes, 
            color=color,
            label="Nonlinear"
        )
    
    if linear_wavenumbers is not None:
        non_nolinear = nonlinear_wavenumbers is None
        no_shot_noise = shot_noise_wavenumbers is None
        only_linear = (non_nolinear and no_shot_noise)
        ax.loglog(
            linear_wavenumbers, 
            linear_amplitudes, 
            '--',
            color=color,
            alpha = (1.0 if only_linear else 0.5),
            label="Linear"
        )

    

    if shot_noise_wavenumbers is not None:
        ax.loglog(
            shot_noise_wavenumbers, 
            shot_noise_amplitudes, 
            '-.', 
            label="Shot Noise",
            lw=2, color="grey", alpha=0.75
        )

    x_label = WAVENUMBER_LABELS["comoving" if in_comoving else "physical"]
    y_label = get_ps_label(False, is_normalized, in_comoving)


    ax.set_xlabel(x_label, fontsize=text_size)
    ax.set_ylabel(y_label, fontsize=text_size)

    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.yaxis.set_tick_params(labelsize=text_size)

    if show_legend:
        ax.legend(fontsize=text_size)

    if return_fig:
        return ax
    
    plt.show()


def display_matter_correlation(
        linear_radii: np.ndarray | None, 
        linear_correlations: np.ndarray | None, 
        nonlinear_radii: np.ndarray | None, 
        nonlinear_correlations: np.ndarray | None, 
        is_scaled: bool = True,
        text_size: int = 16,
        ax: plt.Axes | None = None,
        return_fig: bool = False,
        color: str | tuple | None = None,
        show_legend: bool = True,
        in_comoving: bool = True
    ) -> plt.Axes | None:

    if ax is None:
        _, ax = plt.subplots()

    color = get_default_color(ax) if color is None else color

    if nonlinear_radii is not None:
        ax.loglog(
            nonlinear_radii, 
            nonlinear_correlations, 
            '-',
            color=color,
            label="Nonlinear"
        )
    
    if linear_radii is not None:
        ax.loglog(
            linear_radii, 
            linear_correlations, 
            '--',
            color=color,
            alpha = (1.0 if (nonlinear_radii is None) else 0.5),
            label="Linear"
        )

    x_label = RADII_LABELS["comoving" if in_comoving else "physical"]
    y_label = get_correlation_label(False, is_scaled, in_comoving)

    ax.set_xlabel(x_label, fontsize=text_size)
    ax.set_ylabel(y_label, fontsize=text_size)

    ax.set_xlabel(x_label, fontsize=text_size)
    ax.set_ylabel(y_label, fontsize=text_size)

    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.yaxis.set_tick_params(labelsize=text_size)

    if show_legend:
        ax.legend(fontsize=text_size)

    if return_fig:
        return ax
    
    plt.show()


def display_matter_clustering(
        linear_wavenumbers: np.ndarray | None,
        linear_amplitudes: np.ndarray | None,
        nonlinear_wavenumbers: np.ndarray | None,
        nonlinear_amplitudes: np.ndarray | None,
        shot_noise_wavenumbers: np.ndarray | None,
        shot_noise_amplitudes: np.ndarray | None,
        linear_radii: np.ndarray | None,
        linear_correlations: np.ndarray | None,
        nonlinear_radii: np.ndarray | None,
        nonlinear_correlations: np.ndarray | None,
        is_scaled: bool = True,
        is_linear: bool = True,
        is_normalized: bool = True,
        text_size: int = 16,
        ax_ps: plt.Axes | None = None,
        ax_corr: plt.Axes | None = None,
        return_fig: bool = False,
        wspace: float = 0.3, 
        color: str | tuple | None = None,
        show_legend: bool = True,
        in_comoving: bool = True
    ) -> tuple[plt.Axes | None, plt.Axes | None] | None:

    if (ax_ps is None) or (ax_corr is None):
        _, (ax_ps, ax_corr) = plt.subplots(
            1, 2, figsize=(12, 6), gridspec_kw={"wspace" : wspace}
        )

    color = get_default_color(ax_ps) if color is None else color

    ax_ps = display_matter_power_spectrum(
        linear_wavenumbers=linear_wavenumbers,
        linear_amplitudes=linear_amplitudes,
        nonlinear_wavenumbers=nonlinear_wavenumbers,
        nonlinear_amplitudes=nonlinear_amplitudes,
        shot_noise_wavenumbers=shot_noise_wavenumbers,
        shot_noise_amplitudes=shot_noise_amplitudes,
        is_normalized=is_normalized,
        text_size=text_size,
        ax=ax_ps,
        return_fig=True,
        color=color,
        show_legend=show_legend,
        in_comoving=in_comoving
    )

    ax_corr = display_matter_correlation(
        linear_radii=linear_radii,
        linear_correlations=linear_correlations,
        nonlinear_radii=nonlinear_radii,
        nonlinear_correlations=nonlinear_correlations,
        is_scaled=is_scaled,
        text_size=text_size,
        ax=ax_corr,
        return_fig=True,
        color=color,
        show_legend=show_legend,
        in_comoving=in_comoving
    )

    if return_fig:
        return ax_ps, ax_corr
    
    plt.show()