import numpy as np, pdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..mass_def.base import global_mass_def_mapping


SCALE_FACTOR_LABEL = r"${\rm Scale}$ ${\rm Factor}$, $a$"
RESIDUAL_LABEL = r"${\rm Residuals}$"

MF_PLOT_LABELS = {
    "number_density": r"$n(M, a)$ $[h^4 \, M_{\odot}{\rm cMpc}^{-3}]$",
    "differential": r"$\phi$ $[h^3 \, \mathrm{cMpc}^{-3}]$",
    "normalized": r"$\frac{M^2}{\rho_{{\rm m},0}} n(M, a)$",
    "cumulative": r"$n(>M, a)$ $[h^3 \, \mathrm{cMpc}^{-3}]$",
    "multiplicity": r"$f(\nu, a)$",
}

NORMALIZED_LABELS = {
    "number_density" : r"$\frac{n(M, a)}{n_{\rm s}(M)}$",
    "differential" : r"$\frac{\phi(M, a)}{\phi_{\rm s}(M)}$",
    "normalized" : r"$\frac{n(M, a)}{n_{\rm s}(M)}$",
    "cumulative" : r"$\frac{n(>M, a)}{n_{\rm s}( > M)}$",
    "multiplicity" : r"$\frac{f(\nu, a)}{f_{\rm s}(\nu)}$"
}


RATE_PLOT_LABELS = {
    "number_density" : r"$\frac{{\rm d} \ln{n}}{{\rm d} \ln{a}}$",
    "differential" : r"$\frac{{\rm d} \ln{\phi}}{{\rm d} \ln{a}}$",
    "normalized" : r"$\frac{{\rm d} \ln{n}}{{\rm d} \ln{a}}$",
    "cumulative" : r"$\frac{{\rm d} \ln{n(>M)}}{{\rm d} \ln{a}}$",
    "multiplicity" : r"$\frac{{\rm d} \ln{f(\nu)}}{{\rm d} \ln{a}}$"
}

def get_y_label(mf_type: str, normalize_by_final: bool = False) -> str:
    label_dict = NORMALIZED_LABELS if normalize_by_final else MF_PLOT_LABELS

    if (ylabel := label_dict.get(mf_type)) is None:
        raise ValueError("Invalid mass function type")
    
    return ylabel


def get_mass_xlabel(mass_def_key: str) -> str:
    eqn = global_mass_def_mapping.use_key(mass_def_key).mass.eqn
    return rf"{eqn} $[h^{{-1}} \, M_{{\odot}}]$"

def get_peak_xlabel(mass_def_key: str) -> str:
    return rf"{global_mass_def_mapping.use_key(mass_def_key).peak_height.eqn}"

# Need to get a method that maps the mass_def_key to the mass_def_eqn for the labeling 


# When time permits, refactor the code to consolidate display_mass_function and 
# and display_halo_multiplicity into a single function that can handle both cases
def display_mass_function(
        mass_def_key: str, 
        masses: np.ndarray,
        estimates: np.ndarray,
        errors: np.ndarray | None = None,
        mf_type: str = "differential",
        ax_main: plt.Axes | None = None,
        return_fig: bool = False,
        show_log_x: bool = True,
        show_log_y: bool = True, 
        plot_linestyle: tuple | None = (0, ()),
        marker_style: str | None = '.',
        color: tuple[float, float, float, float] | str | None = None,
        x_label_text_size: int = 10,
        y_label_text_size: int = 10,
        x_tick_text_size: int = 10,
        y_tick_text_size: int = 10
    ) -> plt.Axes | None:

    if ax_main is None:

        _, ax_main = plt.subplots()

    if color is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_idx = len(ax_main.get_lines()) % len(colors)
        color = colors[color_idx]

    xlabel = get_mass_xlabel(mass_def_key)
    ylabel = get_y_label(mf_type, normalize_by_final=False)

    if errors is None:

        if plot_linestyle is None:
            raise ValueError("Must provide a linestyle for plot")

        ax_main.plot(masses, estimates, linestyle=plot_linestyle, color=color)

    elif marker_style is None:
        raise ValueError("Must provide a marker style for errorbar plot")

    else:
        ax_main.errorbar(
            masses, estimates, yerr=errors, 
            fmt=marker_style, color=color
        )

    if show_log_x:
        ax_main.set_xscale("log")

    if show_log_y:
        ax_main.set_yscale("log")

    ax_main.xaxis.set_tick_params(labelsize=x_tick_text_size)
    ax_main.yaxis.set_tick_params(labelsize=y_tick_text_size)

    ax_main.set_xlabel(xlabel, fontsize=x_label_text_size)
    ax_main.set_ylabel(ylabel, fontsize=y_label_text_size)


    if return_fig:
        return ax_main

    plt.show()

def display_halo_multiplicity(
        mass_def_key: str,  # mass def equation, 
        peak_heights: np.ndarray,
        estimates: np.ndarray,
        errors: np.ndarray | None = None,
        ax_main: plt.Axes | None = None,
        return_fig: bool = False,
        show_log_x: bool = True,
        show_log_y: bool = True, 
        plot_linestyle: tuple | None = (0, ()),
        marker_style: str | None = '.',
        color: tuple[float, float, float, float] | str | None = None,
        x_label_text_size: int = 10,
        y_label_text_size: int = 10,
        x_tick_text_size: int = 10,
        y_tick_text_size: int = 10
    ) -> plt.Axes | None:

    if ax_main is None:

        _, ax_main = plt.subplots()

    if color is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_idx = len(ax_main.get_lines()) % len(colors)
        color = colors[color_idx]

    xlabel = get_peak_xlabel(mass_def_key)
    ylabel = get_y_label("multiplicity", normalize_by_final=False)

    if errors is None:

        if plot_linestyle is None:
            raise ValueError("Must provide a linestyle for plot")

        ax_main.plot(peak_heights, estimates, linestyle=plot_linestyle, color=color)

    elif marker_style is None:
        raise ValueError("Must provide a marker style for errorbar plot")

    else:
        ax_main.errorbar(
            peak_heights, estimates, yerr=errors, 
            fmt=marker_style, color=color
        )

    if show_log_x:
        ax_main.set_xscale("log", base=2)

    if show_log_y:
        ax_main.set_yscale("log", base=10)

    ax_main.xaxis.set_tick_params(labelsize=x_label_text_size)
    ax_main.yaxis.set_tick_params(labelsize=y_label_text_size)

    latex_formatter = lambda x, pos: rf"${x:.2f}$"
    ax_main.set_xlabel(xlabel, fontsize=x_tick_text_size)
    ax_main.xaxis.set_major_formatter(ticker.FuncFormatter(latex_formatter))
    ax_main.set_ylabel(ylabel, fontsize=y_tick_text_size)

    if return_fig:
        return ax_main

    plt.show()


# When time permits, refactor the code to consolidate display_mass_function_data and 
# and display_halo_multiplicity_fit into a single function that can handle both cases

def display_mass_function_data(
        mass_def_key: str,
        masses: np.ndarray,
        estimates: np.ndarray,
        theoretical: np.ndarray,
        errors: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
        mf_type: str = "differential",
        ax_main: plt.Axes | None = None,
        ax_resid: plt.Axes | None = None,
        legend_text_size: int = 10,
        x_label_text_size: int = 10,
        main_y_label_text_size: int = 10,
        resid_y_label_text_size: int = 10,
        x_tick_text_size: int = 10,
        main_y_tick_text_size: int = 10,
        resid_y_tick_text_size: int = 10,
        return_fig: bool = False,
        show_log_x: bool = True,
        show_log_y: bool = True,
        show_legend: bool = True,
        resid_min: float | None = None,
        resid_max: float | None = None,
        plot_linestyle: tuple | str | None = None,
        marker_style: str | None = None,
        color: tuple[float, float, float, float] | str | None = None
    ) -> tuple[plt.Axes | None, plt.Axes | None] | None:

    if ((ax_main is None) and (ax_resid is None)):

        _, (ax_main, ax_resid) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, sharey=False,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0}
        )

    if color is None:
        # Get the default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Get the current color cycle index
        color_idx = len(ax_main.get_lines()) % len(colors)
        data_color = colors[color_idx]
    else:
        data_color = color

    fit_color = "grey" if color is None else color

    ylabel = get_y_label(mf_type, normalize_by_final=False)


    if marker_style is None:
        marker_style = '.'

    
    if errors is not None:
        ax_main.errorbar(
            masses, estimates, yerr=errors, 
            fmt=marker_style, label="Data", color=data_color
        )
    else:
        ax_main.plot(masses, estimates, marker_style, label="Data", color=data_color)

    fit_linestyle = '--' if plot_linestyle is None else plot_linestyle
    resid_linestyle = '-' if plot_linestyle is None else plot_linestyle

    ax_main.plot(
        masses, theoretical, linestyle=fit_linestyle, 
        label="Asymptotic Fit", color=fit_color
    )

    if residuals is None:
        residuals = 10.0**(np.log10(estimates) - np.log10(theoretical)) - 1.0

    ax_resid.plot(
        masses, residuals, 
        linestyle=resid_linestyle, 
        color=data_color,
        zorder=1
    )
    ax_resid.axhline(0, color="k", linestyle="--", zorder=0)
    
    ax_main.set_ylabel(ylabel)

    ax_main.yaxis.set_tick_params(labelsize=main_y_tick_text_size)
    ax_resid.yaxis.set_tick_params(labelsize=resid_y_tick_text_size)
    ax_resid.xaxis.set_tick_params(labelsize=x_tick_text_size) 

    ax_resid.set_xlabel(get_mass_xlabel(mass_def_key), fontsize=x_label_text_size)
    ax_resid.set_ylabel(RESIDUAL_LABEL, fontsize=resid_y_label_text_size)
    
    ax_main.set_ylabel(ylabel, fontsize=main_y_label_text_size)

    ax_resid.set_ylim(bottom=resid_min, top=resid_max)

    if show_log_x:
        ax_main.set_xscale("log")
        ax_resid.set_xscale("log")

    if show_log_y:
        ax_main.set_yscale("log")

    if show_legend:
        ax_main.legend(fontsize=legend_text_size)

    if return_fig:
        return ax_main, ax_resid

    plt.show()




def display_halo_multiplicity_fit(
        mass_def_key: str,
        peak_heights: np.ndarray,
        estimates: np.ndarray, 
        theoretical: np.ndarray,
        mf_type: str = "differential",
        errors: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
        ax_main: plt.Axes | None = None,
        ax_resid: plt.Axes | None = None,
        legend_text_size: int = 10,
        x_label_text_size: int = 10,
        main_y_label_text_size: int = 10,
        resid_y_label_text_size: int = 10,
        x_tick_text_size: int = 10,
        main_y_tick_text_size: int = 10,
        resid_y_tick_text_size: int = 10,
        return_fig: bool = False,
        show_log_x: bool = True,
        show_log_y: bool = True,
        show_legend: bool = True,
        resid_min: float | None = None,
        resid_max: float | None = None,
        plot_linestyle: tuple | str | None = (0, ()),
        marker_style: str | None = '.',
        color: tuple[float, float, float, float] | str | None = None
    ) -> tuple[plt.Axes | None, plt.Axes | None] | None:

    if ((ax_main is None) and (ax_resid is None)):

        _, (ax_main, ax_resid) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, sharey=False,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0}
        )


    if color is None:
        # Get the default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Get the current color cycle index
        color_idx = len(ax_main.get_lines()) % len(colors)
        data_color = colors[color_idx]
    else:
        data_color = color

    ylabel = get_y_label(mf_type, normalize_by_final=False)

    fit_color = "grey" if color is None else color

    if marker_style is None:
        marker_style = '.'

    if errors is not None:
        ax_main.errorbar(
            peak_heights, estimates, yerr=errors, 
            fmt=marker_style, label="Data", color=data_color
        )
    else:
        ax_main.plot(
            peak_heights, estimates, 
            marker=marker_style, label="Data", color=data_color
        )

    fit_linestyle = '--' if plot_linestyle is None else plot_linestyle
    resid_linestyle = '-' if plot_linestyle is None else plot_linestyle

    ax_main.plot(
        peak_heights, theoretical, linestyle=fit_linestyle, 
        color=fit_color, label="Asymptotic Fit",
    )

    if residuals is None:
        residuals = 10.0**(np.log10(estimates) - np.log10(theoretical)) - 1.0

    ax_resid.semilogx(
        peak_heights, residuals, 
        linestyle=resid_linestyle, 
        color=data_color,
        zorder=1
    )
    ax_resid.axhline(0, color="k", linestyle="--", zorder=0)    

    if show_log_x:
        ax_main.set_xscale("log", base=2)

    if show_log_y:
        ax_main.set_yscale("log", base=10)

    ax_main.set_ylabel(ylabel, fontsize=main_y_label_text_size)
    ax_resid.set_ylabel(RESIDUAL_LABEL, fontsize=resid_y_label_text_size)


    latex_formatter = lambda x, pos: rf"${x:.2f}$"
    ax_resid.set_xlabel(
        get_peak_xlabel(mass_def_key), fontsize=x_label_text_size
    )
    ax_resid.xaxis.set_major_formatter(ticker.FuncFormatter(latex_formatter))

    ax_main.yaxis.set_tick_params(labelsize=main_y_tick_text_size)
    ax_resid.yaxis.set_tick_params(labelsize=resid_y_tick_text_size)
    ax_resid.xaxis.set_tick_params(labelsize=x_tick_text_size) 

    ax_resid.set_ylim(bottom=resid_min, top=resid_max)

    if show_legend:
        ax_main.legend(fontsize=legend_text_size)


    if return_fig:
        return ax_main, ax_resid

    plt.show()


def plot_single_accumulation_history(
        scale_factors: np.ndarray,
        accumulation_history: np.ndarray, 
        accumulation_rate: np.ndarray, 
        mf_type: str = "differential",
        normalizing_factor: float = 1.0,
        return_fig: bool = False,
        x_label_text_size: int = 10,
        top_y_label_text_size: int = 10,
        bottom_y_label_text_size: int = 10,
        x_tick_text_size: int = 10,
        top_y_tick_text_size: int = 10,
        bottom_y_tick_text_size: int = 10,
        show_top_log_y: bool = True, 
        show_bottom_log_y: bool = True, 
        ax_top: plt.Axes | None = None,
        ax_bottom: plt.Axes | None = None,
        plot_linestyle: tuple = (0, ()),
        color: tuple[float, float, float, float] | None = None
    ) -> tuple[plt.Axes | None, plt.Axes | None] | None:
    
    if (ax_top is None) and (ax_bottom is None):
        _, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(6, 8), sharex=True, sharey=False,
            gridspec_kw={"hspace" : 0.0}
        )

    # If color is not provided, use the default color from matplotlib
    if color is None:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    
    ax_top.semilogx(
        scale_factors, 
        accumulation_history / normalizing_factor, 
        linestyle=plot_linestyle,
        color=color,
        zorder=1
    )

    ax_bottom.semilogx(
        scale_factors, 
        accumulation_rate, 
        linestyle=plot_linestyle, 
        color=color,
        zorder=1
    )

    if (normalize := (normalizing_factor != 1.0)):
        ax_top.axhline(1, color="grey", linestyle="dotted", zorder=0)

    ax_top.yaxis.set_tick_params(labelsize=top_y_tick_text_size)
    ax_bottom.yaxis.set_tick_params(labelsize=bottom_y_tick_text_size)
    ax_bottom.xaxis.set_tick_params(labelsize=x_tick_text_size)

    ax_top.set_ylabel(get_y_label(mf_type, normalize), fontsize=top_y_label_text_size)
    ax_bottom.set_ylabel(RATE_PLOT_LABELS[mf_type], fontsize=bottom_y_label_text_size)
    ax_bottom.set_xlabel(SCALE_FACTOR_LABEL, fontsize=x_label_text_size)


    if show_top_log_y: 
        ax_top.set_yscale("log")

    if show_bottom_log_y:
        ax_bottom.set_yscale("symlog")

    if return_fig: return ax_top, ax_bottom


def plot_accumulation_history_with_fits(
        data_scale_factors: np.ndarray,
        data_history: np.ndarray,
        data_rate: np.ndarray,
        fitted_scale_factors: np.ndarray,
        fitted_history: np.ndarray,
        fitted_rate: np.ndarray,
        history_residual_scale_factors: np.ndarray,
        history_residuals: np.ndarray,
        rate_residual_scale_factors: np.ndarray,
        rate_residuals: np.ndarray,
        mf_type: str = "differential",
        data_normalizing_factor: float = 1.0,
        fitted_normalizing_factor: float = 1.0,
        top_resid_min: float | None = None,
        top_resid_max: float | None = None,
        bottom_resid_min: float | None = None,
        bottom_resid_max: float | None = None,
        return_fig: bool = False,
        show_legend: bool = True,
        legend_text_size: int = 10,
        x_label_text_size: int = 10,
        top_main_y_label_text_size: int = 10,
        top_resid_y_label_text_size: int = 10,
        bottom_main_y_label_text_size: int = 10,
        bottom_resid_y_label_text_size: int = 10,
        x_tick_text_size: int = 10,
        top_main_y_tick_text_size: int = 10,
        top_resid_y_tick_text_size: int = 10,
        bottom_main_y_tick_text_size: int = 10,
        bottom_resid_y_tick_text_size: int = 10,
        show_top_log_y: bool = True, 
        show_bottom_log_y: bool = True, 
        plot_marker_style: str = ".",
        plot_linestyle: tuple = (0, ()),
        ax_main_top: plt.Axes | None = None,
        ax_resid_top: plt.Axes | None = None,
        ax_main_bottom: plt.Axes | None = None,
        ax_resid_bottom: plt.Axes | None = None,
        color: tuple[float, float, float, float] | None = None
    ) -> tuple[plt.Axes | None, plt.Axes | None, plt.Axes | None, plt.Axes | None] | None:

    if any([
        ax_main_top is None, 
        ax_resid_top is None, 
        ax_main_bottom is None, 
        ax_resid_bottom is None]
    ):
        _, axes = plt.subplots(
            4, 1, figsize=(6, 12), sharex=True, sharey=False,
            gridspec_kw={"hspace" : 0.0, "height_ratios" : [3, 1, 3, 1]}
        )
        (ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom) = axes


    if color is None:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        theoretical_color = "grey"
    else:
        theoretical_color = color


    ax_main_top.semilogx(
        data_scale_factors, 
        data_history / data_normalizing_factor,
        linestyle='',
        marker=plot_marker_style, 
        markersize=legend_text_size/3,
        color=color, 
        label=r"${\rm Data}$"
    )
    ax_main_top.semilogx(
        fitted_scale_factors, 
        fitted_history / fitted_normalizing_factor,
        linestyle=plot_linestyle, 
        color=theoretical_color, 
        label=r"${\rm Fit}$"
    )

    is_data_normalized = (data_normalizing_factor != 1.0)
    is_fit_normalized = (fitted_normalizing_factor != 1.0)
    
    if (normalize := (is_data_normalized or is_fit_normalized)):
        ax_main_top.axhline(1, color="grey", linestyle="dotted", zorder=0)


    ax_main_top.set_ylabel(
        get_y_label(mf_type, normalize), fontsize=top_main_y_label_text_size
    )

    ax_resid_top.semilogx(
        history_residual_scale_factors, 
        history_residuals, 
        linestyle=plot_linestyle,
        linewidth=2.0,
        color=color
    )

    ax_resid_top.axhline(0, color="grey", linestyle="--")

    ax_main_bottom.semilogx(
        data_scale_factors, 
        data_rate, 
        linestyle='',
        marker=plot_marker_style, 
        markersize=legend_text_size/3,
        color=color
    )
    ax_main_bottom.semilogx(
        fitted_scale_factors, 
        fitted_rate, 
        linestyle=plot_linestyle,
        color=theoretical_color
    )
    ax_main_bottom.set_ylabel(
        RATE_PLOT_LABELS[mf_type], fontsize=bottom_main_y_label_text_size
    )
    ax_resid_bottom.semilogx(
        rate_residual_scale_factors, 
        rate_residuals, 
        linestyle=plot_linestyle,
        linewidth=2.0,
        color=color,
        zorder=1
    )
    ax_resid_bottom.axhline(0, color="grey", linestyle="--", zorder=0)

    ax_resid_top.set_ylabel(
        RESIDUAL_LABEL, fontsize=top_resid_y_label_text_size
    )
    ax_resid_bottom.set_ylabel(
        RESIDUAL_LABEL, fontsize=bottom_resid_y_label_text_size
    )
    ax_resid_bottom.set_xlabel(
        SCALE_FACTOR_LABEL, fontsize=x_label_text_size
    )

    ax_main_top.yaxis.set_tick_params(labelsize=top_main_y_tick_text_size)
    ax_main_bottom.yaxis.set_tick_params(labelsize=bottom_main_y_tick_text_size)
    ax_resid_top.yaxis.set_tick_params(labelsize=top_resid_y_tick_text_size)
    ax_resid_bottom.yaxis.set_tick_params(labelsize=bottom_resid_y_tick_text_size)
    ax_resid_bottom.xaxis.set_tick_params(labelsize=x_tick_text_size)

    ax_resid_top.set_ylim(bottom=top_resid_min, top=top_resid_max)
    ax_resid_bottom.set_ylim(bottom=bottom_resid_min, top=bottom_resid_max)

    if show_top_log_y:
        ax_main_top.set_yscale("log")

    if show_bottom_log_y:
        ax_main_bottom.set_yscale("symlog")

    if show_legend:
        ax_main_top.legend(fontsize=legend_text_size)

    if return_fig: 
        return ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom