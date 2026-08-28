import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

PHYSICAL_PLT_LABELS = {
    "x" : r"$x$ $[{\rm Mpc}/h]$", 
    "y" : r"$y$ $[{\rm Mpc}/h]$", 
    "z" : r"$z$ $[{\rm Mpc}/h]$"
}

COMOVING_PLT_LABELS = {
    "x" : r"$x$ $[{\rm cMpc}/h]$", 
    "y" : r"$y$ $[{\rm cMpc}/h]$", 
    "z" : r"$z$ $[{\rm cMpc}/h]$"
}

PLT_LABELS = {"physical" : PHYSICAL_PLT_LABELS, "comoving" : COMOVING_PLT_LABELS}

DEFAULT_PLOTS_OPTIONS = {
    "label_size" : 16,
    "tick_size" : 14,
    "legend_size" : 14,
}

DEFAULT_TEXTBOX_OPTIONS = {
    "x" : 0.05,
    "y" : 0.85,
    "fontsize" : 10,
    "multialignment" : "left",
    "bbox" : {
        "boxstyle" : "round,pad=0.3",
        "facecolor" : "white",
        "edgecolor" : "black",
        "alpha" : 0.5
    }
}

DEFAULT_SCALEBAR_OPTIONS = {
    "x": 0.9,           # normalized x-position (axes fraction)
    "y": 0.88,          # normalized y-position (axes fraction)
    "color": "white",   # color of the scale bar
    "lw": 2,            # line width of the scale bar
    "text_color": "white",  # color of the scale bar label
    "label_offset": 0.02,   # vertical offset (as a fraction of the y-range) for the label
    "fontsize": 12,         # font size for the label
}

DEFAULT_HALO_OPTIONS = {
    "inset_fraction": 0.3,         # Fraction of the main Axes size used for the inset
    "inset_location": "lower right",  # Inset location relative to the main Axes
    "inset_location_x" : 0.9,       # Normalized x-position of the inset
    "inset_location_y" : 0.1,      # Normalized y-position of the inset
    "borderpad": 1.0,              # Padding around the inset Axes
    "circle_color": "white",       # Color for the circle on the main map
    "circle_lw": 2.0,              # Line width of the circle
    "line_color": "white",         # Color for the connector lines
    "line_lw": 1.5,                # Line width for the connector lines
    "connector_angles": [30, -30], # Angles (in degrees) from halo center to use for connecting lines
    "inset_text_color": "white",  # Color for the inset text
    "inset_text_fontsize": 10,    # Font size for the inset text
    "tick_size": 10
}

def display_particles(
        coordinates: np.ndarray,
        spacing: int,
        color: tuple[float, float, float, float] | None = None,
        plt_setup: tuple[Figure, list[Axes], list[Axes]] | None = None,
    ) -> tuple[Figure, list[Axes], list[Axes]] | None:

    """Display the particles in the given particle collection."""
    if plt_setup is None:
        fig = plt.figure(figsize=(28, 16))
        axs = [fig.add_subplot(2, 3, i + 1) for i in range(3)]
        axs_3d = [
            fig.add_subplot(2, 3, i + 1, projection="3d")
            for i in range(3, 6)
        ]
    else:
        fig, axs, axs_3d = plt_setup

    data = {
        "x": coordinates[:, 0][::spacing],
        "y": coordinates[:, 1][::spacing],
        "z": coordinates[:, 2][::spacing],
    }

    two_d_title = lambda i, j: rf"${{\rm 2D view}}$ $({{{i}}}-{{{j}}})$"
    three_d_title = lambda i, j, k: rf"${{\rm 3D view}}$ $({{{i}}}-{{{j}}}-{{{k}}})$"
    # three_d_title = lambda i, j, k: f"3D view ({i}-{j}-{k})"
    views = [["x", "y", "z"], ["y", "z", "x"], ["z", "x", "y"]]

    for i, j, k, ax, ax_3d in zip(*views, axs, axs_3d):
        ax.scatter(data[i], data[j], s=0.01, c=color)
        ax.set_title(two_d_title(i, j), fontsize=16)
        ax.set_xlabel(rf"{i} [$h^{{-1}}{{\rm Mpc}}$]", fontsize=16)
        ax.set_ylabel(rf"{j} [$h^{{-1}}{{\rm Mpc}}$]", fontsize=16)

        ax_3d.scatter(data[i], data[j], data[k], s=0.01, c=color)
        ax_3d.set_title(three_d_title(i, j, k), fontsize=16)
        ax_3d.set_xlabel(rf"{i} [$h^{{-1}}{{\rm Mpc}}$]", fontsize=12)
        ax_3d.set_ylabel(rf"{j} [$h^{{-1}}{{\rm Mpc}}$]", fontsize=12)
        ax_3d.set_zlabel(rf"{k} [$h^{{-1}}{{\rm Mpc}}$]", fontsize=12)

    for ax in axs_3d:
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    if plt_setup is not None:
        return fig, axs, axs_3d
    plt.tight_layout()
    plt.show()

def get_plot_labels(normal_axes: str, use_comoving: bool = True) -> tuple[str, str]:
    labels = PLT_LABELS["comoving" if use_comoving else "physical"]
    match normal_axes:
        case "x":
            return labels["y"], labels["z"]
        case "y":
            return labels["x"], labels["z"]
        case "z":
            return labels["x"], labels["y"]
        case _:
            raise ValueError(
                f"Invalid normal axes {normal_axes}: must be 'x', 'y', or 'z'"
            )
        
def plot_density_map(
        density_map: np.ndarray,
        extent: tuple[float, float, float, float],
        cmap: str = "mako", 
        normal: str = "z",
        return_fig: bool = False,
        use_comoving: bool = True,
        plt_options: dict | None = None,
    ) -> plt.Figure | None:

    fig, ax = plt.subplots()

    im = ax.imshow(
        density_map.T,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="auto",
        cmap=sns.color_palette(cmap, as_cmap=True),
        norm=LogNorm()
    )

    x_label, y_label = get_plot_labels(normal, use_comoving=use_comoving)

    if plt_options is None:
        plt_options = DEFAULT_PLOTS_OPTIONS
    else:
        for key, value in DEFAULT_PLOTS_OPTIONS.items():
            if key not in plt_options:
                plt_options[key] = value

    ax.set_xlabel(x_label, fontsize=plt_options["label_size"])
    ax.set_ylabel(y_label, fontsize=plt_options["label_size"])
    ax.tick_params(axis="both", labelsize=plt_options["tick_size"])

    # Create the colorbar in the separate axis (cax)
    cbar = plt.colorbar(im, ax=ax, pad=0.00)
    cbar.set_label(
        r"$\rho$ $[h^{2} M_{\odot} / {\rm Mpc}^{3}]$",
        fontsize=plt_options["label_size"]
    )

    if return_fig:
        return fig
    else:
        plt.show()

def plot_projected_density_map(
        projected_density_map: np.ndarray,
        extent: tuple[float, float, float, float],
        cmap: str = "mako", 
        normal: str = "z",
        return_fig: bool = False,
        use_comoving: bool = True,
        plt_options: dict | None = None,
    ) -> plt.Figure | None:

    fig, ax = plt.subplots()

    im = ax.imshow(
        projected_density_map.T,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="auto",
        cmap=sns.color_palette(cmap, as_cmap=True),
        norm=LogNorm()
    )

    x_label, y_label = get_plot_labels(normal, use_comoving=use_comoving)

    if plt_options is None:
        plt_options = DEFAULT_PLOTS_OPTIONS
    else:
        for key, value in DEFAULT_PLOTS_OPTIONS.items():
            if key not in plt_options:
                plt_options[key] = value

    ax.set_xlabel(x_label, fontsize=plt_options["label_size"])
    ax.set_ylabel(y_label, fontsize=plt_options["label_size"])
    ax.tick_params(axis="both", labelsize=plt_options["tick_size"])

    # Create the colorbar in the separate axis (cax)
    cbar = plt.colorbar(im, ax=ax, pad=0.00)
    cbar.set_label(
        r"$\rho$ $[h M_{\odot} / {\rm Mpc}^{2}]$",
        fontsize=plt_options["label_size"]
    )

    if return_fig:
        return fig
    else:
        plt.show()

def plot_overdensity_map(
        overdensity_map: np.ndarray,
        extent: tuple[float, float, float, float],
        cmap: str = "mako", 
        normal: str = "z",
        return_fig: bool = False,
        use_comoving: bool = True,
    ) -> plt.Figure | None:
    
    fig, ax = plt.subplots()

    # Plot the density histogram
    im = ax.imshow(
        overdensity_map.T,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=cmap,
        norm=LogNorm()
    )

    x_label, y_label = get_plot_labels(normal, use_comoving=use_comoving)

    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Create the colorbar in the separate axis (cax)
    cbar = plt.colorbar(im, ax=ax, pad=0.00)
    cbar.set_label(r"${\rm Density \; Contrast}$, $1+\delta(a)$")

    if return_fig:
        return fig
    else:
        plt.show()

def add_timestep_info( 
        fig: plt.Figure, 
        scale_factor: float,
        redshift: float,
        age: float, 
        textbox_options: dict | None = None,
        bbox_options: dict | None = None,
    ) -> None:

    ax = fig.axes[0]

    if textbox_options is None:
        textbox_options = DEFAULT_TEXTBOX_OPTIONS
    else:
        for key, value in DEFAULT_TEXTBOX_OPTIONS.items():
            if key not in textbox_options:
                textbox_options[key] = value
    
    if bbox_options is not None:
        for key, value in DEFAULT_TEXTBOX_OPTIONS["bbox"].items():
            if key not in bbox_options:
                bbox_options[key] = value
        textbox_options["bbox"] = bbox_options

    fig_width = ax.get_xlim()[1] - ax.get_xlim()[0]
    fig_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    textbox_options["x"] = textbox_options["x"] * fig_width
    textbox_options["y"] = textbox_options["y"] * fig_height

    rounded_redshift = round(redshift, 2)
    plt_redshift = (
        abs(rounded_redshift) 
        if abs(rounded_redshift) == 0.0 else 
        rounded_redshift
    )

    text_str = (
        f"$t={age:.2f}$ ${{\\rm Gyr}}$\n"
        f"$a={scale_factor:.2f}$\n"
        f"$z={plt_redshift:.2f}$"
    )

    ax.text(s=text_str, **textbox_options)


def annotate_length_scale(
    ax: plt.Axes,
    length: float,
    units: str = "Mpc",
    use_comoving: bool = True,
    scale_factor: float = 1.0,
    scalebar_options: dict | None = None,
) -> None:
    """
    Annotate the given Axes with a horizontal scale bar that represents a given
    length scale. The conversion between comoving and physical coordinates is done
    using the provided scale_factor.

    Parameters
    ----------
    ax : plt.Axes
        The Axes object on which to annotate the scale bar.
    length : float
        The length to display.
    units : str, optional
        The units of 'length': either "Mpc" (physical) or "cMpc" (comoving).
        Default is "Mpc".
    use_comoving : bool, optional
        If True, the data in the Axes is in comoving coordinates.
        If False, the data is in physical coordinates.
    scale_factor : float, optional
        The scale factor to convert between comoving and physical lengths.
    scalebar_options : dict, optional
        A dictionary with options for the scale bar and label. If not provided,
        defaults are used. Recognized keys are:
            - "x": normalized x position (default 0.9)
            - "y": normalized y position (default 0.88)
            - "color": line color (default "white")
            - "lw": line width (default 2)
            - "text_color": label text color (default "white")
            - "label_offset": vertical offset (fraction of y-range) for the label (default 0.02)
            - "fontsize": font size for the label (default 12)

    Returns
    -------
    None
        The Axes is modified in place.
    """
    # Merge in defaults if scalebar_options is not complete.
    if scalebar_options is None:
        scalebar_options = DEFAULT_SCALEBAR_OPTIONS.copy()
    else:
        for key, value in DEFAULT_SCALEBAR_OPTIONS.items():
            if key not in scalebar_options:
                scalebar_options[key] = value

    # Determine conversion: we want to draw a bar whose length in data coordinates
    # represents the given 'length' in the units provided.
    if use_comoving:
        # Data are in comoving coordinates (cMpc/h)
        if units == "Mpc":
            data_length = length / scale_factor
        elif units == "cMpc":
            data_length = length
        else:
            raise ValueError("units must be either 'Mpc' or 'cMpc'")
    elif units == "Mpc":
        data_length = length
    elif units == "cMpc":
        data_length = length * scale_factor
    else:
        raise ValueError("units must be either 'Mpc' or 'cMpc'")

    # Get data coordinate limits.
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Convert normalized position to data coordinates.
    # bar_right = x_min + scalebar_options["x"] * x_range
    # bar_y = y_min + scalebar_options["y"] * y_range
    text_x = x_min + scalebar_options["x"] * x_range
    text_y = y_min + scalebar_options["y"] * y_range
    bar_y = text_y + scalebar_options["label_offset"] * y_range
    bar_right = text_x  # Right edge of the bar at text_x
    bar_left = bar_right - data_length

    # The bar will be drawn horizontally, with its right end at bar_right and left end
    # offset by the data_length.
    bar_left = bar_right - data_length

    # Draw the scale bar.
    ax.plot(
        [bar_left, bar_right], 
        [bar_y, bar_y],
        color=scalebar_options["color"],
        lw=scalebar_options["lw"]
    )

    # Create a label centered above the bar.
    # unit_text = r"cMpc" if units == "cMpc" else "Mpc"
    # label_text = f"${length:.0f}$ ${{\mathrm{{units}}}}/h$"
    label_text = f"${length:.0f}$ ${{\\rm {units}}}/h$"
    label_y = bar_y + scalebar_options["label_offset"] * y_range

    ax.text(
        0.5 * (bar_left + bar_right),
        label_y,
        label_text,
        color=scalebar_options["text_color"],
        ha="center",
        va="bottom",
        fontsize=scalebar_options["fontsize"],
    )


def annotate_halo(
        ax: plt.Axes,
        density_map: np.ndarray,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        halo_pos: np.ndarray,      # 1x3 array [x, y, z]
        halo_size: float,
        cmap=None,
        norm=None,
        halo_options: dict | None = None,
    ) -> None:
    """
    Annotate a halo on a density/overdensity map by drawing a white circle on the main Axes 
    and adding an inset zoomed view of the halo's region. Connector lines link the halo 
    in the main plot to the inset.

    Parameters
    ----------
    ax : plt.Axes
        The main Axes on which the density/overdensity map is plotted.
    density_map : np.ndarray
        2D array of the density or overdensity values.
    x_edges, y_edges : np.ndarray
        The bin edges for the x and y axes (used to define the extent of the image).
    halo_pos : np.ndarray
        A 1×3 array representing the (x, y, z) position of the halo.
        Only (x, y) are used for the 2D projection.
    halo_size : float
        The halo's size (radius) in the same units as x_edges and y_edges.
    cmap : optional
        The colormap used in the main plot (passed to the inset plot for consistency).
    norm : optional
        The normalization (e.g., LogNorm) used in the main plot (passed to the inset).
    halo_options : dict, optional
        A dictionary of options to customize the halo annotation. Keys include:
          - "inset_fraction": Fraction (0–1) of the Axes size for the inset (default 0.3).
          - "inset_location": Inset location string for inset_axes (default "upper right").
          - "borderpad": Padding for the inset (default 1.0).
          - "circle_color": Color for the circle on the main map (default "white").
          - "circle_lw": Line width for the circle (default 2.0).
          - "line_color": Color for the connector lines (default "white").
          - "line_lw": Line width for the connector lines (default 1.5).
          - "connector_angles": List of two angles in degrees to choose points on the circle for connection (default [30, -30]).
          - "inset_text_color": Color for the inset text (default "white").
          - "inset_text_fontsize": Font size for the inset text (default 10).
        Missing keys are filled in with defaults.
    
    Returns
    -------
    None
        The Axes is modified in place.
    """
    # Merge in default halo options
    if halo_options is None:
        halo_options = DEFAULT_HALO_OPTIONS.copy()
    else:
        for key, value in DEFAULT_HALO_OPTIONS.items():
            if key not in halo_options:
                halo_options[key] = value

    # Extract halo center (only x, y used for the 2D projection)
    halo_x, halo_y = halo_pos[0], halo_pos[1]

    # 1) Draw a circle on the main Axes around the halo.
    circle = Circle(
        (halo_x, halo_y),
        radius=halo_size,
        edgecolor=halo_options["circle_color"],
        fill=False,
        lw=halo_options["circle_lw"]
    )
    ax.add_patch(circle)

    # 2) Create an inset Axes using the specified inset fraction, location, and borderpad.
    inset_width = f"{100 * halo_options['inset_fraction']}%"   # e.g., "30%"
    inset_height = f"{100 * halo_options['inset_fraction']}%"
    main_extent = (
        x_edges[0] + halo_x, 
        x_edges[-1] + halo_x, 
        y_edges[0] + halo_y, 
        y_edges[-1] + halo_y
    )
    inset_x_left = halo_options["inset_location_x"] * main_extent[0]
    inset_x_right = halo_options["inset_location_x"] * main_extent[1]
    inset_y_bottom = halo_options["inset_location_y"] * main_extent[2]
    inset_y_top = halo_options["inset_location_y"] * main_extent[3]

    inset_ax = inset_axes(
        ax,
        width=inset_width,
        height=inset_height,
        # bbox_to_anchor=(inset_x_left, inset_x_right, inset_y_bottom, inset_y_top),
        loc=halo_options["inset_location"],
        borderpad=halo_options["borderpad"]
    )

    # 3) Plot the same density map in the inset.
    
    inset_ax.imshow(
        density_map.T,
        origin="lower",
        extent=main_extent,
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        aspect="auto"
    )

    # Set the inset's view to zoom around the halo (x and y limits based on halo center and size).
    inset_ax.set_xlim(halo_x - halo_size, halo_x + halo_size)
    inset_ax.set_ylim(halo_y - halo_size, halo_y + halo_size)

    # Remove ticks for a cleaner inset.
    inset_ax.tick_params(axis="both", labelsize=halo_options["tick_size"])
    inset_ax.tick_params(axis="both", labelcolor=halo_options["inset_text_color"])

    # 4) Connect the halo's circle to the inset via connector lines.
    # We'll choose two points on the circle using the angles in 'connector_angles'.
    for angle in halo_options["connector_angles"]:
        theta = np.radians(angle)
        x_circ = halo_x + halo_size * np.cos(theta)
        y_circ = halo_y + halo_size * np.sin(theta)

        # Transform the circle boundary point from data to figure coordinates.
        circ_disp = ax.transData.transform((x_circ, y_circ))
        circ_fig = ax.figure.transFigure.inverted().transform(circ_disp)

        # For the inset, choose a corner in inset Axes coordinates.
        # Here we choose (0,1) for positive angles and (0,0) for negative angles.
        corner = (0, 1) if angle > 0 else (0, 0)
        inset_disp = inset_ax.transAxes.transform(corner)
        inset_fig = ax.figure.transFigure.inverted().transform(inset_disp)

        # Create and add a line in figure coordinates connecting the two points.
        line = plt.Line2D(
            [circ_fig[0], inset_fig[0]],
            [circ_fig[1], inset_fig[1]],
            transform=None,  # Coordinates are already in figure space.
            color=halo_options["line_color"],
            lw=halo_options["line_lw"]
        )
        ax.figure.add_artist(line)