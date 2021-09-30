"""Plotting code for the PNMOL experiments."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

plt.style.use(["experiments/lines_and_ticks.mplstyle", "experiments/font.mplstyle"])

PATH_RESULTS = "experiments/results/figure1/"


# Extract from the paper template:
#
# Papers are in 2 columns with the overall line width of 6.75~inches (41~picas).
# Each column is 3.25~inches wide (19.5~picas).  The space
# between the columns is .25~inches wide (1.5~picas).  The left margin is 0.88~inches (5.28~picas).
# Use 10~point type with a vertical spacing of
# 11~points. Please use US Letter size paper instead of A4.
AISTATS_LINEWIDTH_DOUBLE = 6.75
AISTATS_TEXTWIDTH_SINGLE = 3.25


def figure_1(
    path=PATH_RESULTS, methods=("pnmol_white", "pnmol_latent", "tornadox", "reference")
):

    results = [figure_1_load_results(prefix=method, path=path) for method in methods]

    results_reference = results[-1]
    means_reference, *_, x_reference = results_reference

    figure_size = (AISTATS_LINEWIDTH_DOUBLE, AISTATS_TEXTWIDTH_SINGLE)
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=3,
        dpi=200,
        figsize=figure_size,
        sharex=True,
        sharey=True,
    )

    vmin, vmax = None, None
    pnmol_colorbar = None
    for axis_row, method, result in zip(axes, methods, results):
        m, s, t, x = result
        n = jnp.minimum(len(m), len(means_reference))
        T, X = jnp.meshgrid(t[:n], x[:, 0])
        error = jnp.abs(means_reference[:n] - m[:n])
        if vmin is None and vmax is None:
            vmin_error, vmax_error = jnp.amin(error), jnp.amax(error)
            vmin_std, vmax_std = jnp.amin(s), jnp.amax(s)
            vmin = jnp.minimum(vmin_error, vmin_std)
            vmax = jnp.maximum(vmax_error, vmax_std)

        contour_args = {"alpha": 0.8}
        contour_args_means = {"vmin": 0.0, "vmax": 1.0, "cmap": "Greys"}
        contour_args_errors = {"vmin": vmin, "vmax": vmax, "cmap": "inferno"}
        figure_1_plot_contour(
            axis_row[0], X, T, m[:n].T, **contour_args, **contour_args_means
        )
        figure_1_plot_contour(
            axis_row[1], X, T, s[:n].T, **contour_args, **contour_args_errors
        )
        bar_error = figure_1_plot_contour(
            axis_row[2], X, T, error.T, **contour_args, **contour_args_errors
        )
        if pnmol_colorbar is None:
            fig.colorbar(bar_error, ax=axes[:, -1].ravel().tolist())
            pnmol_colorbar = 1

        for ax in axis_row:
            ax.set_yticks(t[:n:4])
            ax.set_xticks(x[:, 0])

    # x-labels
    bottom_row_axis = axes[-1]
    for ax in bottom_row_axis:
        ax.set_xlabel("Space")

    # y-labels
    nicer_label = {
        "pnmol_white": "White",
        "pnmol_latent": "Latent",
        "tornadox": "PN+MOL",
        "reference": "Scipy",
    }
    left_column_axis = axes[:, 0]
    for ax, label in zip(left_column_axis, methods):
        ax.set_ylabel(nicer_label[label])

    # Common settings for all plots
    for ax in axes.flatten():
        ax.set_xticklabels(())
        ax.set_yticklabels(())

    # Column titles
    top_row_axis = axes[0]
    ax1, ax2, ax3 = top_row_axis
    ax1.set_title(r"$\bf a.$ " + "Mean", loc="left", fontsize="medium")
    ax2.set_title(r"$\bf b.$ " + "Std.-dev.", loc="left", fontsize="medium")
    ax3.set_title(r"$\bf c.$ " + "Error", loc="left", fontsize="medium")
    plt.savefig(path + "figure1.pdf")
    plt.show()


def figure_1_load_results(*, prefix, path=PATH_RESULTS):
    path_means = path + prefix + "_means.npy"
    path_stds = path + prefix + "_stds.npy"
    path_ts = path + prefix + "_ts.npy"
    path_xs = path + prefix + "_xs.npy"
    means = jnp.load(path_means)
    stds = jnp.load(path_stds)
    ts = jnp.load(path_ts)
    xs = jnp.load(path_xs)
    return means, stds, ts, xs


def figure_1_plot_contour(ax, /, *args, **kwargs):
    """Contour lines with fill color and sharp edges."""
    ax.contour(*args, **kwargs)
    return ax.contourf(*args, **kwargs)
