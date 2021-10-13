"""Plotting code for the PNMOL experiments."""

import pathlib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

STYLESHEETS = [
    "experiments/style/lines_and_ticks.mplstyle",
    "experiments/style/font.mplstyle",
    "experiments/style/colors.mplstyle",
    "experiments/style/markers.mplstyle",
    "experiments/style/bottomleftaxes.mplstyle",
]

PATH_RESULTS = "experiments/results/"


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
    path = path + "figure1/"
    plt.style.use(STYLESHEETS)

    results = [figure_1_load_results(prefix=method, path=path) for method in methods]

    results_reference = results[-1]
    means_reference, *_, x_reference = results_reference

    figure_size = (AISTATS_LINEWIDTH_DOUBLE, 0.8 * AISTATS_TEXTWIDTH_SINGLE)
    fig, axes = plt.subplots(
        nrows=len(methods) - 1,
        ncols=3,
        dpi=200,
        figsize=figure_size,
        sharex=True,
        sharey=True,
    )

    vmin, vmax = None, None
    pnmol_colorbar = None
    for axis_row, method, result in zip(axes, methods[:-1], results):
        m, s, t, x = result
        n = jnp.minimum(len(m), len(means_reference))
        T, X = jnp.meshgrid(t[:n], x[:, 0])
        error = jnp.abs(means_reference[:n] - m[:n])
        if vmin is None and vmax is None:
            vmin_error, vmax_error = jnp.amin(error), jnp.amax(error)
            vmin_std, vmax_std = jnp.amin(s), jnp.amax(s)
            vmin = jnp.minimum(vmin_error, vmin_std)
            vmax = jnp.maximum(vmax_error, vmax_std)

        contour_args = {"alpha": 0.8, "levels": 20}
        contour_args_means = {"vmin": 0.0, "vmax": 0.1, "cmap": "Greys"}
        contour_args_errors = {"cmap": "inferno"}
        figure_1_plot_contour(
            axis_row[0], X, T, m[:n].T, **contour_args, **contour_args_means
        )
        bar = figure_1_plot_contour(
            axis_row[1], X, T, s[:n].T + 1e-12, **contour_args, **contour_args_errors
        )
        fig.colorbar(bar, ax=axis_row[1])

        bar = figure_1_plot_contour(
            axis_row[2], X, T, error.T + 1e-12, **contour_args, **contour_args_errors
        )
        fig.colorbar(bar, ax=axis_row[2])

        # if pnmol_colorbar is None:
        #     fig.colorbar(bar_error, ax=axes[:, -1].ravel().tolist())
        #     pnmol_colorbar = 1

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
    plt.savefig(path + "figure.pdf")
    plt.show()


def figure_1_singlerow(
    path=PATH_RESULTS, methods=("pnmol_white", "pnmol_latent", "tornadox", "reference")
):
    path = path + "figure1/"
    plt.style.use(STYLESHEETS)

    results = [figure_1_load_results(prefix=method, path=path) for method in methods]

    results_reference = results[-1]
    means_reference, *_, x_reference = results_reference

    figure_size = (AISTATS_LINEWIDTH_DOUBLE, 0.4 * AISTATS_TEXTWIDTH_SINGLE)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        dpi=200,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [3, 2, 2, 3]},
    )

    axes = np.flip(axes)
    ax_mol_error, ax_mol_mean = axes[:2]
    ax_pnmol_mean, ax_pnmol_error = axes[2:]

    contour_args = {"alpha": 0.7, "levels": 20}
    contour_args_means = {"vmin": 0.0, "vmax": 0.1, "cmap": "Greys"}
    contour_args_errors = {}

    cmap_mol, cmap_pnmol = "RdYlBu_r", "RdYlGn_r"

    bar_mol = _figure_1_plot_errors(
        ax=ax_mol_error,
        result=results[2],
        result_reference=results[-1],
        **contour_args,
        **contour_args_errors,
        cmap=cmap_mol,
        # norm=LogNorm(vmin=1e-3,vmax=1e3, clip=True)
    )
    fig.colorbar(bar_mol, ax=ax_mol_error, location="right", ticks=(0.0, 6_500, 13_000))

    bar_pnmol = _figure_1_plot_errors(
        ax=ax_pnmol_error,
        result=results[0],
        result_reference=results[-1],
        **contour_args,
        **contour_args_errors,
        cmap=cmap_pnmol,
        # norm=LogNorm(vmin=1e-3,vmax=1e3, clip=True)
    )
    fig.colorbar(bar_pnmol, ax=ax_pnmol_error, location="left", ticks=(0.0, 7.5, 15.0))

    _figure_1_plot_means(
        ax=ax_mol_mean, result=results[2], **contour_args, **contour_args_means
    )
    _figure_1_plot_means(
        ax=ax_pnmol_mean, result=results[0], **contour_args, **contour_args_means
    )

    title_style = {"loc": "left", "fontsize": "small"}
    ax_mol_error.set_title(r"$\bf MOL/1$. " + "Error/Uncertainty Ratio", **title_style)
    ax_mol_mean.set_title(r"$\bf MOL/2$. " + "Posterior mean", **title_style)

    ax_pnmol_error.set_title(r"$\bf PN/1$. " + "Error/Uncertainty Ratio", **title_style)
    ax_pnmol_mean.set_title(r"$\bf PN/2$. " + "Posterior mean", **title_style)

    plt.savefig(path + "figure.pdf")
    plt.show()

    #
    # means_mol, std_mol, t_mol, x_mol = results[2]
    # means_pnmol, std_pnmol, t_pnmol, x_pnmol = results[0]  # white
    #
    # n_mol = jnp.minimum(len(means_mol), len(means_reference))
    # error_mol =  jnp.abs(means_reference[:n_mol] - means_mol[:n_mol])
    # n_pnmol = jnp.minimum(len(means_pnmol), len(means_reference))
    # error_pnmol =  jnp.abs(means_reference[:n_pnmol] - means_mol[:n_pnmol])
    #
    # ax_pnmol_error.contour()
    #
    # vmin, vmax = None, None
    # pnmol_colorbar = None
    # for axis_row, method, result in zip(axes, methods[:-1], results):
    #     m, s, t, x = result
    #     n = jnp.minimum(len(m), len(means_reference))
    #     T, X = jnp.meshgrid(t[:n], x[:, 0])
    #     error = jnp.abs(means_reference[:n] - m[:n])
    #     if vmin is None and vmax is None:
    #         vmin_error, vmax_error = jnp.amin(error), jnp.amax(error)
    #         vmin_std, vmax_std = jnp.amin(s), jnp.amax(s)
    #         vmin = jnp.minimum(vmin_error, vmin_std)
    #         vmax = jnp.maximum(vmax_error, vmax_std)
    #
    #     contour_args = {"alpha": 0.8}
    #     contour_args_means = {"vmin": 0.0, "vmax": 1.0, "cmap": "Greys"}
    #     contour_args_errors = {"cmap": "inferno"}
    #     figure_1_plot_contour(
    #         axis_row[0], X, T, m[:n].T, **contour_args, **contour_args_means
    #     )
    #     bar = figure_1_plot_contour(
    #         axis_row[1], X, T, s[:n].T + 1e-12, **contour_args, **contour_args_errors
    #     )
    #     fig.colorbar(bar, ax=axis_row[1])
    #
    #     bar = figure_1_plot_contour(
    #         axis_row[2], X, T, error.T + 1e-12, **contour_args, **contour_args_errors
    #     )
    #     fig.colorbar(bar, ax=axis_row[2])
    #
    #     # if pnmol_colorbar is None:
    #     #     fig.colorbar(bar_error, ax=axes[:, -1].ravel().tolist())
    #     #     pnmol_colorbar = 1
    #
    #     for ax in axis_row:
    #         ax.set_yticks(t[:n:4])
    #         ax.set_xticks(x[:, 0])
    #
    # # x-labels
    # bottom_row_axis = axes[-1]
    # for ax in bottom_row_axis:
    #     ax.set_xlabel("Space")
    #
    # # y-labels
    # nicer_label = {
    #     "pnmol_white": "White",
    #     "pnmol_latent": "Latent",
    #     "tornadox": "PN+MOL",
    #     "reference": "Scipy",
    # }
    # left_column_axis = axes[:, 0]
    # for ax, label in zip(left_column_axis, methods):
    #     ax.set_ylabel(nicer_label[label])
    #
    # # Common settings for all plots
    # for ax in axes.flatten():
    #     ax.set_xticklabels(())
    #     ax.set_yticklabels(())
    #
    # # Column titles
    # top_row_axis = axes[0]
    # ax1, ax2, ax3 = top_row_axis
    # ax1.set_title(r"$\bf a.$ " + "Mean", loc="left", fontsize="medium")
    # ax2.set_title(r"$\bf b.$ " + "Std.-dev.", loc="left", fontsize="medium")
    # ax3.set_title(r"$\bf c.$ " + "Error", loc="left", fontsize="medium")


def _figure_1_plot_errors(ax, result, result_reference, **style):

    means_reference, *_, x_reference = result_reference

    m, s, t, x = result
    n = jnp.minimum(len(m), len(means_reference))
    T, X = jnp.meshgrid(t[:n], x[:, 0])
    error = jnp.abs(means_reference[:n] - m[:n])

    ax.set_xticks(x[:, 0])
    ax.set_yticks(t)
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    quotient = error / (s[:n] + 1e-12)
    return figure_1_plot_contour(ax, X.T, T.T, quotient, **style)


def _figure_1_plot_stds(ax, result, **style):

    m, s, t, x = result
    n = len(m)
    T, X = jnp.meshgrid(t[:n], x[:, 0])

    ax.set_xticks(x[:, 0])
    ax.set_yticks(t)
    ax.set_xticklabels(())
    ax.set_yticklabels(())

    return figure_1_plot_contour(ax, X.T, T.T, s + 1e-12, **style)


def _figure_1_plot_means(ax, result, **style):

    m, s, t, x = result
    n = len(m)
    T, X = jnp.meshgrid(t[:n], x[:, 0])

    ax.contourf(X.T, T.T, m, **style)
    ax.set_xticks(x[:, 0])
    ax.set_yticks(t)
    ax.set_xticklabels(())
    ax.set_yticklabels(())

    return ax


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


def figure_1_plot_contour(ax, *args, **kwargs):
    """Contour lines with fill color and sharp edges."""
    ax.contour(*args, **kwargs, linewidths=0.1)
    return ax.contourf(*args, **kwargs)


def figure_2(path=PATH_RESULTS):
    path = path + "figure2/"
    plt.style.use(STYLESHEETS)

    rmse_all = jnp.load(path + "rmse_all.npy")
    input_scales = jnp.load(path + "input_scales.npy")
    stencil_sizes = jnp.load(path + "stencil_sizes.npy")
    L_sparse = jnp.load(path + "L_sparse.npy")
    L_dense = jnp.load(path + "L_dense.npy")
    E_sparse = jnp.load(path + "E_sparse.npy")
    E_dense = jnp.load(path + "E_dense.npy")
    x = jnp.load(path + "xgrid.npy")
    fx = jnp.load(path + "fx.npy")
    dfx = jnp.load(path + "dfx.npy")
    # s1 = jnp.load(path + "s1.npy")  # not shown, thus not loaded
    s2 = jnp.load(path + "s2.npy")
    s3 = jnp.load(path + "s3.npy")

    figsize = (AISTATS_LINEWIDTH_DOUBLE, 0.8 * AISTATS_TEXTWIDTH_SINGLE)
    fig = plt.figure(constrained_layout=True, figsize=figsize, dpi=200)
    gs = fig.add_gridspec(2, 6)

    ax_L_sparse = fig.add_subplot(gs[0, 0])
    ax_L_dense = fig.add_subplot(gs[1, 0])
    ax_E_sparse = fig.add_subplot(gs[0, 1])
    ax_E_dense = fig.add_subplot(gs[1, 1])

    ax_rmse = fig.add_subplot(gs[:, 2:4])
    ax_curve = fig.add_subplot(gs[:, 4:])

    clip_value = 1e-12
    cmap = {"cmap": "Blues"}
    ax_L_sparse.imshow(jnp.abs(L_sparse) + clip_value, **cmap, aspect="auto")
    ax_L_dense.imshow(
        jnp.abs(L_dense) + clip_value,
        vmax=7 * jnp.median(jnp.abs(L_dense)),
        **cmap,
        aspect="auto",
    )

    # Add 1e-12 for a reasonable log-norm
    ax_E_sparse.imshow(
        jnp.abs(E_sparse @ E_sparse.T) + clip_value,
        **cmap,
        aspect="auto",
        norm=LogNorm(vmin=clip_value),
    )
    ax_E_dense.imshow(
        jnp.abs(E_dense @ E_dense.T) + clip_value, **cmap, aspect="auto", norm=LogNorm()
    )

    s1_style = {
        "color": "C1",
        "linestyle": "-",
    }
    s1_label = {"label": rf"$r={input_scales[1]}$ (~MLE)"}
    s2_style = {
        "color": "C0",
        "linestyle": "dashdot",
    }
    s2_label = {"label": rf"$r={input_scales[2]}$"}

    ax_rmse.semilogy(stencil_sizes, rmse_all.T[1], **s1_style, **s1_label, marker="o")
    ax_rmse.semilogy(stencil_sizes, rmse_all.T[2], **s2_style, **s2_label, marker="s")
    ax_rmse.set_xlabel("Stencil size")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.legend(
        loc="center right", fancybox=False, edgecolor="black"
    ).get_frame().set_linewidth(0.5)

    ax_curve.plot(x, fx, label="u(x)", color="black", linestyle="dashed")
    ax_curve.plot(x, dfx, label="$\Delta u(x)$", color="black")
    ax_curve.plot(x, s2, **s1_style, alpha=0.4)
    ax_curve.plot(x, s3, **s2_style, alpha=0.4)

    endmarker_style = {
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "auto",
        "markeredgewidth": 1.0,
        "markersize": 2.5,
    }
    ax_curve.plot(x[0], fx[0], **endmarker_style, color="black")
    ax_curve.plot(x[-1], fx[-1], **endmarker_style, color="black")
    ax_curve.plot(x[0], dfx[0], **endmarker_style, color="black")
    ax_curve.plot(x[-1], dfx[-1], **endmarker_style, color="black")
    ax_curve.plot(x[0, None], s2[None, 0], **endmarker_style, **s1_style)
    ax_curve.plot(x[-1, None], s2[None, -1], **endmarker_style, **s1_style)
    ax_curve.plot(x[0, None], s3[None, 0], **endmarker_style, **s2_style)
    ax_curve.plot(x[-1, None], s3[None, -1], **endmarker_style, **s2_style)

    ax_curve.set_xlabel("x")
    ax_curve.set_ylabel("u(x)")
    ax_curve.legend(
        loc="lower center", fancybox=False, edgecolor="black"
    ).get_frame().set_linewidth(0.5)

    ax_L_dense.set_xticks(())
    ax_L_dense.set_yticks(())
    ax_L_sparse.set_xticks(())
    ax_L_sparse.set_yticks(())
    ax_E_sparse.set_xticks(())
    ax_E_sparse.set_yticks(())
    ax_E_dense.set_xticks(())
    ax_E_dense.set_yticks(())

    ax_L_sparse.set_title(r"$\bf a.$ " + "Diff. matrix", loc="left", fontsize="medium")
    ax_E_sparse.set_title(r"$\bf b.$ " + "Error cov.", loc="left", fontsize="medium")
    ax_L_dense.set_title(r"$\bf c.$ " + "Diff. matrix", loc="left", fontsize="medium")
    ax_E_dense.set_title(r"$\bf d.$ " + "Error cov.", loc="left", fontsize="medium")

    ax_rmse.set_title(
        r"$\bf e.$ " + "RMSE vs. Stencil Size", loc="left", fontsize="medium"
    )
    ax_curve.set_title(
        r"$\bf f.$ " + "Solution / Prior samples", loc="left", fontsize="medium"
    )

    plt.savefig(path + "figure.pdf", dpi=300)
    plt.show()


def figure_3(path=PATH_RESULTS, methods=("pnmol_white", "tornadox")):
    path = path + "figure3/"
    plt.style.use(STYLESHEETS)

    results = [figure3_load_results(prefix=method, path=path) for method in methods]

    figure_size = (AISTATS_LINEWIDTH_DOUBLE, 0.8 * AISTATS_TEXTWIDTH_SINGLE)
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=5,
        dpi=400,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    vmin_err_rel = jnp.minimum(results[0][0].min(), results[1][0].min())
    vmax_err_rel = jnp.maximum(results[0][0].max(), results[1][0].max())

    vmin_err_abs = jnp.minimum(results[0][1].min(), results[1][1].min())
    vmax_err_abs = jnp.maximum(results[0][1].max(), results[1][1].max())

    vmin_std = jnp.minimum(results[0][2].min(), results[1][2].min())
    vmax_std = jnp.maximum(results[0][2].max(), results[1][2].max())

    vmin_calib = jnp.minimum(results[0][3].min(), results[1][3].min())
    vmax_calib = jnp.maximum(results[0][3].max(), results[1][3].max())

    vmin_time = jnp.minimum(results[0][4].min(), results[1][4].min())
    vmax_time = jnp.maximum(results[0][4].max(), results[1][4].max())

    nicer_method_name = {"tornadox": "MOL", "pnmol_white": "PNMOL"}
    for axis_row, method, result in zip(axes, methods, results):
        axis_row[0].set_ylabel(f"{nicer_method_name[method]}\ndx")
        err_mat_rel, err_mat_abs, std_mat, chi2_mat, runtime_mat, DTs, DXs = result
        extents = [
            float(DTs.min()),
            float(DTs.max()),
            float(DXs.max()),
            float(DXs.min()),
        ]
        style = {"cmap": "RdYlBu"}
        im_err_rel = axis_row[0].imshow(
            err_mat_rel,
            norm=LogNorm(
                vmin=vmin_err_rel,
                vmax=vmax_err_rel,
            ),
            extent=extents,
            aspect="auto",
            **style,
        )
        im_err_abs = axis_row[1].imshow(
            err_mat_abs,
            norm=LogNorm(
                vmin=vmin_err_abs,
                vmax=vmax_err_abs,
            ),
            extent=extents,
            aspect="auto",
            **style,
        )
        im_std = axis_row[2].imshow(
            std_mat,
            norm=LogNorm(
                vmin=vmin_std,
                vmax=vmax_std,
            ),
            extent=extents,
            aspect="auto",
            **style,
        )
        im_calib = axis_row[3].imshow(
            chi2_mat,
            norm=LogNorm(
                vmin=vmin_calib,
                vmax=vmax_calib,
            ),
            extent=extents,
            aspect="auto",
            **style,
        )
        im_rt = axis_row[4].imshow(
            runtime_mat,
            norm=LogNorm(vmin=vmin_time, vmax=vmax_time),
            extent=extents,
            aspect="auto",
            **style,
        )

        fig.colorbar(im_err_rel, ax=axis_row[0])
        fig.colorbar(im_err_abs, ax=axis_row[1])
        fig.colorbar(im_std, ax=axis_row[2])
        fig.colorbar(im_calib, ax=axis_row[3])
        fig.colorbar(im_rt, ax=axis_row[4])

    axes[0, 0].set_title(r"$\bf a1$. Relative Error", fontsize="small", loc="left")
    axes[0, 1].set_title(r"$\bf a2$. Absolute Error", fontsize="small", loc="left")
    axes[0, 2].set_title(r"$\bf a3$. Std. dev.", fontsize="small", loc="left")
    axes[0, 3].set_title(r"$\bf a4$. $\chi^2$-statistic", fontsize="small", loc="left")
    axes[0, 4].set_title(r"$\bf a5$. Run time [s]", fontsize="small", loc="left")

    axes[1, 0].set_title(r"$\bf b1$.", fontsize="small", loc="left")
    axes[1, 1].set_title(r"$\bf b2$.", fontsize="small", loc="left")
    axes[1, 2].set_title(r"$\bf b3$.", fontsize="small", loc="left")
    axes[1, 3].set_title(r"$\bf b4$.", fontsize="small", loc="left")
    axes[1, 4].set_title(r"$\bf b5$.", fontsize="small", loc="left")
    for bottom_ax in axes[-1, :]:
        bottom_ax.set_xlabel("dt")

    plt.savefig(path + "figure.pdf")
    plt.show()


def figure_3_2x2(path=PATH_RESULTS, methods=("pnmol_white", "tornadox")):
    path = path + "figure3/"
    plt.style.use(STYLESHEETS)

    results = [figure3_load_results(prefix=method, path=path) for method in methods]

    figure_size = (AISTATS_TEXTWIDTH_SINGLE, 0.8 * AISTATS_TEXTWIDTH_SINGLE)
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=2,
        dpi=400,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    print(results)
    vmin_err_rel = jnp.minimum(results[0][0].min(), results[1][0].min())
    vmax_err_rel = jnp.maximum(results[0][0].max(), results[1][0].max())
    # vmax_err_rel = 1.0
    # vmin_err_rel = 1e-3

    vmin_calib = jnp.minimum(results[0][3].min(), results[1][3].min())
    vmax_calib = jnp.maximum(results[0][3].max(), results[1][3].max())
    vmin_calib = 1.0 / vmax_calib
    # vmin_calib = 1e-2
    # vmax_calib = 100.
    nicer_method_name = {"tornadox": "MOL", "pnmol_white": "PNMOL"}
    for axis_row, method, result in zip(axes, methods, results):
        axis_row[0].set_ylabel(f"Step-size $\Delta x$", fontsize="small")
        err_mat_rel, err_mat_abs, std_mat, chi2_mat, runtime_mat, DTs, DXs = result
        print(DTs, DXs)
        extents = [
            float(DTs.min()),
            float(DTs.max()),
            float(DXs.min()),
            float(DXs.max()),
        ]
        style_chi2 = {"cmap": "RdYlBu_r"}
        style_error = {"cmap": "RdYlBu_r"}
        im_err_rel = axis_row[0].imshow(
            jnp.flip(err_mat_rel, axis=0),
            norm=LogNorm(vmin=vmin_err_rel, vmax=vmax_err_rel),
            extent=extents,
            aspect="auto",
            **style_error,
        )

        im_calib = axis_row[1].imshow(
            jnp.flip(chi2_mat, axis=0),
            norm=LogNorm(
                vmin=vmin_calib,
                vmax=vmax_calib,
            ),
            extent=extents,
            aspect="auto",
            **style_chi2,
        )

        axis_row[1].autoscale(False)
        axis_row[1].set_xticks((0.02, 1.02, 2.02))  # black magic
        axis_row[1].set_xticklabels(("$2^{-6}$", "$2^{-2.5}$", "$2^{1}$"))

        axis_row[0].autoscale(False)
        axis_row[0].set_yticks((0.07, 0.17, 0.27))  # black magic
        axis_row[0].set_yticklabels(("$2^{-6}$", "$2^{-4}$", "$2^{-2}$"))

        fig.colorbar(im_err_rel, ax=axis_row[0])
        fig.colorbar(im_calib, ax=axis_row[1])

    axes[0, 0].set_title(r"$\bf PN/1$. Relative RMSE", fontsize="small", loc="left")
    axes[0, 1].set_title(
        r"$\bf PN/1$. $\chi^2$-statistic", fontsize="small", loc="left"
    )

    axes[1, 0].set_title(r"$\bf MOL/1$. Relative RMSE", fontsize="small", loc="left")
    axes[1, 1].set_title(
        r"$\bf MOL/2$. $\chi^2$-statistic", fontsize="small", loc="left"
    )

    for bottom_ax in axes[-1, :]:
        bottom_ax.set_xlabel(r"Step-size $\Delta t$", fontsize="small")

    plt.savefig(path + "figure.pdf")
    plt.show()


def figure3_load_results(*, prefix, path=PATH_RESULTS):
    print(path)
    path_error_rel = path + prefix + "_error_rel.npy"
    path_error_abs = path + prefix + "_error_abs.npy"
    path_std = path + prefix + "_std.npy"
    path_chi2 = path + prefix + "_chi2.npy"
    path_runtime = path + prefix + "_runtime.npy"
    path_dt = path + (prefix + "_dt.npy")
    path_dx = path + (prefix + "_dx.npy")

    error_rel = jnp.load(path_error_rel)
    error_abs = jnp.load(path_error_abs)
    std = jnp.load(path_std)
    chi2 = jnp.load(path_chi2)
    runtime = jnp.load(path_runtime)
    dt = jnp.load(path_dt)
    dx = jnp.load(path_dx)

    return error_rel, error_abs, std, chi2, runtime, dt, dx


def figure3_plot_contour(ax, /, *args, **kwargs):
    """Contour lines with fill color and sharp edges."""
    ax.contour(*args, **kwargs)
    return ax.contourf(*args, **kwargs)


def figure_4():

    dxs = [0.02, 0.1]

    figsize = (AISTATS_LINEWIDTH_DOUBLE, 0.75 * AISTATS_TEXTWIDTH_SINGLE)
    fig, axes = plt.subplots(
        ncols=3, sharey=True, figsize=figsize, dpi=200, constrained_layout=True
    )
    ax_nsteps, ax_runtime, ax_chi2 = axes

    for dx, ls in zip(dxs, [":", "-", "--", "-."]):
        path = "./experiments/results/figure4/" + f"dx_{dx}_"

        style_mol = {
            "color": "gray",
            "linewidth": 3.0,
            "alpha": 0.4,
            "label": f"MOL (dx={dx})",
        }
        style_pnmol_white = {
            "color": "C1",
            "linewidth": 3.0,
            "label": f"PNMOL (white; dx={dx})",
            "alpha": 0.75,
        }
        style_pnmol_latent = {
            "color": "C0",
            "linewidth": 3.0,
            "label": f"PNMOL (latent; dx={dx})",
            "alpha": 0.75,
        }
        style_all = {"linestyle": ls, "markersize": 4}

        plt.style.use(STYLESHEETS)

        # dt = jnp.load(path + "dts.npy")

        mol_rmse = jnp.load(path + "mol_rmse.npy")
        mol_chi2 = jnp.load(path + "mol_chi2.npy")
        mol_nsteps = jnp.load(path + "mol_nsteps.npy")
        mol_time = jnp.load(path + "mol_time.npy")

        pnmol_white_rmse = jnp.load(path + "pnmol_white_rmse.npy")
        pnmol_white_chi2 = jnp.load(path + "pnmol_white_chi2.npy")
        pnmol_white_nsteps = jnp.load(path + "pnmol_white_nsteps.npy")
        pnmol_white_time = jnp.load(path + "pnmol_white_time.npy")

        pnmol_latent_rmse = jnp.load(path + "pnmol_latent_rmse.npy")
        pnmol_latent_chi2 = jnp.load(path + "pnmol_latent_chi2.npy")
        pnmol_latent_nsteps = jnp.load(path + "pnmol_latent_nsteps.npy")
        pnmol_latent_time = jnp.load(path + "pnmol_latent_time.npy")

        ax_nsteps.set_xlabel("Number of time-steps")
        ax_nsteps.loglog(mol_nsteps, mol_rmse, marker="d", **style_mol, **style_all)
        ax_nsteps.loglog(
            pnmol_white_nsteps,
            pnmol_white_rmse,
            marker="o",
            **style_pnmol_white,
            **style_all,
        )
        ax_nsteps.loglog(
            pnmol_latent_nsteps,
            pnmol_latent_rmse,
            marker="^",
            **style_pnmol_latent,
            **style_all,
        )

        ax_runtime.set_xlabel("Run time [s]")
        ax_runtime.loglog(mol_time, mol_rmse, **style_mol, **style_all)
        ax_runtime.loglog(
            pnmol_white_time,
            pnmol_white_rmse,
            marker="o",
            **style_pnmol_white,
            **style_all,
        )
        ax_runtime.loglog(
            pnmol_latent_time,
            pnmol_latent_rmse,
            marker="^",
            **style_pnmol_latent,
            **style_all,
        )

        ax_chi2.set_xlabel(r"Normalised $\chi^2$-statistic")
        ax_chi2.loglog(
            mol_chi2,
            mol_rmse,
            marker="d",
            markersize=5,
            label="MOL",
            color="gray",
            alpha=0.5,
            linestyle="-",
            linewidth=0.5,
        )
        ax_chi2.loglog(
            pnmol_white_chi2,
            pnmol_white_rmse,
            marker="o",
            markersize=5,
            label="PNMOL (white)",
            color="C1",
            linestyle="-",
            linewidth=0.5,
        )
        ax_chi2.loglog(
            pnmol_latent_chi2,
            pnmol_latent_rmse,
            marker="^",
            markersize=5,
            label="PNMOL (latent)",
            color="C0",
            linestyle="-",
            linewidth=0.5,
        )

    # ax_nsteps.set_xlim((1e0, 1e3))

    axes[0].set_ylabel("RMSE")
    axes[0].legend(
        loc="upper right",
        handlelength=3.5,
        fontsize="x-small",
        fancybox=False,
        edgecolor="black",
    ).get_frame().set_linewidth(0.5)

    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(
        by_label.values(),
        by_label.keys(),
        loc="center right",
        handlelength=2,
        fontsize="x-small",
        fancybox=False,
        edgecolor="black",
    ).get_frame().set_linewidth(0.5)
    axes[-1].set_xlim((1e-7, 1e7))

    axes[-1].annotate(
        "Underconfident", (5e-6, 1e0), color="gray", alpha=0.8, fontsize="small"
    )
    axes[-1].annotate(
        "Overconfident", (5e1, 2e0), color="gray", alpha=0.8, fontsize="small"
    )

    for ax in axes:
        ax.grid(which="minor", axis="y", linewidth=0.5, linestyle="dotted", alpha=0.75)

    ax_nsteps.set_title(
        r"$\bf a.$ " + "RMSE vs. Number of time-steps", loc="left", fontsize="medium"
    )
    ax_runtime.set_title(
        r"$\bf b.$ " + "RMSE vs. Run time", loc="left", fontsize="medium"
    )
    ax_chi2.set_title(
        r"$\bf c.$ " + "RMSE vs. Calibration", loc="left", fontsize="medium"
    )

    ax_chi2.axvline(1.0, color="gray", linewidth=1, alpha=0.8)
    plt.savefig("./experiments/results/figure4/figure.pdf", dpi=300)
    plt.show()
