# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import collections
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import binom
from torch import Tensor

from sbi.utils import conditional_pairplot as utils_conditional_pairplot
from sbi.utils import pairplot as utils_pairplot

try:
    collectionsAbc = collections.abc
except:
    collectionsAbc = collections


def pairplot(
    samples: Union[
        List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor
    ] = None,
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    limits: Optional[Union[List, torch.Tensor]] = None,
    subset: List[int] = None,
    upper: Optional[str] = "hist",
    diag: Optional[str] = "hist",
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Union[List, torch.Tensor] = [],
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    fig=None,
    axes=None,
    **kwargs,
):
    """Plot samples in a 2D grid showing marginals and pairwise marginals.

    Each of the diagonal plots can be interpreted as a 1D-marginal of the distribution
    that the samples were drawn from. Each upper-diagonal plot can be interpreted as a
    2D-marginal of the distribution.

    Args:
        samples: Samples used to build the histogram.
        points: List of additional points to scatter.
        limits: Array containing the plot xlim for each parameter dimension. If None,
            just use the min and max of the passed samples
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on).
        upper: Plotting style for upper diagonal, {hist, scatter, contour, cond, None}.
        diag: Plotting style for diagonal, {hist, cond, None}.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        points_colors: Colors of the `points`.
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot, see the source code in
            `_get_default_opts()` in `sbi.utils.plot` for more details.

    Returns: figure and axis of posterior distribution plot
    """
    return utils_pairplot(
        samples=samples,
        points=points,
        limits=limits,
        subset=subset,
        upper=upper,
        diag=diag,
        figsize=figsize,
        labels=labels,
        ticks=ticks,
        points_colors=points_colors,
        warn_about_deprecation=False,
        fig=fig,
        axes=axes,
        **kwargs,
    )


def conditional_pairplot(
    density: Any,
    condition: torch.Tensor,
    limits: Union[List, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    subset: List[int] = None,
    resolution: int = 50,
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Union[List, torch.Tensor] = [],
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    fig=None,
    axes=None,
    **kwargs,
):
    r"""Plot conditional distribution given all other parameters.

    The conditionals can be interpreted as slices through the `density` at a location
    given by `condition`.

    For example:
    Say we have a 3D density with parameters $\theta_0$, $\theta_1$, $\theta_2$ and
    a condition $c$ passed by the user in the `condition` argument.
    For the plot of $\theta_0$ on the diagonal, this will plot the conditional
    $p(\theta_0 | \theta_1=c[1], \theta_2=c[2])$. For the upper
    diagonal of $\theta_1$ and $\theta_2$, it will plot
    $p(\theta_1, \theta_2 | \theta_0=c[0])$. All other diagonals and upper-diagonals
    are built in the corresponding way.

    Args:
        density: Probability density with a `log_prob()` method.
        condition: Condition that all but the one/two regarded parameters are fixed to.
            The condition should be of shape (1, dim_theta), i.e. it could e.g. be
            a sample from the posterior distribution.
        limits: Limits in between which each parameter will be evaluated.
        points: Additional points to scatter.
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on)
        resolution: Resolution of the grid at which we evaluate the `pdf`.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        points_colors: Colors of the `points`.
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot, see the source code in
            `_get_default_opts()` in `sbi.utils.plot` for more details.

    Returns: figure and axis of posterior distribution plot
    """
    return utils_conditional_pairplot(
        density=density,
        condition=condition,
        limits=limits,
        points=points,
        subset=subset,
        resolution=resolution,
        figsize=figsize,
        labels=labels,
        ticks=ticks,
        points_colors=points_colors,
        warn_about_deprecation=False,
        fig=fig,
        axes=axes,
        **kwargs,
    )


def sbc_rank_plot(
    ranks,
    num_bins=100,
    num_repeats=50,
    parameter_labels=None,
    ranks_labels=None,
    colors=None,
    line_alpha=0.8,
    show_uniform_region=True,
    uniform_region_alpha=0.2,
    fig=None,
    ax=None,
    figsize=None,
):
    """Plot simulation-based calibration ranks as empirical CDFs.

    Args:
        ranks: Tensor of ranks to be plotted, or list of Tensors when comparing several sets
            of ranks, e.g., set of ranks obtained from different methods.
        num_bins: number of bins used for calculating empirical cdfs via histograms.
        num_repeats: number of repeats for each empirical CDF step (resolution).
        parameter_labels: list of labels for each parameter dimension.
        ranks_labels: list of labels for each set of ranks.
        colors: list of colors for each parameter dimension, or each set of ranks.
        line_alpha: alpha for cdf lines
        show_uniform_region: whether to plot the region showing the cdfs expected under uniformity
        uniform_region_alpha: alpha for region showing the cdfs expected under uniformity.
        fig: figure object to plot in.
        ax: axis object, must have shape (number of set of ranks,) or be a raw Axes object.
        figsize: dimensions of figure object, defaults to (8, 5) or (number of set of ranks * 4, 5).

    Returns:
        fig, ax: figure and axis objects.

    """

    if isinstance(ranks, (Tensor, np.ndarray)):
        ranks = [ranks]
    else:
        assert isinstance(ranks, List)

    num_sbc_runs, num_parameters = ranks[0].shape
    num_ranks = len(ranks)

    for ranki in ranks:
        assert (
            ranki.shape == ranks[0].shape
        ), "all ranks in list must have the same shape."

    if figsize is None:
        figsize = (num_parameters * 4, 5) if num_ranks > 1 else (8, 5)

    if parameter_labels is None:
        parameter_labels = [f"dim {i+1}" for i in range(num_parameters)]
    if ranks_labels is None:
        ranks_labels = [f"rank set {i+1}" for i in range(num_ranks)]

    # Plot one row subplot for each parameter, different "methods" in each subplot.
    if num_ranks > 1:
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, num_parameters, figsize=figsize)
        else:
            assert ax.shape == (
                num_parameters,
            ), "Expecting subplots of shape 1, num_parameters."

        for ii, ranki in enumerate(ranks):
            for jj in range(num_parameters):
                plt.sca(ax[jj])

                plot_ranks_as_cdfs(
                    ranki[:, jj],
                    num_bins,
                    num_repeats,
                    label=ranks_labels[ii],
                    color=f"C{ii}" if colors is None else colors[ii],
                    xlabel=f"posterior rank {parameter_labels[jj]}",
                    # Show legend and ylabel only in first subplot.
                    show_ylabel=jj == 0,
                    show_legend=jj == 0,
                    alpha=line_alpha,
                )
                if ii == 0 and show_uniform_region:
                    plot_cdf_region_expected_under_uniformity(
                        num_sbc_runs, num_bins, num_repeats, alpha=0.1
                    )

    # When there is only one set of ranks show all params in a single subplot.
    else:
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plt.sca(ax)
        ranki = ranks[0]
        for jj in range(num_parameters):
            plot_ranks_as_cdfs(
                ranki[:, jj],
                num_bins,
                num_repeats,
                label=parameter_labels[jj],
                color=f"C{jj}" if colors is None else colors[jj],
                xlabel=f"posterior rank",
                # Plot ylabel and legend at last.
                show_ylabel=jj == (num_parameters - 1),
                show_legend=jj == (num_parameters - 1),
                alpha=line_alpha,
            )
        if show_uniform_region:
            plot_cdf_region_expected_under_uniformity(
                num_sbc_runs, num_bins, num_repeats, alpha=uniform_region_alpha
            )

    return fig, ax


def plot_ranks_as_cdfs(
    ranks,
    num_bins,
    num_repeats,
    label=None,
    color=None,
    alpha: float = 0.8,
    xlabel=None,
    show_ylabel=False,
    show_legend=False,
    num_ticks: int = 3,
    legend_kwargs=dict(),
) -> None:
    """Plot ranks as empirical CDFs.

    Args:
        ranks:
        num_bins:
        num_repeats:
        label:
        color:
        alpha:
        xlabel:
        show_ylabel:
        show_legend:
        num_ticks:
        legend_kwargs:

    """
    # Generate histogram of ranks.
    hist, *_ = np.histogram(ranks, bins=num_bins, density=False)
    # Construct empirical CDF.
    histcs = hist.cumsum()
    # Plot cdf and repeat each stair step
    plt.plot(
        np.linspace(0, num_bins, num_repeats * num_bins),
        np.repeat(histcs / histcs.max(), num_repeats),
        label=label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.yticks(np.linspace(0, 1, 3))
        plt.ylabel("empirical CDF")
    else:
        # Plot ticks only
        plt.yticks(np.linspace(0, 1, 3), [])
    if show_legend and label:
        plt.legend(loc=2, handlelength=0.8, **legend_kwargs)

    plt.ylim(0, 1)
    plt.xlim(0, num_bins)
    plt.xticks(np.linspace(0, num_bins, num_ticks))
    plt.xlabel("posterior rank" if xlabel is None else xlabel)


def plot_cdf_region_expected_under_uniformity(
    num_sbc_runs, num_bins, num_repeats, alpha: float = 0.1, color="grey"
):
    """Plot region of empirical cdfs expected under uniformity."""

    # Construct uniform histogram.
    uni_bins = binom(num_sbc_runs, p=1 / num_bins).ppf(0.5) * np.ones(num_bins)
    uni_bins_cdf = uni_bins.cumsum() / uni_bins.sum()
    # Decrease value one in last entry by epsilon to find valid
    # confidence intervals.
    uni_bins_cdf[-1] -= 1e-9

    lower = [binom(num_sbc_runs, p=p).ppf(0.005) for p in uni_bins_cdf]
    upper = [binom(num_sbc_runs, p=p).ppf(0.995) for p in uni_bins_cdf]

    # Plot grey area with expected ECDF.
    plt.fill_between(
        x=np.linspace(0, num_bins, num_repeats * num_bins),
        y1=np.repeat(lower / np.max(lower), num_repeats),
        y2=np.repeat(upper / np.max(upper), num_repeats),
        color=color,
        alpha=alpha,
    )
