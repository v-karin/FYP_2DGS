from matplotlib import pyplot as plt
import scienceplots # Keep this if using "science" style
import xarray as xr

from utils_metrics import prepare_metric_xy


plt.style.use("science")




def init_fig(figsize, log=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])

    if log: ax.set_yscale("log")
    ax.grid(which="major", color="0.8")
    ax.grid(which="minor", color="0.9")

    return fig, ax


def finish_and_save(path, fig, ax, title, dpi, xlabel, ylabel):
    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    fig.savefig(f"{path}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_single(path, x, y, title=None, dpi=300, log=False, xlabel=None, ylabel=None, figsize=(5.906, 4.176)):
    fig, ax = init_fig(figsize, log)
    ax.plot(x, y)
    finish_and_save(path, fig, ax, title, dpi, xlabel, ylabel)


def fig_multi(path, xy, title=None, dpi=300, log=False, xlabel=None, ylabel=None, figsize=(5.906, 4.176)):
    fig, ax = init_fig(figsize, log)

    if isinstance(xy, tuple):
        xs, ys = xy[0], xy[1]
        if isinstance(ys, dict):
            for x, y, key in zip(xs, ys.values(), ys.keys()):
                ax.plot(x, y, label=f"{key}")

        elif isinstance(ys, list):
            for x, y in zip(xs, ys):
                ax.plot(x, y)

        ax.legend()

    elif isinstance(xy, xr.DataArray):
        xy.plot.line(ax=ax, hue=xy.dims[0])

    else:
        raise ValueError(ys)

    finish_and_save(path, fig, ax, title, dpi, xlabel, ylabel)


time_axis_label = "Time (Seconds)"


def fig_and_save_metrics(metrics, fig_root, metric_funcs):
    metrics.to_csv(f"{fig_root}_metrics.csv", sep=";")

    fig_single(f"{fig_root}_time", metrics["time"].index, metrics["time"], title="Time per Epoch", xlabel="Epoch")
    fig_single(f"{fig_root}_loss", metrics["loss"].index, metrics["loss"], title="Loss per Epoch", xlabel="Epoch")
    fig_single(f"{fig_root}_loss_per_time", metrics["time"], metrics["loss"], title="Loss over Time", xlabel=time_axis_label)
    for key in metric_funcs.keys():
        fig_single(f"{fig_root}_{key}_per_time", metrics["time"], metrics[key], title=f"{key} over Time", xlabel=time_axis_label)


def fig_x_per_y(fig_root: str, metrics: xr.DataArray, x_dim: str, y_dim: str, **kwargs):
    fig_multi(
        f"{fig_root}_{y_dim}_per_{x_dim}",
        prepare_metric_xy(metrics, x_dim, y_dim),
        **kwargs
    )
