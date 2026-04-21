from matplotlib import pyplot as plt
import scienceplots # Keep this if using "science" style
import xarray as xr


plt.style.use("science")




def init_fig(log=False):
    fig = plt.figure()
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


def fig_single(path, x, y, title=None, dpi=300, log=False, xlabel=None, ylabel=None):
    fig, ax = init_fig(log)
    ax.plot(x, y)
    finish_and_save(path, fig, ax, title, dpi, xlabel, ylabel)


def fig_multi(path, xy, title=None, dpi=300, log=False, xlabel=None, ylabel=None):
    fig, ax = init_fig(log)

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
