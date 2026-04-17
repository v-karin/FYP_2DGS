from matplotlib import pyplot as plt
import xarray as xr




def init_fig(title, log=False):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])

    if len(title) > 0: ax.set_title(title)
    if log: ax.set_yscale("log")
    ax.grid(which="major", color="0.8")
    ax.grid(which="minor", color="0.9")

    return fig, ax


def fig_single(path, x, y, title="", dpi=300, log=False):
    fig, ax = init_fig(title, log)
    ax.plot(x, y)
    fig.savefig(f"{path}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_multi(path, xy, title="", dpi=300, log=False):
    fig, ax = init_fig(title, log)

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

    fig.savefig(f"{path}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
