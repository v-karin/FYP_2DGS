import time

import pandas as pd
import torch
from torch import Tensor, nn

from image.utils import coords_from_img, save_img
from model.gaussian import WrapperNaive




def print_min_max(name: str, value: Tensor):
    print(f"{name} (min, max): {value.amin().item()}, {value.amax().item()}")




def loop_single(model, coords, loss_fn, gt, optim):
    y = model(coords)
    loss = loss_fn(y, gt)
    optim.zero_grad()

    print("Backwarding..")
    loss.backward()
    optim.step()

    return y, loss


def time_single(model, loss_fn, gt):
    coords = coords_from_img(gt[..., 0])
    optim = torch.optim.Adam(model.parameters(), 0.1)

    ot = time.monotonic_ns()
    loop_single(model, coords, loss_fn, gt, optim)
    return (time.monotonic_ns() - ot) / 1e9


def get_mean_time(model, loss_fn, gt, iters: int):
    time_spent_list = [time_single(model, loss_fn, gt) for i in range(iters)]
    time_spent = sum(time_spent_list) / len(time_spent_list)
    print("Time:", time_spent)

    return time_spent






def train_loop(
        model: WrapperNaive, img_root: str, gt: Tensor,
        epochs: int, lr: float,
        save_intervals=0, save_final=True,
        metric_funcs={}
    ):

    coords = coords_from_img(gt[..., 0])

    loss_fn = nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr)

    metrics = {key: [] for key in metric_funcs}
    metrics["loss"] = []
    metrics["time_ns"] = []


    for i in range(epochs):
        print("\n\nEpoch:", i)

        ot = time.monotonic_ns()
        y, loss = loop_single(model, coords, loss_fn, gt, optim)

        metrics["time_ns"].append(time.monotonic_ns() - ot)
        metrics["loss"].append(loss.item())
        for key, func in metric_funcs.items():
            metrics[key].append(func(y, gt).item())

        print(f"Loss: {loss.item()}")
        if save_intervals > 0:
            if (i % save_intervals) == 0:
                save_img(f"{img_root}_train_{i}", y.clamp(0, 1))


    with torch.no_grad():
        y = model(coords)
        loss = loss_fn(y, gt)

        metrics["time_ns"].append(metrics["time_ns"][-1])
        metrics["loss"].append(loss.item())
        for key, func in metric_funcs.items():
            metrics[key].append(func(y, gt).item())

        print(f"Loss: {loss.item()}")
        if save_final:
            save_img(f"{img_root}_pred", y.clamp(0, 1))

        print("\n\nDiagnosing Gaussians...")
        print_min_max("Mu", model.mus)
        print_min_max("Covariance", model.covs)
        print_min_max("Colour", model.cols)

    metrics_df = pd.DataFrame(metrics)
    metrics_df["time"] = metrics_df["time_ns"].cumsum() * 1e-9

    return metrics_df
