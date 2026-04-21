import os
from os import path as osp
from random import randint

import pandas as pd
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
import torchviz
import xarray as xr

from utils_fig import fig_single, fig_multi
from utils_train import train_loop, time_single

from data_prep.downloader import load_data, dataset_profiles
from image.utils import coords_from_img, save_img, cvt_img
from model.gaussian import *


ROOT_OUT_PATH = os.getcwd()
RESULTS_PATH = osp.join(ROOT_OUT_PATH, "results")


device = "cpu"
#device = "cuda:0" if torch.cuda.is_available() else "cpu"




def root_folder(folder: str, attach: str):
    root = osp.join(folder, attach)
    os.makedirs(folder, exist_ok=True)
    return root


def sanit_join(root: str, *values: str):
    return osp.join(root, *[
        f"{value}"
        .replace(":", "_")
        .replace("\n", "")
        .replace(" ", "")
        for value in values
    ])


def fig_and_save_metrics(metrics, fig_root, metric_funcs):
    metrics.to_csv(f"{fig_root}_metrics.csv", sep=";")

    fig_single(f"{fig_root}_time", metrics["time"].index, metrics["time"], title="Time per Epoch")
    fig_single(f"{fig_root}_loss", metrics["loss"].index, metrics["loss"], title="Loss per Epoch")
    fig_single(f"{fig_root}_loss_per_time", metrics["time"], metrics["loss"], title="Loss over Time")
    for key in metric_funcs.keys():
        fig_single(f"{fig_root}_{key}_per_time", metrics["time"], metrics[key], title=f"{key} over Time")


def prepare_metric_xy(metrics: xr.DataArray, x_dim: str, y_dim: str):
    return (
        metrics.sel(metric=y_dim)
        .assign_coords(epoch=metrics.sel(metric=x_dim))
        .rename(epoch=x_dim)
    )


def fig_x_per_y(fig_root: str, metrics: xr.DataArray, x_dim: str, y_dim: str, **kwargs):
    fig_multi(
        f"{fig_root}_{y_dim}_per_{x_dim}",
        prepare_metric_xy(metrics, x_dim, y_dim),
        **kwargs
    )




class PermuteBatchWrapper(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
    
    def forward(self, pred: Tensor, gt: Tensor):
        return self.metric(
            pred.permute(2, 0, 1).unsqueeze(0).clamp(0, 1),
            gt.permute(2, 0, 1).unsqueeze(0)
        )






def main_bench():
    key = "kodak"
    dataloader = load_data(key)
    #img = cvt_img(dataloader.__getitem__(randint(0, len(dataloader) - 1)), device)[:96, :96]
    img = cvt_img(dataloader.__getitem__(0), device)[:96, :96]

    splatters = [SplatterSigRot, SplatterCov]
    lrs = [0.025, 0.05, 0.1, 0.2]
    metric_funcs = {
        "PSNR": PeakSignalNoiseRatio(1.0),
        "MS-SSIM": PermuteBatchWrapper(MultiScaleStructuralSimilarityIndexMeasure(True, 5)),
        "LPIPS": PermuteBatchWrapper(LearnedPerceptualImagePatchSimilarity()),
    }
    for func in metric_funcs:
        metric_funcs[func] = metric_funcs[func].to(device=device)

    for splatter in splatters:
        metrics_per_lr = {}

        for lr in lrs:
            model = WrapperTiledV1(
                splatter(400, 3, 0.2 / max(img.shape[:2])),
                RendererNaive(),
                (4, 4),
            ).to(device=device)
            model_name = f"{model}"

            root = sanit_join(RESULTS_PATH, "bench", model, lr, device)
            img_root = root_folder(osp.join(root, "images"), "img")

            save_img(f"{img_root}_gt", img)

            metrics = train_loop(
                model, img_root, img, 50, lr,
                save_intervals=25,
                metric_funcs=metric_funcs,
            )
            model.splatter.save_params(root, "params_final")
            del model

            fig_and_save_metrics(metrics, root_folder(root, "fig"), metric_funcs)
            metrics_per_lr[lr] = xr.DataArray(metrics, dims=["epoch", "metric"])


        metrics_per_lr_arr = xr.Dataset(metrics_per_lr).to_dataarray("lr")
        fig_root_global = root_folder(
            sanit_join(RESULTS_PATH, "bench", model_name, "lr_all", device),
            "fig"
        )

        fig_multi(
            f"{fig_root_global}_time",
            metrics_per_lr_arr.sel(metric="time"),
            title="Time per Learning Rate"
        )

        fig_multi(
            f"{fig_root_global}_loss",
            metrics_per_lr_arr.sel(metric="loss"),
            title="Loss per Learning Rate"
        )

        fig_x_per_y(
            fig_root_global, metrics_per_lr_arr, "time", "loss",
            title="Loss over Time per Learning Rate"
        )

        for key in metric_funcs.keys():
            fig_x_per_y(
                fig_root_global, metrics_per_lr_arr, "time", key,
                title=f"{key} over Time per Learning Rate"
            )

        metrics_per_lr_arr.to_netcdf(f"{fig_root_global}_metrics.h5")




def optimal_n_blocks(n_gaussians: int, img: Tensor, gaussians_sqrt_pixels_per_block: float):
    img_sizes = torch.tensor(img.shape[:2], dtype=torch.float64)
    n_blocks = img_sizes.sqrt().mul(n_gaussians).div(gaussians_sqrt_pixels_per_block)
    return n_blocks.to(dtype=torch.int64, device=device).clamp(1)


def main_example():
    for key in dataset_profiles:
        metric_funcs = {
            "PSNR": PeakSignalNoiseRatio(1.0),
            "MS-SSIM": PermuteBatchWrapper(MultiScaleStructuralSimilarityIndexMeasure(True, 5)),
            "LPIPS": PermuteBatchWrapper(LearnedPerceptualImagePatchSimilarity()),
        }
        for func in metric_funcs:
            metric_funcs[func] = metric_funcs[func].to(device=device)

        dataloader = load_data(key)
        img = cvt_img(dataloader.__getitem__(0), device)
        print(f"Image Dimensions: {img.shape}")

        n_gaussians = 16000
        n_blocks = optimal_n_blocks(n_gaussians, img, 3000)
        print(f"Blocks: {n_blocks}")

        model = WrapperTiledV1(
            SplatterCov(n_gaussians, 3, 0.2 / max(img.shape[:2])),
            RendererNaive(), #RendererTopK(k=10),
            (16, 16),
        ).to(device=device)

        root = sanit_join(RESULTS_PATH, "splat", model, key)
        img_root = root_folder(osp.join(root, "images"), "img")
        save_img(f"{img_root}_gt", img)

        print(f"\n\n\nTraining Single image: {key}_0")
        metrics = train_loop(model, img_root, img, 1000, 0.1, save_intervals=10, metric_funcs=metric_funcs)
        model.splatter.save_params(root, "params_final")

        fig_and_save_metrics(metrics, root_folder(root, "fig"), metric_funcs)





def flat_from_dict(d: dict, dim_x, dim_new, idx_suffix=("", "")):
    return flatten_xarray(xr.Dataset(d).to_array(dim="intermediate"), "intermediate", dim_x, dim_new, idx_suffix)


def flatten_xarray(arr: xr.DataArray, dim0, dim1, dim_new, idx_suffix=("", "")):
    arr_flat = arr.stack(**{dim_new: (dim0, dim1)})
    idx_flat = [f"{idx[0]}{idx_suffix[0]}_{idx[1]}{idx_suffix[1]}" for idx in arr_flat.indexes[dim_new]]
    return arr_flat.assign_coords(**{dim_new: idx_flat})


def get_mean_time(model, loss_fn, gt, iters: int):
    time_spent_list = [time_single(model, loss_fn, gt) for i in range(iters)]
    time_spent = sum(time_spent_list) / len(time_spent_list)
    print("Time:", time_spent)

    return time_spent


def get_wrapper_tiles_perfplot(splatter, renderer, block_size):
    if block_size == "Naive":
        return WrapperNaive(splatter, renderer)

    return WrapperTiledV1(splatter, renderer, (block_size, block_size))


def compute_score(n_gaussians, img_size, block_size):
    return n_gaussians * img_size / (block_size * block_size)


def main_tiles_perfplot():
    key = list(dataset_profiles)[0]
    dataloader = load_data(key)

    fig_root = root_folder(
        sanit_join(RESULTS_PATH, "tiles_perfplot", device),
        "fig"
    )

    ns_gaussians = [250, 500, 1000, 2000, 4000]
    squares = [32, 64, 128, 256, 512, 768]
    blocks = [2, 4, 6, 8, 12, 16, 20, 24, 32]

    max_bound = compute_score(1000, 256, 4)

    loss_fn = nn.L1Loss()
    times_global = {}

    for n_gaussians in ns_gaussians:
        times = xr.DataArray(coords=[squares, blocks], dims=["square", "block"])

        for square in squares:
            gt = cvt_img(dataloader.__getitem__(0), device)[:square, :square]

            for block_size in blocks:
                if compute_score(n_gaussians, square, block_size) > max_bound:
                    print(f"Skipped: {n_gaussians:4} Gaussians, {square:3}x{square:3}px, {block_size}x{block_size} blocks")
                    continue

                print(f"Iteration: {n_gaussians:4} Gaussians, {square:3}x{square:3}px, {block_size}x{block_size} blocks")
                model = get_wrapper_tiles_perfplot(
                    SplatterCov(n_gaussians, 3, 0.2 / max(gt.shape[:2])),
                    RendererNaive(),
                    block_size
                ).to(device=device)

                times.loc[{"square": square, "block": block_size}] = get_mean_time(model, loss_fn, gt, 10)
                del model


        pd.DataFrame(times).to_csv(f"{fig_root}_{n_gaussians}_gaussians_metrics.csv", sep=";")
        fig_multi(
            f"{fig_root}_{n_gaussians}_gaussians",
            times,
            log=True
        )

        times_global[n_gaussians] = times

    times_global_flat = flat_from_dict(times_global, "square", "gs_sq", ("gs", "px2")).T
    pd.DataFrame(times_global_flat).to_csv(f"{fig_root}_all_metrics.csv", sep=";")
    fig_multi(
        f"{fig_root}_all",
        times_global_flat,
        log=True
    )




def get_renderer_topk_perfplot(k):
    if k == "Naive":
        return RendererNaive()
    
    if k == "Clamp":
        return RendererClamp()
    
    return RendererTopK(k=k)


def main_topk_perfplot():
    key = list(dataset_profiles)[0]
    dataloader = load_data(key)

    fig_root = root_folder(
        sanit_join(RESULTS_PATH, "topk_perfplot", device),
        "fig"
    )

    n_gaussians = 1000
    squares = [32, 64, 128, 256, 512]
    blocks = [2, 4, 8, 16, 32]
    top_ks = [5, 10, 20, 40, "Naive", "Clamp"]

    loss_fn = nn.L1Loss()

    for square in squares:
        gt = cvt_img(dataloader.__getitem__(0), device)[:square, :square]
        times = {}

        for k in top_ks:
            times[k] = []

            for block_size in blocks:
                block = (block_size, block_size)

                print("\nIteration:", square, k, block_size)
                model = WrapperTiledV1(
                    SplatterCov(n_gaussians, 3, 0.2 / max(gt.shape[:2])),
                    get_renderer_topk_perfplot(k),
                    block
                ).to(device=device)

                times[k].append(get_mean_time(model, loss_fn, gt, 10))
                del model

        pd.DataFrame(times).to_csv(f"{fig_root}_{square}_metrics.csv", sep=";")
        fig_multi(
            f"{fig_root}_{square}",
            ([blocks for key in top_ks], times),
            title=f"Average Time Taken per Epoch\nover Renderer Type (Top-K vs others)\nfor ${square} \\times {square}$px Images",
            log=True
        )




def main_torch_profile():
    print("\n\nSetting up Profile")
    key = list(dataset_profiles)[0]

    dataloader = load_data(key)
    img = cvt_img(dataloader.__getitem__(0), device)[:128, :128]

    n_gaussians = 1000
    model = WrapperTiledV1(
        SplatterCov(n_gaussians, 3, 0.2 / max(img.shape[:2])),
        RendererNaive(), #RendererTopK(k=10),
        (4, 4),
    ).to(device=device)

    loss_fn = nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), 0.1)
    coords = coords_from_img(img)

    prof = profile(
        activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda:0" else []),
        record_shapes=True,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) # even though unofficial, still needs listing


    print("\nRunning and profiling")
    with prof:
        with record_function("fwd_loss"):
            y = model(coords)
            loss = loss_fn(y, img)
            optim.zero_grad()

        with record_function("bwd_step"):
            loss.backward(retain_graph=True)
            optim.step()


    print("\nWriting profiler logs")
    root = osp.join(RESULTS_PATH, "profile", key)
    prof.export_chrome_trace(root_folder(root, "trace.json"))
    prof.export_stacks(osp.join(root, "stacks.txt"))

    log = prof.key_averages(True, 20).table(
        "self_cuda_time_total", 40,
        75, 100, 80
    )

    with open(osp.join(root, "log.txt"), "w") as file:
        file.write(log)

    torchviz.make_dot(
        loss,
        dict(model.named_parameters()),
        True,
        True,
        100
    ).render("graph.gv", root, format="png", renderer="gd")




def main():
    main_example()
    main_bench()
    main_tiles_perfplot()
    main_topk_perfplot()
    main_torch_profile()


main()
