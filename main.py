import os
from os import path as osp
from random import randint

import pandas as pd
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
import torchviz
import xarray as xr

from utils_fig import fig_multi, fig_and_save_metrics, fig_x_per_y
from utils_metrics import array_from_dict, flatten_xarray, save_2d_xr, PermuteBatchWrapper
from utils_train import train_loop, get_mean_time

from data_prep.downloader import load_data, dataset_profiles
from image.utils import coords_from_img, save_img, cvt_img
from model.gaussian import *
from model.init_params import from_density, compute_score


ROOT_OUT_PATH = os.getcwd()
RESULTS_PATH = osp.join(ROOT_OUT_PATH, "results")


device = "cpu"
#device = "cuda:0" if torch.cuda.is_available() else "cpu"


def measure_and_clear():
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak Allocated (MB): {peak_alloc}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return peak_alloc






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




def get_renderer_topk(k):
    if k == "Naive":
        return RendererNaive()
    
    if k == "Clamp":
        return RendererClamp()
    
    return RendererTopK(k=k)


def get_wrapper_tiles(splatter, renderer, block_size):
    if block_size == "Naive":
        return WrapperNaive(splatter, renderer)

    return WrapperTiledV1(splatter, renderer, (block_size, block_size))






def main_bench():
    key = "kodak"
    dataloader = load_data(key)
    #img = cvt_img(dataloader.__getitem__(randint(0, len(dataloader) - 1)), device)[:96, :96]
    img = cvt_img(dataloader.__getitem__(0), device)[:96, :96]

    splatters = [SplatterSigRot, SplatterCov]
    lrs = [0.025, 0.05, 0.1, 0.2]
    block_sizes = ["Naive", 6]
    ks = ["Naive", 10]

    metric_funcs = {
        "PSNR": PeakSignalNoiseRatio(1.0),
        "MS-SSIM": PermuteBatchWrapper(MultiScaleStructuralSimilarityIndexMeasure(True, 5)),
        "LPIPS": PermuteBatchWrapper(LearnedPerceptualImagePatchSimilarity()),
    }
    for func in metric_funcs:
        metric_funcs[func] = metric_funcs[func].to(device=device)


    for block_size in block_sizes:
        blocksize_root = sanit_join(RESULTS_PATH, "bench", f"bs_{block_size}")
        if device == "cuda:0":
            memory_usage = xr.DataArray(
                coords=[ks, [f"{splatter}" for splatter in splatters]],
                dims=["Ks", "Splatter"]
            )

        for k in ks:
            for splatter in splatters:

                metrics_per_lr = {}
                root = sanit_join(blocksize_root, f"k_{k}", splatter)

                for lr in lrs:
                    model = get_wrapper_tiles(
                        splatter(1000, 3, 0.2 / max(img.shape[:2])),
                        get_renderer_topk(k),
                        block_size
                    ).to(device=device)

                    local_root = sanit_join(root, lr, device)
                    local_img_root = root_folder(osp.join(local_root, "images"), "img")
                    save_img(f"{local_img_root}_gt", img)

                    metrics = train_loop(
                        model, local_img_root, img, 100, lr,
                        save_intervals=25,
                        metric_funcs=metric_funcs,
                    )
                    model.splatter.save_params(local_root, "params_final")

                    if device == "cuda:0":
                        memory_usage.loc[{"Ks": f"{k}", "Splatter": f"{splatter}"}] = measure_and_clear()

                    del model
                    fig_and_save_metrics(metrics, root_folder(local_root, "fig"), metric_funcs)
                    metrics_per_lr[lr] = xr.DataArray(metrics, dims=["epoch", "metric"])


                metrics_per_lr_arr = array_from_dict(metrics_per_lr, "lr")
                fig_root_global = root_folder(
                    sanit_join(root, "lr_all", device),
                    "fig"
                )

                fig_multi(
                    f"{fig_root_global}_time",
                    metrics_per_lr_arr.sel(metric="time"),
                    title="Time per Learning Rate",
                    xlabel="Epoch"
                )

                fig_multi(
                    f"{fig_root_global}_loss",
                    metrics_per_lr_arr.sel(metric="loss"),
                    title="Loss per Learning Rate",
                    xlabel="Epoch"
                )

                fig_x_per_y(
                    fig_root_global, metrics_per_lr_arr, "time", "loss",
                    title="Loss over Time\nper Learning Rate",
                    xlabel="Time"
                )

                for key in metric_funcs.keys():
                    fig_x_per_y(
                        fig_root_global, metrics_per_lr_arr, "time", key,
                        title=f"{key} over Time\nper Learning Rate",
                        xlabel="Time"
                    )

                metrics_per_lr_arr.to_netcdf(f"{fig_root_global}_metrics.h5")
                save_2d_xr(flatten_xarray(metrics_per_lr_arr, "metric", "lr", "metric_lr"), f"{fig_root_global}_metrics.csv")


        if device == "cuda:0":
            save_2d_xr(memory_usage, f"{blocksize_root}_memory_usage.csv")






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

        gaussians_per_pixel = 0.3 # ~4000 per 128x128 px
        n_gaussians, n_blocks = from_density(gaussians_per_pixel, img, 11, device=device) # 4000gs * 128 -> 16 blocks per side
        print(f"Blocks: {n_blocks}")

        model = WrapperTiledV1(
            SplatterCov(n_gaussians, 3, 0.2 / max(img.shape[:2])),
            RendererNaive(), #RendererTopK(k=10),
            n_blocks,
        ).to(device=device)

        root = sanit_join(RESULTS_PATH, "splat", f"{img.shape[0]}x{img.shape[1]}", model, key)
        img_root = root_folder(osp.join(root, "images"), "img")
        save_img(f"{img_root}_gt", img)

        print(f"\n\n\nTraining Single image: {key}_0")
        metrics = train_loop(model, img_root, img, 1000, 0.1, save_intervals=10, metric_funcs=metric_funcs)
        model.splatter.save_params(root, "params_final")

        fig_and_save_metrics(metrics, root_folder(root, "fig"), metric_funcs)

        if device == "cuda:0":
            with open(osp.join(root, "memory.txt"), "w") as file:
                file.write(f"Model:\n\n{model}\n\nMemory (MB): {measure_and_clear()}\n")

        del model






time_label = "Average Time Taken per Epoch"
block_label = "Block Count per Side"
square_label = "Image Size per Side"
topk_label = "Renderer Type (Top-K vs others)"
gs_label = "Gaussians"
perfplot_ylabel = "Time in Seconds"




def main_tiles_perfplot():
    key = list(dataset_profiles)[1]
    dataloader = load_data(key)

    root = sanit_join(RESULTS_PATH, "tiles_perfplot", device)
    gs_root = root_folder(sanit_join(root, "gaussians"), "fig")

    ns_gaussians = [250, 500, 1000, 2000, 4000, 8000, 16000]
    squares = [32, 64, 128, 256, 512]
    blocks = [2, 4, 6, 8, 12, 16, 20, 24, 32]

    max_bound = compute_score(4000, 512, 8)
    loss_fn = nn.L1Loss()
    times_global_dict = {}

    for n_gaussians in ns_gaussians:
        times = xr.DataArray(coords=[squares, blocks], dims=[square_label, block_label])

        for square in squares:
            gt = cvt_img(dataloader.__getitem__(0), device)[:square, :square]

            for block_size in blocks:
                if compute_score(n_gaussians, square, block_size) > max_bound:
                    print(f"Skipped: {n_gaussians:4} Gaussians, {square:3}x{square:3}px, {block_size}x{block_size} blocks")
                    continue

                print(f"Iteration: {n_gaussians:4} Gaussians, {square:3}x{square:3}px, {block_size}x{block_size} blocks")
                model = get_wrapper_tiles(
                    SplatterCov(n_gaussians, 3, 0.2 / max(gt.shape[:2])),
                    RendererNaive(),
                    block_size
                ).to(device=device)

                times.loc[{square_label: square, block_label: block_size}] = get_mean_time(model, loss_fn, gt, 10)
                del model


        save_2d_xr(times, f"{gs_root}_{n_gaussians}_gaussians_metrics.csv")
        fig_multi(
            f"{gs_root}_{n_gaussians}_gaussians",
            times,
            title=f"{time_label}\nper {block_label}\nover {square_label} (px)\nfor {n_gaussians} Gaussians",
            log=True,
            ylabel=perfplot_ylabel
        )

        times_global_dict[n_gaussians] = times

    fig_root = root_folder(root, "fig")
    times_global = array_from_dict(times_global_dict, gs_label)
    times_global_flat = flatten_xarray(times_global, gs_label, square_label, f"{gs_label}/{square_label}", ("gs", "px")).T

    save_2d_xr(times_global_flat, f"{fig_root}_all_metrics.csv")
    fig_multi(
        f"{fig_root}_all",
        times_global_flat,
        title=f"{time_label}\nper {block_label}\nover {square_label} (px)\nfor All Gaussians",
        log=True,
        ylabel=perfplot_ylabel
    )

    squares_root = root_folder(sanit_join(root, "squares"), "fig")
    for square, times in times_global.groupby(square_label):
        fig_multi(
            f"{squares_root}_{square}px",
            times.squeeze(square_label),
            title=f"{time_label}\nper {block_label}\nover {gs_label} (px)\nfor ${square} \\times {square}$px Images",
            log=True,
            ylabel=perfplot_ylabel
        )






def main_topk_perfplot():
    key = list(dataset_profiles)[1]
    dataloader = load_data(key)

    fig_root = root_folder(
        sanit_join(RESULTS_PATH, "topk_perfplot", device),
        "fig"
    )

    n_gaussians = 1000
    squares = [32, 64, 128, 256, 512]
    blocks = ["Naive", 1, 2, 4, 8, 16, 32]
    top_ks = [5, 10, 20, 40, "Naive", "Clamp"]

    max_bound = compute_score(n_gaussians, 512, 2) 
    loss_fn = nn.L1Loss()

    for square in squares:
        gt = cvt_img(dataloader.__getitem__(0), device)[:square, :square]
        times = xr.DataArray(coords=[top_ks, blocks], dims=[topk_label, block_label])

        for k in top_ks:
            for block_size in blocks:
                if compute_score(n_gaussians, square, block_size) > max_bound:
                    print(f"Skipped: {square:3}x{square:3}px, {block_size}x{block_size} blocks, {k}-K renderer")
                    continue

                print(f"Iteration: {square:3}x{square:3}px, {block_size}x{block_size} blocks, {k}-K renderer")

                model = get_wrapper_tiles(
                    SplatterCov(n_gaussians, 3, 0.2 / max(gt.shape[:2])),
                    get_renderer_topk(k),
                    block_size
                ).to(device=device)

                times.loc[{topk_label: str(k), block_label: str(block_size)}] = get_mean_time(model, loss_fn, gt, 10)
                del model

        save_2d_xr(times, f"{fig_root}_{square}_metrics.csv")
        fig_multi(
            f"{fig_root}_{square}",
            times,
            title=f"Average Time Taken per Epoch\nover Renderer Type (Top-K vs others)\nfor ${square} \\times {square}$px Images",
            log=True,
            ylabel=perfplot_ylabel
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
    root = sanit_join(RESULTS_PATH, "profile", key, device)
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
