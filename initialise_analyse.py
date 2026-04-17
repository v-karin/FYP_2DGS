from image.utils import coords_from_img, fig_img
from model.gaussian import *
from matplotlib import pyplot as plt
import os

def get_initial(n_gaussians, square_size):
    img = torch.rand(square_size, square_size, 3)

    model = WrapperTiledV1(
        SplatterCov(n_gaussians, 3, 0.2 / max(img.shape[:2])),
        RendererNaive(), #RendererTopK(k=10),
        (4, 4),
    )

    coords = coords_from_img(img)
    y = model(coords)

    return y, model

def gen_save_image(n_gaussians, square_size):
    y, model = get_initial(n_gaussians, square_size)

    new_path = os.path.join("results", "init_analysis", f"{square_size}px", f"{model.splatter}")
    os.makedirs(new_path, exist_ok=True)

    fig_img(os.path.join(new_path, f"init_clamped"), y.clamp(0, 1))
    fig_img(os.path.join(new_path, f"init_unbounded"), y / y.max())

for n_g in [100, 200, 500, 1000]:
    for sq in [64, 128, 256]:
        gen_save_image(n_g, sq)