from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torchvision.utils import save_image


def coords_from_img(img: Tensor):
    shape = img.shape
    x = torch.linspace(0, 1, shape[0], device=img.device)
    y = torch.linspace(0, 1, shape[1], device=img.device)
    return torch.stack(torch.meshgrid(x, y, indexing="ij"), -1)


def fig_img(name, data: Tensor):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    ax.imshow(data.cpu().numpy(force=True))

    fig.savefig(f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def save_img(name, data: Tensor):
    save_image(data.permute(2, 0, 1), f"{name}.png")


def cvt_img(img: torch.Tensor, device):
    return img.to(dtype=torch.float32, device=device).permute(1, 2, 0) / 256
