from torch import Tensor, tensor, float64, int64


def compute_score(n_gaussians, img_size, block_size):
    block_size = 1 if block_size == "Naive" else block_size
    return (img_size ** 0.5) * (n_gaussians ** 0.25) / block_size


def optimal_n_blocks(n_gaussians: int, img_sizes: Tensor, ratio=11.0, device=None):
    n_blocks = img_sizes.sqrt().mul(n_gaussians ** 0.25).div(ratio)
    return n_blocks.to(dtype=int64, device=device).clamp(1)


def from_density(gaussians_per_pixel: float, img: Tensor, ratio=11.0, device=None):
    img_sizes = tensor(img.shape[:2], dtype=float64)
    n_gaussians = int(img_sizes.prod().item() * gaussians_per_pixel)
    n_blocks = optimal_n_blocks(n_gaussians, img_sizes, ratio, device)
    return n_gaussians, n_blocks
