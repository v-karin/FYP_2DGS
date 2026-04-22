import math
import os

import numpy as np
import torch
from torch import Tensor, nn

from .utils import new_rot_mat


# MOVE preliminary math to "Preliminary Notation" section in document
# Move the math behind various papers to notes document




# Inspiration: Image-GS, Fast-2DGS
class SplatterBase(nn.Module):
    def __init__(self, n_gaussians: int, n_colours=3, min_sig=0.001, **init_params):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.min_sig = min_sig

        self._mus = nn.Parameter(init_params.get("_mus") or torch.rand(1, n_gaussians, 2).logit(0.0001))
        self.cols = nn.Parameter(init_params.get("cols") or torch.rand(1, n_gaussians, n_colours) * 0.25)

        self._scale_offset = -2
        self._eps = 1e-6
        self._k = 10

        self._i = [0.8, 0.4]


    @property
    def mus(self):
        return self._mus.sigmoid()

    @property
    def covs(self) -> Tensor:
        raise NotImplementedError


    def export_params(self):
        raise NotImplementedError


    def save_params(self, folder: str, name: str):
        os.makedirs(folder, exist_ok=True)
        path = f"{os.path.join(folder, name)}.npz"
        np.savez_compressed(path, **self.export_params())
        return path




class SplatterSigRot(SplatterBase):
    def __init__(self, n_gaussians: int, n_colours=3, min_sig=0.001, **init_params):
        """
        n_colours: Number of channels in image

        n_gaussians: Number of Gaussians to initialise

        min_sig: Minimum sizes of Gaussians (w.r.t. image bounds)
        """
        super().__init__(n_gaussians, n_colours, min_sig, **init_params)

        self.rots = nn.Parameter(init_params.get("rots") or torch.rand(n_gaussians).mul(math.tau))
        self._sigs = nn.Parameter(init_params.get("_sigs") or (
            torch.randn(n_gaussians, 2, 1) * self._i[0]
            + torch.randn(n_gaussians, 1, 1) * self._i[1]
            + (self._scale_offset - math.log(n_gaussians) / 2)
        )) # likely to do w/ Normal dist. e^x^2 vs logistic e^-x
        self._sigdiag = nn.Parameter(torch.eye(2).unsqueeze(0), requires_grad=False)


    @property
    def sigs(self):
        return self._sigs.sigmoid().mul(2).add(self.min_sig) # max size: 2 x image

    @property
    def covs(self):
        sig_mat = self._sigdiag * self.sigs
        rot_mat = new_rot_mat(self.rots)
        L = rot_mat @ sig_mat # (N, 2, 2)
        covs = (L @ L.mT).unsqueeze(0) # (1, N, 2, 2)
        return covs


    def export_params(self):
        params = {
            "type": "sigrot",
            "_mus": self._mus.numpy(force=True),
            "cols": self.cols.numpy(force=True),
            "rots": self.rots.numpy(force=True),
            "_sigs": self._sigs.numpy(force=True)
        }
        return params


    def __repr__(self):
        return f"SigRot({self.n_gaussians}gs_{self.cols.shape[-1]}cols_ms{self.min_sig})"




class SplatterCov(SplatterBase):
    def __init__(self, n_gaussians: int, n_colours=3, min_sig=0.001, **init_params):
        """
        n_colours: Number of channels in image

        n_gaussians: Number of Gaussians to initialise

        min_sig: Minimum sizes of Gaussians (w.r.t. image bounds)
        """
        super().__init__(n_gaussians, n_colours, min_sig, **init_params)

        self ._acd = nn.Parameter(init_params.get("_acd") or (
            torch.randn(n_gaussians, 3) # a, c, d
            * torch.tensor([self._i[0], self._i[0], 2]).unsqueeze(0)
            + torch.randn(n_gaussians, 1)
            * torch.tensor([self._i[1], self._i[1], 0]).unsqueeze(0)
            + torch.tensor([1, 1, 0]).unsqueeze(0) * (self._scale_offset - math.log(n_gaussians) / 2)
        ))
        self._ac_ind = nn.Parameter(torch.tensor([0, 1]), requires_grad=False)
        self._id2 = nn.Parameter(torch.eye(2).unsqueeze(0), requires_grad=False)


    @property
    def covs(self):
        ac = self._acd.index_select(1, self._ac_ind).sigmoid().mul(2).add(self.min_sig).square() # (N, 2)
        b = self._acd.select(1, 2).unsqueeze(-1).div(2).tanh().mul(ac.prod(1, keepdim=True).sqrt()) # (N)
        covs = self._id2 * ac.unsqueeze(-1) + self._id2.flip(-1) * b.unsqueeze(-1)
        return covs.unsqueeze(0) # (1, N, 2, 2)


    def export_params(self):
        params = {
            "type": "cov",
            "_mus": self._mus.numpy(force=True),
            "cols": self.cols.numpy(force=True),
            "_acd": self._acd.numpy(force=True)
        }
        return params


    def __repr__(self):
        return f"Cov({self.n_gaussians}gs_{self.cols.shape[-1]}cols_ms{self.min_sig})"






class RendererNaive(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self._eps = eps
    

    def _render_gaussians(self, x: Tensor, mus: Tensor, covs: Tensor):
        d = (x.unsqueeze(1) - mus).unsqueeze(-1) # (D, N, 2, 1)
        return d.mT.matmul(torch.linalg.solve(covs, d)).squeeze(-1).mul(-0.5).exp() # (D, N, 1) - d^T \Sigma^-1 d


    def forward(self, x: Tensor, mus: Tensor, covs: Tensor, cols: Tensor):
        raw = self._render_gaussians(x, mus, covs).mul(cols).sum(-2) # (D, 3)
        return raw


    def __repr__(self):
        return f"Naive(eps{self._eps})"




class RendererClamp(RendererNaive):
    def forward(self, x: Tensor, mus: Tensor, covs: Tensor, cols: Tensor):
        raw = self._render_gaussians(x, mus, covs).mul(cols).sum(-2) # (D, 3)
        return raw / raw.clamp(1).amax(-1, keepdim=True)


    def __repr__(self):
        return f"Clamp(eps{self._eps})"




class RendererTopK(RendererNaive):
    def __init__(self, eps=1e-6, k=10):
        super().__init__(eps)
        self._k = k


    def forward(self, x: Tensor, mus: Tensor, covs: Tensor, cols: Tensor):
        gauss = self._render_gaussians(x, mus, covs)
        top_k = gauss.topk(min(self._k, mus.shape[1]), 1, sorted=False) # may be wasteful, redo later
        ncols = cols.broadcast_to(x.shape[0], *cols.shape[1:])
        top_k_cols = ncols.gather(1, top_k[1].broadcast_to(*top_k[1].shape[:-1], ncols.shape[-1]))
        raw = top_k[0].mul(top_k_cols).sum(-2) # (D, 3)
        return raw / (top_k[0].sum(-2) + self._eps) # fast 2dgs balancing


    def __repr__(self):
        return f"TopK(eps{self._eps}_K{self._k})"






class WrapperNaive(nn.Module):
    def __init__(self, splatter: SplatterBase, renderer: RendererNaive):
        """
        splatter: Initialised Splatter

        renderer: Initialised Renderer
        """
        super().__init__()

        self.splatter = splatter
        self.renderer = renderer


    def _splat(self, x: Tensor):
        "x: (D, 2)"
        return self.renderer(x, self.mus, self.covs, self.cols)


    def forward(self, x: Tensor):
        # (N1, N2, ..., 2)
        x_flattened = x.reshape(-1, 2)
        y = self._splat(x_flattened)
        #print("Fwd", x.shape, x_flattened.shape, y.shape)

        return y.reshape(*x.shape[:-1], self.cols.shape[-1])


    @property
    def mus(self) -> Tensor:
        return self.splatter.mus
    
    @property
    def covs(self) -> Tensor:
        return self.splatter.covs
    
    @property
    def cols(self) -> Tensor:
        return self.splatter.cols


    def __repr__(self):
        return f"Naive()_{self.splatter}_{self.renderer}"




class WrapperTiledV1(WrapperNaive):
    def __init__(
            self, splatter: SplatterBase, renderer: RendererNaive,
            block: tuple[int, int]=(4, 4),
            min_bound=0.01
            ):
        """
        splatter: Initialised Splatter

        renderer: Initialised Renderer

        n_gaussians: Number of Gaussians to initialise

        block: Tuple - how many blocks for width and height segment to split image into

        min_bound: Gaussian cutoff (Below this maximum value in a given Tile, a Gaussian is not rendered)
        """
        super().__init__(splatter, renderer)

        if min(block) == 0:
            raise Exception(f"Block needs to be at least 1 on each side, got {block}")
        self.block = block
        self.min_bound = min_bound

        self._posneg_min_b = nn.Parameter(
            torch.tensor([-1, 1]).reshape([1, 1, 2])
            * math.sqrt(math.log(min_bound) * -2),
            requires_grad=False
        )
        self._subround = nn.Parameter(torch.tensor([1, 0]).reshape([1, 1, 2]), requires_grad=False)
        self._g_range = nn.Parameter(torch.arange(splatter.n_gaussians), requires_grad=False)


    def _render_tiles(
            self, grid_x: Tensor, grid_y: Tensor,
            x: Tensor, x_tile_inds: Tensor, g_tile_bounds: Tensor,
            final: Tensor, mus: Tensor, covs: Tensor, cols: Tensor
            ):

        x_range = torch.arange(x.shape[0], device=x.device)
        for tile_x in range(len(grid_x) - 1):
            for tile_y in range(len(grid_y) - 1):
                pair = torch.tensor([tile_x, tile_y], device=x.device).reshape(1, 2)

                x_inds = x_range.masked_select((x_tile_inds == pair).all(-1))
                g_inds = self._g_range.masked_select((
                    (g_tile_bounds.select(-1, 0) <= pair)
                    .logical_and(g_tile_bounds.select(-1, 1) > pair)
                ).all(dim=-1)) # (1, K, 1)

                final.index_copy_(0, x_inds, self.renderer(
                    x.index_select(0, x_inds),
                    mus.index_select(1, g_inds), # (1, K, 2)
                    covs.index_select(1, g_inds), # (1, K, 2, 2)
                    cols.index_select(1, g_inds) # (1, K, C)
                )) # K is whatever gaussians qualify for the current tile


    def _splat(self, x: Tensor):
        "x: (D, 2)"
        grid_x = torch.linspace(0, 1, self.block[0] + 1, device=x.device)
        grid_y = torch.linspace(0, 1, self.block[1] + 1, device=x.device)

        x_tile_inds = torch.stack([
            torch.bucketize(x.select(-1, 0), grid_x),
            torch.bucketize(x.select(-1, 1), grid_y)
        ], -1).sub(1).clamp(0) # (D, 2)

        mus, covs, cols = self.mus, self.covs, self.cols

        ac = covs.diagonal(0, -2, -1).permute(1, 2, 0) # (N, 2, 1)
        g_bounds = mus.permute(1, 2, 0) + self._posneg_min_b * ac.sqrt() # (N, 2, 2)

        g_tile_bounds = torch.stack([
            torch.bucketize(g_bounds.select(-2, 0), grid_x),
            torch.bucketize(g_bounds.select(-2, 1), grid_y)
        ], -2) - self._subround # (N, 2, 2) # add clamp if planning to use "==" operations

        final = torch.zeros(size=[x.shape[0], cols.shape[-1]], device=x.device)
        self._render_tiles(grid_x, grid_y, x, x_tile_inds, g_tile_bounds, final, mus, covs, cols)

        return final


    def __repr__(self):
        return f"TiledV1({self.block}_mb{self.min_bound})_{self.splatter}_{self.renderer}"




splatters = [
    SplatterBase,
    SplatterSigRot,
    SplatterCov
]

renderers = [
    RendererNaive,
    RendererClamp,
    RendererTopK
]

wrappers = [
    WrapperNaive,
    WrapperTiledV1
]
