from torch import Tensor, nn
import xarray as xr


def prepare_metric_xy(metrics: xr.DataArray, x_dim: str, y_dim: str):
    return (
        metrics.sel(metric=y_dim)
        .assign_coords(epoch=metrics.sel(metric=x_dim))
        .rename(epoch=x_dim)
    )


def array_from_dict(d: dict, dim_new):
    return xr.Dataset(d).to_dataarray(dim_new)


def flatten_xarray(arr: xr.DataArray, dim0, dim1, dim_new, idx_suffix=("", "")):
    arr_flat = arr.stack(**{dim_new: (dim0, dim1)})
    idx_flat = [f"{idx[0]}{idx_suffix[0]}_{idx[1]}{idx_suffix[1]}" for idx in arr_flat.indexes[dim_new]]
    return arr_flat.drop_vars([dim_new, dim0, dim1]).assign_coords(**{dim_new: idx_flat})


def save_2d_xr(arr: xr.DataArray, path: str):
    arr.to_pandas().to_csv(path, sep=";")




class PermuteBatchWrapper(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
    
    def forward(self, pred: Tensor, gt: Tensor):
        return self.metric(
            pred.permute(2, 0, 1).unsqueeze(0).clamp(0, 1),
            gt.permute(2, 0, 1).unsqueeze(0)
        )
