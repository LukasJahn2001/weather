
import xarray as xr
import zarr


data = xr.open_dataset('/home/lukas/datasets/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr').isel(time=slice(0, 120))
data.to_zarr('./testdataset.zarr')


