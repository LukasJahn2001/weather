import apache_beam  # Needs to be imported separately to avoid TypingError
import weatherbench2
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

from weatherbench2 import config
from weatherbench2.metrics import MSE, ACC
from weatherbench2.regions import SliceRegion, ExtraTropicalRegion
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam

forecast_path = '/home/lukas/datasets/date_range_2017-11-16_2019-02-01_12_hours-64x32_equiangular_conservative.zarr'
obs_path = '/home/lukas/datasets/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'
climatology_path = '/home/lukas/datasets/1990-2017_6h_64x32_equiangular_conservative.zarr'

print(xr.open_zarr(forecast_path).dims)
print(xr.open_zarr(obs_path).dims)

climatology = xr.open_zarr(climatology_path)

print(climatology.dims)

paths = config.Paths(
    forecast=forecast_path,
    obs=obs_path,
    output_dir='./',  # Directory to save evaluation results
)

selection = config.Selection(
    variables=[
        'geopotential',
        '2m_temperature'
    ],
    levels=[500, 700, 850],
    time_slice=slice('2017-12-01', '2017-12-31'),
)

data_config = config.Data(selection=selection, paths=paths)

eval_configs = {
    'deterministic': config.Eval(
        metrics={
            'mse': MSE(),
            'acc': ACC(climatology=climatology)
        },
    )
}

regions = {
    'global': SliceRegion(),
    'tropics': SliceRegion(lat_slice=slice(-20, 20)),
    'extra-tropics': ExtraTropicalRegion(),
}

eval_configs2 = {
    'deterministic': config.Eval(
        metrics={
            'mse': MSE(),
            'acc': ACC(climatology=climatology)
        },
        regions=regions
    )
}

evaluate_in_memory(data_config, eval_configs)

results = xr.open_dataset('./deterministic.nc')
print(results)

results = xr.concat(
    [
        results,
        results.sel(metric=['mse']).assign_coords(metric=['rmse']) ** 0.5
    ],
    dim='metric'
)

results['geopotential'].sel(metric='rmse', level=500).plot()

plt.show()
