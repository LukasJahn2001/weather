import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import const


class CustomImageDataset(Dataset):

    def __init__(self, filepathdata, multi_step):
        self.multi_step = multi_step + 1
        self.data = xr.open_dataset(filepathdata).isel(time=slice(0, 100))
        print(self.data)

    def __len__(self):
        return self.data.sizes.get('time') - self.multi_step +1


    def standardization(self, value, mean, std):
        return (value - mean) / std

    def __getitem__(self, timestamp):
        items = []

        for step in range(0, self.multi_step):
            levels = [0, 6, 12]
            variblesWithLevels = ['u_component_of_wind', 'v_component_of_wind', 'geopotential', 'temperature',
                                  'relative_humidity', 'specific_humidity']
            variblesWithoutLevels = ['surface_pressure', 'mean_sea_level_pressure']

            # variblesWithoutLevels
            data = self.data.isel(time=timestamp + step)

            item_without_level = np.stack(
                [
                    self.standardization(data[f"{var}"].values, const.FORECAST_MEANS[var], const.FORECAST_STD[var]) for
                    var in variblesWithoutLevels
                ],
                axis=-1,
            )

            item_with_level = np.stack(
                [
                    self.standardization(data.isel(level=level)[f"{var}"].values,
                                         const.FORECAST_MEANS[var + "_" + str(level)],
                                         const.FORECAST_STD[var + "_" + str(level)]) for var in variblesWithLevels for
                    level in levels
                ],
                axis=-1,
            )

            item_with_level = np.dstack((item_with_level, item_without_level))

            item_with_level = item_with_level.T.reshape((-1, item_with_level.shape[-1]))

            items.append(torch.as_tensor(item_with_level))

        return items


"""test = CustomImageDataset('/home/lukas/datasets/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr',4)

test = test.__getitem__(0)

print(len(test))

for item in test:
    print(item)"""
