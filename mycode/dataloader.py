import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from mycode import const
from mycode import parameters


class CustomImageDataset(Dataset):

    def __init__(self, filepathdata, multi_step, startTime, endTime, stepLength):
        self.multi_step = multi_step + 1
        self.stepLength = stepLength
        self.data = xr.open_zarr(filepathdata).sel(time=slice(np.datetime64(startTime), np.datetime64(endTime)))

    def __len__(self):
        return self.data.sizes.get('time') - ((self.multi_step - 1) * self.stepLength)

    def standardization(self, value, mean, std):
        return value #(value - mean) / std

    def __getitem__(self, timestamp):
        items = []
        for step in range(0, self.multi_step):
            levels = parameters.levels
            variablesWithLevels = parameters.variablesWithLevels
            variablesWithoutLevels = parameters.variablesWithoutLevels

            data = self.data.isel(time=timestamp + step * self.stepLength)
            # variablesWithoutLevels
            item_without_level = np.stack(
                [
                    self.standardization(data.variables[var].values, const.FORECAST_MEANS[var], const.FORECAST_STD[var])
                    for
                    var in variablesWithoutLevels
                ],
                axis=0,
            )

            item_with_level = np.stack(
                [
                    self.standardization(data.sel(level=level[1]).variables[f"{var}"].values,
                                         const.FORECAST_MEANS[var + "_" + str(level[0])],
                                         const.FORECAST_STD[var + "_" + str(level[0])]) for var in variablesWithLevels
                    for
                    level in levels
                ],
                axis=0,
            )

            all_levels = np.concatenate((item_without_level, item_with_level))
            all_levels = all_levels.reshape((len(all_levels), -1))

            items.append(torch.as_tensor(all_levels))

        return items
