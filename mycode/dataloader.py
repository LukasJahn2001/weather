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
        self.data = xr.open_zarr(filepathdata).isel(time=slice(startTime, endTime))

    def __len__(self):
        return self.data.sizes.get('time') - ((self.multi_step -1) * self.stepLength)


    def standardization(self, value, mean, std):
        return (value - mean) / std

    def __getitem__(self, timestamp):
        items = []
        for step in range(0, self.multi_step):
            levels = parameters.levels
            variablesWithLevels = parameters.variablesWithLevels
            variablesWithoutLevels = parameters.variablesWithoutLevels


            data = self.data.isel(time=timestamp + step * self.stepLength)

            testing = data.sel(level=500).temperature

        
        
            # variablesWithoutLevels
            item_without_level = np.stack(
                [
                    self.standardization(data.variables[var].values, const.FORECAST_MEANS[var], const.FORECAST_STD[var]) for
                    var in variablesWithoutLevels
                ],
                axis=-1,
            )
            
            item_with_level = np.stack(
                [
                    self.standardization(data.sel(level=level[1]).variables[f"{var}"].values,
                                         const.FORECAST_MEANS[var + "_" + str(level[0])],
                                         const.FORECAST_STD[var + "_" + str(level[0])]) for var in variablesWithLevels for
                    level in levels
                ],
                axis=-1,
            )

            item_with_level = np.dstack((item_with_level, item_without_level))
            # print("Stack:")
            # print(item_with_level.shape)
            # print(item_with_level)
            # start = item_with_level

            item_with_level = item_with_level.T.reshape((-1, item_with_level.shape[-1]))
            # print("Reshape:")
            # print(item_with_level.shape)
            # print(item_with_level)

            # original_shape = start.shape
            # reversed_result = item_with_level.reshape((20, 32, 64)).T

            # print("Original reversed:")
            # print(reversed_result.shape)
            # print(reversed_result)

            # print("Are close:")
            # are_close = np.allclose(start, reversed_result)
            # print(are_close)

            items.append(torch.as_tensor(item_with_level))
            # items.append(data.variables['time'])
            

        return items
