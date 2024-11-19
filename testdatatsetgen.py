from datetime import datetime, timedelta

import torch
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np
import pandas as pd
from mycode import const



from graph_weather import GraphWeatherForecaster
from mycode.dataloader import CustomImageDataset

from mycode import parameters as para


multi_step = 0


dataset = CustomImageDataset('/home/lukas/datasets/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr', multi_step, 0, 2)
dataloader = DataLoader(dataset, batch_size=1)

dates = pd.date_range(start='1/1/1959', periods=len(dataloader), freq='6h')
startTime = 0
endTime = 0



vars = [] #TODO namen anpassen
for var in para.variablesWithLevels:
     for lvl in para.levels:
          vars.append([var, lvl])

for var in para.variablesWithoutLevels:
     vars.append([var, None])


newDataset = xr.Dataset(
    coords={'longitude': ('longitude', para.long),
            'latitude': ('latitude', para.latt),
            'level': ('level', para.selected_levels),
            'prediction_timedelta': ('prediction_timedelta', (np.arange(0, 41, 1, dtype='timedelta64[ns]')*21600000000000)),
            'time': ('time', pd.date_range(start='1/1/1959', periods=len(dataloader), freq='6h'))
    }
)

for var in para.variablesWithLevels:
    newDataset[var] = (('time', 'latitude', 'longitude', 'level', 'prediction_timedelta'), np.zeros((len(dataloader), len(para.latt), len(para.long), len(para.selected_levels), 41), dtype='f4'))

for var in para.variablesWithoutLevels:
    newDataset[var] = (('time', 'latitude', 'longitude', 'prediction_timedelta'), np.zeros((len(dataloader), len(para.latt), len(para.long),  41), dtype='f4'))

# newDataset.to_zarr("path/to/directory.zarr")
# print(newDataset)
# print(newDataset.variables['u_component_of_wind'].shape)
# print(newDataset['u_component_of_wind'].sel(time=dates[0], prediction_timedelta=21600000000000, level=4))

def destandardization(value, mean, std):
        return (value * std) + mean

for data in dataloader:
    prediction = data[0]
    prediction = prediction.reshape((para.feature_dim, len(para.latt), len(para.long))).T
    # time delta for prediction in ns
    prediction_timedelta = 0
    date = dates[0]
    print("prediction after reshape")
    print(prediction)

    # print(prediction[:, :, 0].transpose(0, 1).shape)
    # print(prediction[:, :, 0].detach().numpy().shape)

    for i in range(prediction.shape[2]):
        var = vars[i]
        if(var[1] == None):
            variable_name = str(var[0])
            variable = prediction[:, :, 0].transpose(0, 1).detach().numpy()
            variable = [destandardization(var, const.FORECAST_MEANS[variable_name], const.FORECAST_STD[variable_name])
                        for var in variable]
            print("Variable: " + str(var[0]) + " " + str(var[1]))
            print(variable)
            newDataset[var[0]].loc[dict(time=date, prediction_timedelta=np.timedelta64(prediction_timedelta))] = prediction[:, :, 0].transpose(0, 1).detach().numpy()
            print(prediction[:, :, 0].transpose(0, 1).shape)
        else:
            variable_name = str(var[0]) + "_" + str(var[1][0])
            variable = prediction[:, :, 0].transpose(0, 1).detach().numpy()
            variable = [destandardization(var, const.FORECAST_MEANS[variable_name], const.FORECAST_STD[variable_name])
                        for var in variable]
            print("Variable: " + str(var[0]) + " " + str(var[1]))
            print(variable)
            newDataset[var[0]].loc[dict(time=date, prediction_timedelta=np.timedelta64(prediction_timedelta), level=var[1][1])] = prediction[:, :, 0].transpose(0, 1).detach().numpy()

    safe = newDataset.isel(time=0, prediction_timedelta=0).sel(level=500).temperature
    print("ending values")
    print(safe.values)




# newDataset.to_zarr("predictions/testing.zarr")
        




        



        
      






        

 



