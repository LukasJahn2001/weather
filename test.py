from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import zarr
import time
import pandas as pd
from torch_geometric.data import DataLoader

import mycode.parameters as para
import torch

from mycode import const
from mycode.dataloader import CustomImageDataset

datasetPath = "testdataset.zarr"

dataset = CustomImageDataset(datasetPath, 1, para.start_time_train, para.end_time_train, 1)
dataloader = DataLoader(dataset, batch_size=para.batch_size)

dataset_test = xr.open_zarr(datasetPath).sel(time=slice(np.datetime64(para.start_time_train), np.datetime64(para.end_time_train)))


    
vars = []  # TODO namen anpassen
for var in para.variablesWithoutLevels:
    vars.append([var, None])

for var in para.variablesWithLevels:
    for lvl in para.levels:
        vars.append([var, lvl])



newDataset = xr.Dataset(
    coords={
            'longitude': ('longitude', para.long),
            'latitude': ('latitude', para.latt),
            'level': ('level', para.selected_levels),
            'prediction_timedelta': (
            'prediction_timedelta', (np.arange(0, 41, 1, dtype='timedelta64[ns]') * 21600000000000)),
            'time': ('time', pd.date_range(start=para.start_time_evaluation, periods=len(dataloader), freq='6h'))
            }
)

for var in para.variablesWithLevels:
    newDataset[var] = (('time', 'longitude', 'latitude', 'level', 'prediction_timedelta'),
                       np.zeros((len(dataloader), len(para.long), len(para.latt), len(para.selected_levels), 41),
                                dtype='f4'))

for var in para.variablesWithoutLevels:
    newDataset[var] = (('time', 'longitude', 'latitude', 'prediction_timedelta'),
                       np.zeros((len(dataloader), len(para.long), len(para.latt), 41), dtype='f4'))


def destandardization(value, mean, std):
    return value #(value * std) + mean


def turn_to_array(prediction, j, time):
    prediction_timedelta = j * 21600000000000
    prediction = prediction.reshape((len(prediction), len(para.long), len(para.latt)))
    
    for i in range(len(prediction)):
        var = vars[i]
        if var[1] is None:
            variable_name = str(var[0])
            variable = prediction[i].detach().numpy()#
            variable = [destandardization(value, const.FORECAST_MEANS[variable_name], const.FORECAST_STD[variable_name]) for value in variable]
            newDataset[var[0]].loc[
                dict(time=time, prediction_timedelta=np.timedelta64(prediction_timedelta))] = variable
        else:
            variable_name = str(var[0]) + "_" + str(var[1][0])
            variable = prediction[i].detach().numpy()
            variable = [destandardization(value, const.FORECAST_MEANS[variable_name], const.FORECAST_STD[variable_name])
                        for value in variable]
            newDataset[var[0]].loc[
                dict(time=time, prediction_timedelta=np.timedelta64(prediction_timedelta), level=var[1][1])] = variable

    return 0


# TODO Training mit validation Kurve von loss
# TODO Neuer run mit neuen lvl
# TODO Passthrough schöner und fertig
# TODO Parameterdatei erstellen

time = np.datetime64(para.start_time_evaluation)
with torch.no_grad():
    for data in dataloader:
        print(time)
        for j in range(0, 1):
            print("j: " + str(j))
            if (j != 0):
                # hier muss 1 bis 40 mal der passthrough gemacht werden -> 6h bis 10d in 6h Schritten
                prediction = data[0]
            else:
                # tag 0 ist ohne passthrough
                prediction = data[0]

            turn_to_array(prediction[0], j, time)

        time = time + np.timedelta64(6, 'h')


for var in vars:
        variable_name = str(var[0])
        variable_level = var[1]
        if variable_level is None:
            before = dataset_test.isel(time=0).variables[variable_name].values
            after = newDataset.isel(time=0, prediction_timedelta=0).variables[variable_name].values
            print(variable_name + "--------------------------------------------------------------------------------")
            print(np.allclose(before,after))
            print(before)
            print(after)
        else:
            before = dataset_test.isel(time=0).sel(level=variable_level[1]).variables[variable_name].values 
            after = newDataset.isel(time=0, prediction_timedelta=0).sel(level=int(variable_level[1])).variables[variable_name].values
            print(variable_name + " : " + str(variable_level[1]) + "--------------------------------------------------------------------------------")
            print(np.allclose(before,after))
            print(before)
            print(after)
            
   

'''a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[13,14,15],[16,17,18],[19,20,21]])
c = np.array([[25,26,27],[28,29,30],[31,32,33]])
d = np.array([[37,38,39],[40,41,42],[43,44,45]])

Test = [value / 2 for value in a]

print("Test:")
print(Test)



print("Start:")
print(a.shape)



withoutlvl = np.stack([a,b], axis=0)
withlvl = np.stack([c,d], axis=0)

print("Concatenate:")
print(withlvl.shape)
print(withlvl)
print(withoutlvl.shape)
print(withoutlvl)

concatenated = np.concatenate((withoutlvl, withlvl))

print("Concatenate:")
print(concatenated.shape)
print(concatenated)

print("Reshape:")

reshaped = concatenated.reshape((len(concatenated),-1))

print(reshaped.shape)
print(reshaped)

items = []
items.append(torch.as_tensor(reshaped))

print("Tensor:")
print(items[0].shape)
print(items[0])

print("As numpy:")

my_numpy = items[0].numpy()

print(my_numpy.shape)
print(my_numpy)

print("As numpy reshaped:")
my_numpy = my_numpy.reshape((4,3,3))

print(my_numpy.shape)
print(my_numpy)

print("Each item:")
for i in range(4):
    print("Item " + str(i))
    print(my_numpy[i])'''
