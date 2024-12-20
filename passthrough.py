from datetime import datetime, timedelta
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np
import pandas as pd
from mycode import const
import time as t



from graph_weather import GraphWeatherForecaster
from mycode.dataloader import CustomImageDataset

from mycode import parameters as para

parser = ArgumentParser()
parser.add_argument('-d', '--dataset_path')


args = parser.parse_args()
datasetPath = args.dataset_path + "/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GraphWeatherForecaster(
    para.lat_lons,
    edge_dim=para.edge_dim,
    hidden_dim_processor_edge=para.hidden_dim_processor_edge, 
    node_dim=para.node_dim,
    hidden_dim_processor_node=para.hidden_dim_processor_node,
    hidden_dim_decoder=para.hidden_dim_decoder,
    feature_dim=para.feature_dim, # feature_dim: Input feature size
    aux_dim=0, # aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc) -> feature dim + aux dim = input_dim als input in dem Encoder
    num_blocks=6,
).to(device)

multi_step = 1

model.load_state_dict(torch.load("/home/hpc/b214cb/b214cb14/safes/run5/safe14.ckt", map_location=torch.device('cpu')))

dataset = CustomImageDataset(datasetPath, 1, para.start_time_evaluation, para.end_time_evaluation, para.stepLength)
dataloader = DataLoader(dataset, batch_size=1)


vars = []  # TODO namen anpassen
for var in para.variablesWithoutLevels:
    vars.append([var, None])

for var in para.variablesWithLevels:
    for lvl in para.levels:
        vars.append([var, lvl])



newDataset = xr.Dataset(
    coords={'longitude': ('longitude', para.long),
            'latitude': ('latitude', para.latt),
            'level': ('level', para.selected_levels),
            'prediction_timedelta': ('prediction_timedelta', (np.arange(0, 41, 1, dtype='timedelta64[ns]')*21600000000000)),
            'time': ('time', pd.date_range(start=para.start_time_evaluation, periods=len(dataloader), freq='6h'))
    }
)

for var in para.variablesWithLevels:
    newDataset[var] = (('time', 'longitude', 'latitude', 'level', 'prediction_timedelta'), np.zeros((len(dataloader), len(para.long), len(para.latt), len(para.selected_levels), 41), dtype='f4'))

for var in para.variablesWithoutLevels:
    newDataset[var] = (('time', 'longitude', 'latitude', 'prediction_timedelta'), np.zeros((len(dataloader), len(para.long), len(para.latt),  41), dtype='f4'))


def destandardization(value, mean, std):
        return (value * std) + mean

def turn_to_array (prediction, j, time):
    prediction_timedelta = j * 21600000000000
    prediction = prediction.detach()
    prediction = prediction.swapaxes(1, 0)
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
        prediction = data[0].to(device)
        for j in range(0, 41):
            print("j: " + str(j))
            if(j != 0):
                #hier muss 1 bis 40 mal der passthrough gemacht werden -> 6h bis 10d in 6h Schritten
                prediction = model(prediction)
            else: 
                #tag 0 ist ohne passthrough
                prediction = data[0].to(device)

            

            turn_to_array(prediction[0], j, time)

        time = time + np.timedelta64(6, 'h')
        end_time = t.time()

newDataset.to_zarr("/home/hpc/b214cb/b214cb14/safes/run5/prediction_after_14.zarr")
        




        



        
      






        

 



