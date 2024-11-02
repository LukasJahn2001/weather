import torch
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np
import pandas as pd
from mycode import const



from graph_weather import GraphWeatherForecaster
from mycode.dataloader import CustomImageDataset

from mycode import parameters as para


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GraphWeatherForecaster(
    para.lat_lons,
    edge_dim=32,
    hidden_dim_processor_edge=32,
    node_dim=32,
    hidden_dim_processor_node=32,
    hidden_dim_decoder=32,
    feature_dim=19, # feature_dim: Input feature size
    aux_dim=0, # aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc) -> feature dim + aux dim = input_dim als input in dem Encoder
    num_blocks=6,
).to(device)

multi_step = 1

model.load_state_dict(torch.load("/home/lukas/safes/safe.ckt", map_location=torch.device('cpu')))

dataset = CustomImageDataset('/home/lukas/datasets/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr', multi_step)
dataloader = DataLoader(dataset, batch_size=1)

dates = pd.date_range(start='1/1/1959', periods=len(dataloader), freq='6h')
startTime = 0
endTime = 0

vars = []
for var in para.variablesWithLevels:
     for lvl in para.levels:
          vars.append([var, lvl])

for var in para.variablesWithoutLevels:
     vars.append([var, None])


newDataset = xr.Dataset(
    coords={'longitude': ('longitude', para.long),
            'latitude': ('latitude', para.latt),
            'level': ('level', para.levels),
            'prediction_timedelta': ('prediction_timedelta', np.arange(41)*21600000000000),
            'time': ('time', pd.date_range(start='1/1/1959', periods=len(dataloader), freq='6h'))
    }
)

for var in para.variablesWithLevels:
    newDataset[var] = (('time', 'latitude', 'longitude', 'level', 'prediction_timedelta'), np.zeros((len(dataloader), len(para.latt), len(para.long), len(para.levels), 41)))

for var in para.variablesWithoutLevels:
    newDataset[var] = (('time', 'latitude', 'longitude', 'prediction_timedelta'), np.zeros((len(dataloader), len(para.latt), len(para.long),  41)))

# newDataset.to_zarr("path/to/directory.zarr")
# print(newDataset)
# print(newDataset.variables['u_component_of_wind'].shape)
# print(newDataset['u_component_of_wind'].sel(time=dates[0], prediction_timedelta=21600000000000, level=4))

def destandardization(self, value, mean, std):
        return (value * std) + mean

def turn_to_array (prediction, j, time):
    prediction = prediction.reshape((para.feature_dim, len(para.latt), len(para.long))).T
    # time delta for prediction in ns
    prediction_timedelta = j * 21600000000000
    date = dates[time]

    print(prediction[:, :, 0].transpose(0, 1).shape)
    print(prediction[:, :, 0].detach().numpy().shape)

    for i in range(prediction.shape[2]):
        var = vars[i]
        if(var[1] == None):
            newDataset[var[0]].loc[dict(time=date, prediction_timedelta=prediction_timedelta)] = prediction[:, :, 0].transpose(0, 1).detach().numpy()
        else:
            newDataset[var[0]].loc[dict(time=date, prediction_timedelta=prediction_timedelta, level=var[1])] = prediction[:, :, 0].transpose(0, 1).detach().numpy()
        

        # for long in range(prediction.shape[0]):
        #     for lat in range(prediction.shape[1]):
        #         prediction[att][long][lat]

    
    return 0

# TODO Training mit validation Kurve von loss
# TODO Neuer run mit neuen lvl
# TODO Passthrough schöner und fertig
# TODO Parameterdatei erstellen

time = startTime 
for data in dataloader:
    print(time)
    for j in range(0, 41):
        print("j: " + str(j))
        if(j != 0):
            #hier muss 1 bis 40 mal der passthrough gemacht werden -> 6h bis 10d in 6h Schritten
            prediction = model(prediction)
        else: 
            #tag 0 ist ohne passthrough
            prediction = data[0]

        

        turn_to_array(prediction[0], j, time)

    time = time + 1

newDataset.to_zarr("path/to/directory.zarr")
        




        



        
      






        

 



