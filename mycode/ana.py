import torch
import apache_beam   # Needs to be imported separately to avoid TypingError
import weatherbench2
import xarray as xr
from graph_weather import GraphWeatherForecaster




long = [0., 5.625, 11.25, 16.875, 22.5, 28.125, 33.75, 39.375,
        45., 50.625, 56.25, 61.875, 67.5, 73.125, 78.75, 84.375,
        90., 95.625, 101.25, 106.875, 112.5, 118.125, 123.75, 129.375,
        135., 140.625, 146.25, 151.875, 157.5, 163.125, 168.75, 174.375,
        180., 185.625, 191.25, 196.875, 202.5, 208.125, 213.75, 219.375,
        225., 230.625, 236.25, 241.875, 247.5, 253.125, 258.75, 264.375,
        270., 275.625, 281.25, 286.875, 292.5, 298.125, 303.75, 309.375,
        315., 320.625, 326.25, 331.875, 337.5, 343.125, 348.75, 354.375]
latt = [-87.1875, -81.5625, -75.9375, -70.3125, -64.6875, -59.0625, -53.4375,
       -47.8125, -42.1875, -36.5625, -30.9375, -25.3125, -19.6875, -14.0625,
        -8.4375,  -2.8125,   2.8125,   8.4375,  14.0625,  19.6875,  25.3125,
        30.9375,  36.5625,  42.1875,  47.8125,  53.4375,  59.0625,  64.6875,
        70.3125,  75.9375,  81.5625,  87.1875]

lat_lons = []
for lat in latt:
    for lon in long:
        lat_lons.append((lat, lon))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GraphWeatherForecaster(
    lat_lons,
    edge_dim=128,
    hidden_dim_processor_edge=128,
    node_dim=128,
    hidden_dim_processor_node=128,
    hidden_dim_decoder=128,
    feature_dim=20, # feature_dim: Input feature size
    aux_dim=0, # aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc) -> feature dim + aux dim = input_dim als input in dem Encoder
    num_blocks=6,
).to(device)
model.load_state_dict(torch.load("/home/lukas/safes/safe"))
print(model)
#print(model.eval())