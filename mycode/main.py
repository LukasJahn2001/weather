import torch
from graph_weather import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss

lat_lons = []
for lat in range(-90, 90, 1):
    for lon in range(0, 360, 1):
        lat_lons.append((lat, lon))
model = GraphWeatherForecaster(lat_lons)

# Generate 78 random features + 24 non-NWP features (i.e. landsea mask)
features = torch.randn((2, len(lat_lons), 102))

target = torch.randn((2, len(lat_lons), 78))
out = model(features)
print(out.shape)

criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=torch.randn((78,)))
loss = criterion(out, target)
loss.backward()
print(loss)