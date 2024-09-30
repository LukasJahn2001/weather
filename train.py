import time

import numpy as np
import torch
from torch_geometric.graphgym import optim
#from argparse import ArgumentParser
from graph_weather import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss
from mycode.dataloader import CustomImageDataset
from torch.utils.data import DataLoader


print("Cuda", torch.cuda.is_available())

#parser = ArgumentParser()
#parser.add_argument('-d', '--dataset_path')

#args = parser.parse_args()
#print(type(args.dataset_path))


running_loss = 0.
last_loss = 0.
multi_step = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = CustomImageDataset('/home/lukas/datasets/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr', multi_step)
dataloader = DataLoader(dataset, batch_size=1)


# Here, we use enumerate(training_loader) instead of
# iter(training_loader) so that we can track the batch
# index and do some intra-epoch reporting

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

model = GraphWeatherForecaster(
    lat_lons,
    edge_dim=32,
    hidden_dim_processor_edge=32,
    node_dim=32,
    hidden_dim_processor_node=32,
    hidden_dim_decoder=32,
    feature_dim=20, # feature_dim: Input feature size
    aux_dim=0, # aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc) -> feature dim + aux dim = input_dim als input in dem Encoder
    num_blocks=6,
).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Done Setup")

feature_variances = []  # has to be feature dim but does not work
for var in range(20):
    feature_variances.append(0.0)
i=0



for epoch in range(1):

    start2 = time.time()

    for batch in dataloader:
        batch = [b.to(device) for b in batch]
        start = time.time()
        optimizer.zero_grad()
        out = batch[0]
        print(len(batch))

        losses = []
        for j in range(0, len(batch)-1):
            # Every data instance is an input + label pair

            target = batch[j+1]
            # Zero your gradients for every batch!
            outputs = model(out)
            # Make predictions for this batch

            criterion = NormalizedMSELoss(lat_lons=lat_lons,
                                          feature_variance=feature_variances,
                                          device=device,

                                          ).to(device)
            loss = criterion(outputs, target)

            losses.append(loss) # tensor oder unten if


            out = outputs




        # Compute the loss and its gradients

        # Adjust learning weights



        for lo in losses:
            print(lo)

        loss_mean = torch.mean(torch.stack(losses))

        loss_mean.backward()
        optimizer.step()


        # Gather data and report
        running_loss += loss_mean.item()
        end = time.time()



        print(
            f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.6f} Time: {end - start} sec"
        )

        i = i + 1
    end2 = time.time()
    print(end2 - start2)
    torch.save(model.state_dict(), "/home/lukas/safes/safe.ckt")



print("Finished Training")

