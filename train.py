import time
import numpy as np
import torch
import csv
from torch_geometric.graphgym import optim
#from argparse import ArgumentParser
from graph_weather import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss
from mycode.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
import mycode.parameters as para


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

# def collate_fn(data):
#     tensors, targets, time = data
#     features = torch.pad_sequence(tensors, batch_first=True)
#     targets = torch.stack(targets)
#     return features, targets

dataloader = DataLoader(dataset, batch_size=4)


# Here, we use enumerate(training_loader) instead of
# iter(training_loader) so that we can track the batch
# index and do some intra-epoch reporting

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

# TODO seed fixen

optimizer = optim.Adam(model.parameters(), lr=para.learning_rate)
print("Done Setup")

with open('losses.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                field = ["time", "loss"]
                writer.writerow(["time", "loss"])

counter = 0
i=0

for epoch in range(100):

    start2 = time.time()

    for batch in dataloader:
        batch = [b.to(device) for b in batch]
        start = time.time()
        optimizer.zero_grad()
        print(batch[0].shape)
        out = batch[0]
        print(len(batch))

        losses = []
        for j in range(0, len(batch)-1):
            # Every data instance is an input + label pair

            target = batch[j+1]
            # Zero your gradients for every batch!
            outputs = model(out)
            # Make predictions for this batch
            
            # TODO: Ändern sich die Parameter überhaupt. Schrittweise debuggen

            criterion = NormalizedMSELoss(lat_lons=para.lat_lons,
                                          feature_variance=para.feature_variances,
                                          device=device,

                                          ).to(device)
            # TODO: criterion nach außen ziehen

            loss = criterion(outputs, target)

            with open('losses.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                field = ["time", "loss"]
                writer.writerow([counter, loss.item()])
                print(type(loss))
            
            # TODO: with open aus der Schleife raus 

            counter = counter + 1

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

