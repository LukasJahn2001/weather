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

torch.manual_seed(0)



print("Cuda", torch.cuda.is_available())

#parser = ArgumentParser()
#parser.add_argument('-d', '--dataset_path')

#args = parser.parse_args()
#print(type(args.dataset_path))


running_loss = 0.
last_loss = 0.
multi_step = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = CustomImageDataset('/home/lukas/datasets/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr', multi_step)

# def collate_fn(data):
#     tensors, targets, time = data
#     features = torch.pad_sequence(tensors, batch_first=True)
#     targets = torch.stack(targets)
#     return features, targets

dataloader = DataLoader(dataset, batch_size=3)


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


criterion = NormalizedMSELoss(lat_lons=para.lat_lons,
                                            feature_variance=para.feature_variances,
                                            device=device,

                                            ).to(device)

optimizer = optim.Adam(model.parameters(), lr=para.learning_rate)
print("Done Setup")
counter = 0

with open('losses.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for epoch in range(10):
        start_epoch = time.time()
        for batch in dataloader:
            print("Leng")
            print(len(dataloader))
            print(batch[0].shape)
            print("END")
            batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            out = batch[0]

            losses = []
            print()
            for j in range(0, len(batch)-1):
                # Every data instance is an input + label pair

                target = batch[j+1]
                # Zero your gradients for every batch!
                outputs = model(out)
                # Make predictions for this batch
                
                # TODO: Ändern sich die Parameter überhaupt. Schrittweise debuggen

            

                loss = criterion(outputs, target)

                
                writer.writerow([counter, loss.item()])
                    
                
                # TODO: with open aus der Schleife raus 

                counter = counter + 1

                losses.append(loss) # tensor oder unten if

                out = outputs





            loss_mean = torch.mean(torch.stack(losses))

            loss_mean.backward()
            optimizer.step()

        end_epoch = time.time()
        print("Epoch " + str(epoch) + ": " + str(end_epoch - start_epoch) + "s")


        torch.save(model.state_dict(), "/home/lukas/safes/safe.ckt")



    print("Finished Training")

