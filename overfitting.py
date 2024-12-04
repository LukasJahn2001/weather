from argparse import ArgumentParser
import time
import numpy as np
import torch
import csv
from torch_geometric.graphgym import optim
from graph_weather import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss
from mycode.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
import mycode.parameters as para

x = print("Finished Training")

torch.manual_seed(0)

print("Cuda", torch.cuda.is_available())

'''parser = ArgumentParser()
parser.add_argument('-d', '--dataset_path')


args = parser.parse_args()'''
# datasetPath = args.dataset_path + "/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
datasetPath = "testdataset.zarr"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainDataset = CustomImageDataset(datasetPath, 1, "1959-01-01T00", "1959-01-01T06",
                                  1)


trainDataloader = DataLoader(trainDataset, batch_size=para.batch_size)

# def collate_fn(data):
#     tensors, targets, time = data
#     features = torch.pad_sequence(tensors, batch_first=True)
#     targets = torch.stack(targets)
#     return features, targets


# Here, we use enumerate(training_loader) instead of
# iter(training_loader) so that we can track the batch
# index and do some intra-epoch reporting

model = GraphWeatherForecaster(
    para.lat_lons,
    edge_dim=para.edge_dim,
    hidden_dim_processor_edge=para.hidden_dim_processor_edge,
    node_dim=para.node_dim,
    hidden_dim_processor_node=para.hidden_dim_processor_node,
    hidden_dim_decoder=para.hidden_dim_decoder,
    feature_dim=para.feature_dim,  # feature_dim: Input feature size
    aux_dim=para.aux_dim,
    # aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc) -> feature dim + aux dim = input_dim als input in dem Encoder
    num_blocks=para.num_blocks,
).to(device)

print("edge_dim:")
print(para.edge_dim)
print("hidden_dim_processor_edge:")
print(para.hidden_dim_processor_edge)
print("node_dim:")
print(para.node_dim)
print("hidden_dim_processor_node:")
print(para.hidden_dim_processor_node)
print("hidden_dim_decoder:")
print(para.hidden_dim_decoder)
print("feature_dim:")
print(para.feature_dim)
print("aux_dim:")
print(para.aux_dim)
print("num_blocks:")
print(para.num_blocks)
print("batch_size:")
print(para.batch_size)
print("multi_step:")
print(para.multi_step)

criterion = NormalizedMSELoss(lat_lons=para.lat_lons,
                              feature_variance=para.feature_variances,
                              device=device,

                              ).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
counter_train = 0
counter_validation = 0
epoch_offset = 0

if para.softStart:
    model.load_state_dict(torch.load("/home/hpc/b214cb/b214cb14/safes/run2/safe9.ckt", map_location=device))
    counter_train = para.softStartTrainOffset + 1
    counter_validation = para.softStartValidationOffset + 1
    epoch_offset = para.epoch_offset + 1

print("Done Setup")

for epoch in range(10000000):
    epoch = epoch + epoch_offset
    # Train

    for batch in trainDataloader:
        batch = [b.to(device) for b in batch]
        optimizer.zero_grad()
        out = batch[0]

        losses = []
        for j in range(0, len(batch) - 1):
            # Every data instance is an input + label pair

            target = batch[j + 1]
            # Zero your gradients for every batch!
            outputs = model(out)
            # Make predictions for this batch

            loss = criterion(outputs, target)

            counter_train = counter_train + 1
            print(loss.item())
            losses.append(loss)  # tensor oder unten if


        loss_mean = torch.mean(torch.stack(losses))
        loss_mean.backward()
        optimizer.step()

    end_epoch = time.time()

# TODO: Overfitting test
# TODO: Modell weiter trainieren
# TODO: Passthrough fixen
