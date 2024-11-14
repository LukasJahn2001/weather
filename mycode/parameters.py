import numpy as np

all_levels = [  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000]

long = np.arange(0, 355, 5.625, dtype='f8')
latt = np.arange(-87.1875, 88, 5.625, dtype='f8')

lat_lons = []
for lat in latt:
    for lon in long:
        lat_lons.append((lat, lon))

levels = [[4,250], [7, 500], [10, 850], [11, 925]] #250, 500, 850, 925 #

new = []

variablesWithLevels = ['u_component_of_wind', 'v_component_of_wind', 'temperature', 'relative_humidity']

variablesWithoutLevels = ['surface_pressure', '10m_u_component_of_wind', '10m_u_component_of_wind']

edge_dim=512
hidden_dim_processor_edge=512
node_dim=512
hidden_dim_processor_node=512
hidden_dim_decoder=512
feature_dim = len(variablesWithLevels) * len(levels) + len(variablesWithoutLevels)
aux_dim=0 # aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc) -> feature dim + aux dim = input_dim als input in dem Encoder
num_blocks=6

learning_rate = 0.0001

feature_variances = []  # has to be feature dim
for var in range(feature_dim):
    feature_variances.append(0.0)

multi_step = 1
batch_size = 8

first_day = "1959-01-01T00:00:00.000000000"

softStart = False
softStartTrainOffset = 0
softStartValidationOffset = 0
