import numpy as np

all_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
selected_levels = [250, 500, 850, 925]

long = np.arange(0, 355, 5.625, dtype='f8')
latt = np.arange(-87.1875, 88, 5.625, dtype='f8')

lat_lons = []
for lat in latt:
    for lon in long:
        lat_lons.append((lat, lon))

levels = [[4, 250], [7, 500], [10, 850], [11, 925]]  # 250, 500, 850, 925 #TODO: Überprüfen ob das so stimmt

variablesWithLevels = ['u_component_of_wind', 'v_component_of_wind', 'temperature', 'relative_humidity']

variablesWithoutLevels = ['surface_pressure', '10m_u_component_of_wind', '10m_u_component_of_wind']

edge_dim = 32
hidden_dim_processor_edge = 32
node_dim = 32
hidden_dim_processor_node = 32
hidden_dim_decoder = 32
feature_dim = len(variablesWithLevels) * len(levels) + len(variablesWithoutLevels)
aux_dim = 0  # aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc) -> feature dim + aux dim = input_dim als input in dem Encoder
num_blocks = 6

learning_rate = 0.0001

feature_variances = []  # has to be feature dim
for var in range(feature_dim):
    feature_variances.append(0.0)

multi_step = 1
batch_size = 1
stepLength = 1

softStart = False
softStartTrainOffset = 0
softStartValidationOffset = 0
epoch_offset = 0

start_time_train = '1959-01-01T00'
end_time_train = '1959-01-01T06'
start_time_validation = '1959-03-01T00'
end_time_validation = '1959-04-01T00'
start_time_evaluation = '2020-01-01T00'
end_time_evaluation = '2020-01-10T18'

