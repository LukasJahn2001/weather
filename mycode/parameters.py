all_levels = [  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000]

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

levels = [4, 7, 10, 11] #250, 500, 850, 925 

variablesWithLevels = ['u_component_of_wind', 'v_component_of_wind', 'temperature', 'relative_humidity']

variablesWithoutLevels = ['surface_pressure', '10m_u_component_of_wind', '10m_u_component_of_wind']

edge_dim=32,
hidden_dim_processor_edge=32,
node_dim=32,
hidden_dim_processor_node=32,
hidden_dim_decoder=32,
feature_dim = len(variablesWithLevels) * len(levels) + len(variablesWithoutLevels)

learning_rate = 0.0001

feature_variances = []  # has to be feature dim
for var in range(feature_dim):
    feature_variances.append(0.0)


first_day = "1959-01-01T00:00:00.000000000"
