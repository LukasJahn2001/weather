import xarray as xr

data = xr.open_dataset('/home/lukas/datasets/1990-2017_6h_64x32_equiangular_conservative.zarr')

print("\"surface_pressure\": " + str(data.surface_pressure.std().values) + ",")
print("\"mean_sea_level_pressure\": " + str(data.mean_sea_level_pressure.std().values) + ",")

for lvl in range(0,13):
    lvlData = data.isel(level=lvl)
    print("\"u_component_of_wind_" + str(lvl) + "\": " + str(lvlData.u_component_of_wind.std().values) + ",")
    print("\"v_component_of_wind_" + str(lvl) + "\": " + str(lvlData.v_component_of_wind.std().values) + ",")
    print("\"geopotential_" + str(lvl) + "\": " + str(lvlData.geopotential.std().values) + ",")
    print("\"temperature_" + str(lvl) + "\": " + str(lvlData.temperature.std().values) + ",")
    print("\"relative_humidity_" + str(lvl) + "\": " + str(lvlData.relative_humidity.std().values) + ",")
    print("\"specific_humidity_" + str(lvl) + "\": " + str(lvlData.specific_humidity.std().values) + ",")
