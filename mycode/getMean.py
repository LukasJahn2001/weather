import xarray as xr

data = xr.open_zarr('/home/lukas/datasets/1990-2017_6h_64x32_equiangular_conservative.zarr')

print("\"surface_pressure\": " + str(data.surface_pressure.mean().values) + ",")
print("\"2m_temperature\": " + str(data.variables["2m_temperature"].mean().values) + ",")
print("\"10m_u_component_of_wind\": " + str(data.variables["10m_u_component_of_wind"].mean().values) + ",")
print("\"10m_v_component_of_wind\": " + str(data.variables["10m_v_component_of_wind"].mean().values) + ",")

for lvl in range(0,13):
    lvlData = data.isel(level=lvl)
    print("\"u_component_of_wind_" + str(lvl) + "\": " + str(lvlData.u_component_of_wind.mean().values) + ",")
    print("\"v_component_of_wind_" + str(lvl) + "\": " + str(lvlData.v_component_of_wind.mean().values) + ",")
    print("\"geopotential_" + str(lvl) + "\": " + str(lvlData.geopotential.mean().values) + ",")
    print("\"temperature_" + str(lvl) + "\": " + str(lvlData.temperature.mean().values) + ",")
    print("\"relative_humidity_" + str(lvl) + "\": " + str(lvlData.relative_humidity.mean().values) + ",")
    print("\"specific_humidity_" + str(lvl) + "\": " + str(lvlData.specific_humidity.mean().values) + ",")




