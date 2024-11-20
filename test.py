from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import zarr
import time
import pandas as pd
import mycode.parameters as para


#data = xr.open_dataset('/home/lukas/datasets/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr').isel(time=slice(0, 120))
#data.to_zarr('./testdataset.zarr')

data = xr.open_zarr('/home/lukas/datasets/2020-64x32_equiangular_conservative.zarr')
data2 = xr.open_zarr('/home/lukas/datasets/date_range_2017-11-16_2019-02-01_12_hours-64x32_equiangular_conservative.zarr')

data3 = xr.open_zarr('/home/lukas/datasets/1990-2017_6h_64x32_equiangular_conservative.zarr')
data4 = xr.open_zarr('/home/lukas/datasets/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr') 
data5 = xr.open_zarr('/home/lukas/git/weather/testdataset.zarr')


# print(data4.level[4])  #250
# print(data4.level[7])  #500
# print(data4.level[10]) #850
# print(data4.level[11]) #925

# data3.variables["mean_sea_level_pressure"].values

#print(data)


'''data6 = xr.open_zarr('/home/lukas/datasets/firstprediction.zarr')

print(data)
print(data6)

print(data.prediction_timedelta)
print(data6.prediction_timedelta.astype('timedelta64[ns]'))'''




# print(changed.sizes.get('time'))
# print(changed)


# print(time.strftime("%H:%M:%S"))


# print(data4.isel(time=slice(0,2)).variables['time'].values)

# print(pd.date_range(start='1/1/1959', periods=(4*365), freq='6h'))

# print(data4.sel(time = np.datetime64('2020-01-01')))
# print(data4.sizes.get("time"))
print(data4.time)
test = data4.sel(time=slice(np.datetime64('2020-01-01T00'), np.datetime64('2020-12-31T18')))
#test = data4.isel(time=slice(89060, 90520))
print(test.sizes.get("time"))
times = test.time
print(times[0])
print(times[-1])
'''



Vars with level
u_component_of_wind
v_component_of_wind
geopotential
temperature
relative_humidity
specific_humidity
u_component_of_wind
v_component_of_wind
geopotential
temperature
relative_humidity
specific_humidity
u_component_of_wind
v_component_of_wind
geopotential
temperature
relative_humidity
specific_humidity
Vars without levels
surface_pressure
mean_sea_level_pressure

Var: surface_pressure
Var: mean_sea_level_pressure
Var: u_component_of_wind Level; 0
Var: u_component_of_wind Level; 6
Var: u_component_of_wind Level; 12
Var: v_component_of_wind Level; 0
Var: v_component_of_wind Level; 6
Var: v_component_of_wind Level; 12
Var: geopotential Level; 0
Var: geopotential Level; 6
Var: geopotential Level; 12
Var: temperature Level; 0
Var: temperature Level; 6
Var: temperature Level; 12
Var: relative_humidity Level; 0
Var: relative_humidity Level; 6
Var: relative_humidity Level; 12
Var: specific_humidity Level; 0
Var: specific_humidity Level; 6
Var: specific_humidity Level; 12

[[-2.8757517  -2.5956948  -2.9500775  ...  0.32376248  0.421044
   0.4490475 ]
 [-2.9140403  -2.8132567  -3.2679684  ...  0.28670627  0.40132505
   0.4435534 ]
 [-2.947033   -2.9671311  -3.5124445  ...  0.25511897  0.31217536
   0.438107  ]
 ...
 [-2.697499   -1.8307189  -1.0025918  ...  0.4076689   0.29070577
   0.46170223]
 [-2.771122   -2.1568673  -1.9190856  ...  0.3976467   0.43125543
   0.4583155 ]
 [-2.8289354  -2.3954675  -2.6502523  ...  0.3591974   0.4356212
   0.4537631 ]]

[[-0.9181486  -1.4419732  -1.6265014  ... -1.3099076  -0.4385323
  -0.18817335]
 [-0.8506235  -1.3938476  -1.6172136  ... -1.6418438  -0.6134773
  -0.23702179]
 [-0.8122131  -1.3215892  -1.5497041  ... -1.8099027  -0.7752018
  -0.2847588 ]
 ...
 [-1.1359107  -1.7349162  -2.0101695  ... -0.40049502 -0.16836195
  -0.07456674]
 [-1.0782175  -1.6140348  -1.8609973  ... -0.6473642  -0.2022644
  -0.10621524]
 [-1.0028969  -1.5255636  -1.7053663  ... -0.9919303  -0.30798998
  -0.1440038 ]]

[[-0.77624524 -0.80175024 -0.8323771  ... -0.48953032 -0.44748527
  -0.42896757]
 [-0.4041434  -0.36705106 -0.34071684 ... -0.08986661 -0.12159541
  -0.15292594]
 [-0.16988835 -0.1550843  -0.11625242 ... -0.66739595 -0.6989253
  -0.7227615 ]
 ...
 [-0.38663715 -0.43857893 -0.4881424  ... -0.33679387 -0.295057
  -0.25273722]
 [-0.2071065  -0.15758188 -0.10635514 ...  0.16698627  0.1722092
   0.17868347]
 [ 0.18448155  0.1886086   0.1897589  ... -0.07456674 -0.10621524
  -0.1440038 ]]


[[[-0.77624524 -0.5066838  -1.0070187  ... -1.1686941  -2.8757517
   -0.9181486 ]
  [-0.67663914 -1.0051913  -0.85387254 ... -1.1658992  -2.5956948
   -1.4419732 ]
  [-0.7068216  -0.9950775  -0.41550007 ... -1.1656874  -2.9500775
   -1.6265014 ]
  ...
  [ 0.6223722   0.0366995  -0.57791626 ... -0.98939234  0.32376248
   -1.3099076 ]
  [ 0.04166539 -0.02812871 -0.18951386 ... -1.1929936   0.421044
   -0.4385323 ]
  [-0.40373614 -0.6959073   0.35887408 ... -1.202476    0.4490475
   -0.18817335]]

 [[-0.80175024 -0.5589704  -0.89444387 ... -1.1711886  -2.9140403
   -0.8506235 ]
  [-0.71582896 -1.0112462  -0.73778945 ... -1.1751477  -2.8132567
   -1.3938476 ]
  [-0.76125133 -0.9399658  -0.49264976 ... -1.176251   -3.2679684
   -1.6172136 ]
  ...
  [ 0.5824559  -0.13562857 -0.97414505 ... -0.8530094   0.28670627
   -1.6418438 ]
  [ 0.1009877   0.28055653 -0.3525182  ... -1.1770697   0.40132505
   -0.6134773 ]
  [-0.40716898 -0.7782758   0.20446607 ... -1.2010057   0.4435534
   -0.23702179]]

 [[-0.8323771  -0.6105317  -0.783972   ... -1.1739391  -2.947033
   -0.8122131 ]
  [-0.8057549  -0.93279785 -0.51578736 ... -1.1818908  -2.9671311
   -1.3215892 ]
  [-0.83282065 -0.8959582  -0.50045294 ... -1.1866273  -3.5124445
   -1.5497041 ]
  ...
  [ 0.5515155  -0.17829207 -0.9143904  ... -0.7699014   0.25511897
   -1.8099027 ]
  [ 0.19058174  0.49961966 -0.5506965  ... -1.1604685   0.31217536
   -0.7752018 ]
  [-0.36137414 -0.7655202   0.04125791 ... -1.200126    0.438107
   -0.2847588 ]]

 ...

 [[-0.7490147  -0.58011246 -1.2572193  ... -1.1537731  -2.697499
   -1.1359107 ]
  [-0.74017566 -0.5612632  -1.1249617  ... -1.1270233  -1.8307189
   -1.7349162 ]
  [-0.72686565 -1.1242788  -0.7555065  ... -1.0644653  -1.0025918
   -2.0101695 ]
  ...
  [ 0.3909998  -0.42498803  0.09788985 ... -1.1814086   0.4076689
   -0.40049502]
  [-0.04755339 -0.86399734  0.12199619 ... -1.1962421   0.29070577
   -0.16836195]
  [-0.39625505 -0.5522231   0.573327   ... -1.2040701   0.46170223
   -0.07456674]]

 [[-0.7543155  -0.52266437 -1.2182105  ... -1.1589779  -2.771122
   -1.0782175 ]
  [-0.71425015 -0.77987236 -1.026417   ... -1.1433418  -2.1568673
   -1.6140348 ]
  [-0.7102699  -1.0120915  -0.6100851  ... -1.1129043  -1.9190856
   -1.8609973 ]
  ...
  [ 0.4274288  -0.12357225 -0.02964632 ... -1.1688124   0.3976467
   -0.6473642 ]
  [ 0.02126658 -0.65339047  0.05528718 ... -1.1925292   0.43125543
   -0.2022644 ]
  [-0.3733385  -0.5418794   0.546215   ... -1.2038244   0.4583155
   -0.10621524]]

 [[-0.7611774  -0.49077666 -1.1232451  ... -1.1645985  -2.8289354
   -1.0028969 ]
  [-0.6921837  -0.92926735 -0.9049825  ... -1.1554589  -2.3954675
   -1.5255636 ]
  [-0.70726204 -1.0380098  -0.4525634  ... -1.1556518  -2.6502523
   -1.7053663 ]
  ...
  [ 0.5866892   0.08398068 -0.42040673 ... -1.0729467   0.3591974
   -0.9919303 ]
  [ 0.067635   -0.3602138  -0.14809074 ... -1.1918463   0.4356212
   -0.30798998]
  [-0.37999555 -0.603408    0.46731552 ... -1.2029761   0.4537631
   -0.1440038 ]]]'''