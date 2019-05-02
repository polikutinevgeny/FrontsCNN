"""
Regrids data on lat-lon grid to LCC centered on Russian Far East

Use conda with iris and iris-grib to run this file.
"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import iris
import iris.cube
from spec_hum import spec_humidity

data = iris.load('/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/era5.nc')
target = iris.load('example.grib')[0]
old_cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
new_cs = iris.coord_systems.LambertConformal(central_lat=50, central_lon=130, false_easting=0, false_northing=0, secant_latitudes=(50.0, 50.0), ellipsoid=iris.coord_systems.GeogCS(6371229.0))
target.coord(axis='x').coord_system = new_cs
target.coord(axis='y').coord_system = new_cs
for i in data:
    i.coord(axis='x').coord_system = old_cs
    i.coord(axis='y').coord_system = old_cs
scheme = iris.analysis.Linear(extrapolation_mode='extrapolate')
reprojected = iris.cube.CubeList([i.regrid(target, scheme) for i in data])
# Order may change, but there isn't a better version (or I am lazy)
h = spec_humidity(reprojected[2].data, reprojected[0].data, reprojected[5].data)
humidity = reprojected[5].copy()
humidity.data = h
humidity.units = 'kg/kg'
del reprojected[5]
del reprojected[2]
reprojected.append(humidity)
names = ['air', 'uwnd', 'vwnd', 'mslet', 'shum']
for i, n in zip(reprojected, names):
    i.rename(n)
iris.save(reprojected, '/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/era5_lcc.nc')

# plt.figure(figsize=(8, 8))
# iplt.pcolormesh(reprojected[0], norm=plt.Normalize(220, 320))
# plt.title('Air temperature')
# ax = plt.gca()
# ax.coastlines(resolution='50m')
# ax.gridlines()
# plt.show()
