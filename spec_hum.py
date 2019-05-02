"""
Computes specific humidity using surface pressure, air temperature and dew point
https://github.com/Unidata/MetPy/issues/791#issuecomment-377501593

Modified to fit the task specifics
"""
import numpy as np


def spec_humidity(pressure, temp, dew, latent='water'):
    """Calculates SH automatically from the dewpt. Returns in kg/kg"""
    # Declaring constants
    e0 = 611.3  # saturation vapor pressure in Pa
    # e0 and Pressure have to be in same units
    c_water = 5423  # L/R for water in Kelvin
    c_ice = 6139  # L/R for ice in Kelvin
    t0 = 273.15  # Kelvin
    if latent == 'water' or latent == 'Water':
        c = c_water  # using c for water
    else:
        c = c_ice  # using c_ice for ice, clear state
    # saturation vapor not required, uncomment to calculate it (units in hPa becuase of e0)
    # sat_vapor = self.e0 * np.exp((self.c * (self.temp -self.t0))/(self.temp * self.t0))

    # calculating specific humidity, q directly from dew point temperature
    # using equation 4.24, Pg 96 Practical Meteorolgy (Roland Stull)
    q = (622 * e0 * np.exp(c * (dew - t0) / (dew * t0))) / pressure  # g/kg
    # 622 is the ratio of Rd/Rv in g/kg
    return q / 1000  # kg/kg
