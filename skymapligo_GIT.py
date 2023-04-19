#%%
from astropy.coordinates import SkyCoord
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import crossmatch
from ligo.skymap import version
import ligo.skymap.io.fits
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
import numpy as np
from ligo.skymap.postprocess import crossmatch
import ligo.skymap.plot
from matplotlib import pyplot as plt

import astropy.io.fits.TableHDU


sky_map = read_sky_map('bayestar.fits.gz', moc=True)


#dec is between -90deg and 90deg
random_dec = []
for i in range(100000):
    declination = round(np.random.uniform(-90, 90), 2)
    random_dec.append(declination)

#ra is from 0h to 24h
random_ra = []
for i in range(100000):
    rightac = round(np.random.uniform(0, 24), 2)
    random_ra.append(rightac)


ra = Longitude(random_ra, unit=u.deg)  # Could also use Angle
dec = np.array(random_dec) * u.deg  # Astropy Quantity
c = SkyCoord(ra, dec, frame='icrs')
c = SkyCoord(frame=ICRS, ra=ra, dec=dec, obstime='2001-01-02T12:34:56')

result = crossmatch(sky_map, c)
probdens = result.probdensity
data = zip(ra,dec,probdens)

#print(result)
#offset is the distance between the ra and dec pairs and 
#the centre of the locationalisation region
#offset = result.offset



ax = plt.axes(projection='astro hours mollweide')
ax.grid()

#works as long as use URL
#url = 'https://gracedb.ligo.org/api/superevents/S190814bv/files/bayestar.fits.gz'
url = 'https://gracedb.ligo.org/api/superevents/S190814bv/files/LALInference.v1.fits.gz'

ax.imshow_hpx(data, cmap='cylon')








#plot sky map on mallweide globe and on the astroglobe
#try and plot the 100 random points using scatter colour coded by the probabilityzx

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
