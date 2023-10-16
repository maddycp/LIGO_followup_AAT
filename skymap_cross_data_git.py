#%%
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from astropy.coordinates import SkyCoord
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import crossmatch
import numpy as np
from matplotlib import pyplot as plt
from astroquery.vizier import VizierClass
from scipy import interpolate
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
import astropy.units as u
#sky_map = read_sky_map('bayestar.fits.gz', moc=True)

sky_map =  read_sky_map('GW190814_skymap.fits.gz', moc=True)

#filename_ligo = get_pkg_data_filename('allsky/ligo_simulated.fits.gz')

target_header = fits.Header.fromstring("""
NAXIS   =                    2
NAXIS1  =                  480
NAXIS2  =                  240
CTYPE1  = 'RA---MOL'
CRPIX1  =                240.5
CRVAL1  =                180.0
CDELT1  =               -0.675
CUNIT1  = 'deg     '
CTYPE2  = 'DEC--MOL'
CRPIX2  =                120.5
CRVAL2  =                  0.0
CDELT2  =                0.675
CUNIT2  = 'deg     '
COORDSYS= 'icrs    '
""", sep='\n')


vizier = VizierClass(

    row_limit=-1,

    columns=['recno', 'GWGC', '_RAJ2000', '_DEJ2000', 'Dist'])

cat, = vizier.get_catalogs('VII/281/glade2')

cat.sort('recno')  
del cat['recno']

coordinates = SkyCoord(cat['_RAJ2000'], cat['_DEJ2000'], cat['Dist'])


#ra is from 0h to 24h or 0 to 360 degrees
random_ra = []
for i in range(10000):
    rightac = round(np.random.uniform(360, 0), 2)
    random_ra.append(rightac)

random_dec = []
for i in range(10000):
    declination = round(np.random.uniform(-90, 90), 2)
    random_dec.append(declination)



ra_0 = Longitude(random_ra, unit=u.deg)  # Could also use Angle
dec_0 = np.array(random_dec) * u.deg  # Astropy Quantity
coordinates = SkyCoord(ra_0, dec_0, frame='icrs')
coordinates = SkyCoord(frame=ICRS, ra=ra_0, dec=dec_0, obstime='2001-01-02T12:34:56')



result = crossmatch(sky_map, coordinates)
probdens = result.probdensity 

#Stripping the data type, lattitude and logitude and recasting into an ndarray ready for interpolation
points = [[coordinates.ra[i].value,coordinates.dec[i].value] for i in range(len(coordinates.ra))]


#print(cat[result.searched_prob_vol < 0.9])


ragrid = 480
decgrid = 240

ra = np.linspace(360,0,ragrid) 
dec =  np.linspace(-90,90,decgrid)
ra,dec = np.meshgrid(ra,dec)

#Interpolating onto the new grid
reshaped = interpolate.griddata(points,probdens,[[ra.flatten()[i],dec.flatten()[i]] for i in range(len(ra.flatten()))])

#Had to flatten the meshgrid to interpolate, now restructuring.
arrayform = reshaped.reshape(decgrid,ragrid)

ax = plt.subplot(1,1,1, projection=WCS(target_header),frame_class=EllipticalFrame)
ax.imshow(arrayform, vmin=0.9, vmax=1, cmap='plasma')

lon = ax.coords[0]
lon.set_ticks(number=9)
ax.grid()
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

# %%

# %%

# %%

# %%
