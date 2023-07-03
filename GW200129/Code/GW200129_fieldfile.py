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
import pandas as pd
import ligo.skymap.plot
from matplotlib import pyplot as plt
from astroquery.vizier import VizierClass
from scipy import interpolate
import os
import warnings
from datetime import date
from types import MethodType
import numpy as np
import numpy.ma
import matplotlib.pyplot as plt
import matplotlib.dates
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, colorConverter
from matplotlib.patches import Ellipse
from astropy.coordinates import SkyCoord, HeliocentricTrueEcliptic, ICRS, Longitude
import astropy.units as u
from astropy.time import Time
from astropy.utils import iers
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
import astropy.units as u
from matplotlib.colors import LinearSegmentedColormap

sky_map =  read_sky_map('GW200129_skymap.fits', moc=True)

df_result = pd.read_csv('GW200129.csv', engine='python')
print("started with", len(df_result))
def get_mags(flux, fibre_flux, transmission, flux_ivar):
    mag = 22.5 - 2.5*np.log10(flux/transmission)
    mag_err = np.abs(2.5/(np.log(10)*flux/transmission*np.sqrt(flux_ivar)))
    fibre_mag = 22.5 - 2.5*np.log10(fibre_flux/transmission)
    return mag, mag_err, fibre_mag

mag_g, mag_g_err, fibre_mag_g = get_mags(df_result['FLUX_G'], df_result['FIBERFLUX_G'], df_result['MW_TRANSMISSION_G'], df_result['FLUX_IVAR_G'])
mag_r, mag_r_err, fibre_mag_r = get_mags(df_result['FLUX_R'], df_result['FIBERFLUX_R'], df_result['MW_TRANSMISSION_R'], df_result['FLUX_IVAR_R'])
mag_z, mag_z_err, fibre_mag_z = get_mags(df_result['FLUX_Z'], df_result['FIBERFLUX_Z'], df_result['MW_TRANSMISSION_Z'], df_result['FLUX_IVAR_Z'])

filter_ra_dec_bigblock =  (df_result['RA'] >315.0) & (df_result['RA'] < 325.0) &  (df_result['DEC'] < 20.0) & (df_result['DEC'] > -3.0) 
filter_ra_dec_smallblock = (df_result['RA'] >290.0) & (df_result['RA'] < 315.0) &  (df_result['DEC'] < 35.0) & (df_result['DEC'] > 10.0) 


help_LS_rr = 22.5 - 2.5*np.log10(df_result['FLUX_R'])
filter_gaia_bright = ((df_result['GAIA_PHOT_G_MEAN_MAG'] == 0))
filter_gaia = ((df_result['GAIA_PHOT_G_MEAN_MAG'] - help_LS_rr) > 0.6) | filter_gaia_bright
filter_bgsmask = (df_result['MASKBITS'] & (2**1) == 0) & (df_result['MASKBITS'] & (2**12) ==0) & (df_result['MASKBITS'] & (2**13) == 0)
filter_fracmasked = (df_result['FRACMASKED_G'] < 0.4) & (df_result['FRACMASKED_R'] < 0.4) & (df_result['FRACMASKED_Z'] < 0.4)
filter_fracflux = (df_result['FRACFLUX_G'] < 5.0) & (df_result['FRACFLUX_R'] < 5.0) & (df_result['FRACFLUX_Z'] < 5.0)
filter_fracin = (df_result['FRACIN_G'] > 0.3) & (df_result['FRACIN_R'] > 0.3) & (df_result['FRACIN_Z'] > 0.3)


obsfilter = (df_result['NOBS_G'] > 0) & (df_result['NOBS_R'] > 0) & (df_result['NOBS_Z'] > 0)
filter_minflux = (df_result['FLUX_G'] > 0) & (df_result['FLUX_R'] > 0) & (df_result['FLUX_Z'] > 0)
basic_cuts = obsfilter &  filter_minflux


gmr, rmz = mag_g - mag_r, mag_r - mag_z
bgs_magcut = (-1 < gmr) & (gmr < 4) & (-1 < rmz) & (rmz < 4)
bgs_fibremag_cut = ((fibre_mag_r < (5.1 + mag_r)) & (mag_r <= 17.8)) | ((fibre_mag_r < 22.9) & (mag_r > 17.8) & (mag_r < 20))


allbgsmask1 = basic_cuts & bgs_magcut & bgs_fibremag_cut & filter_gaia & filter_fracmasked & filter_bgsmask & filter_fracflux & filter_fracin & filter_gaia_bright & filter_ra_dec_bigblock
allbgsmask2 = basic_cuts & bgs_magcut & bgs_fibremag_cut & filter_gaia & filter_fracmasked & filter_bgsmask & filter_fracflux & filter_fracin & filter_gaia_bright & filter_ra_dec_smallblock

filtered_data1 = df_result.loc[allbgsmask1]
filtered_data2 = df_result.loc[allbgsmask2]

print("first block is", len(filtered_data1))
print("second block is", len(filtered_data2))
print("total points are", len(filtered_data1)+len(filtered_data2))
filtered_ra1 = filtered_data1['RA']
filtered_dec1 = filtered_data1['DEC']

filtered_ra2 = filtered_data2['RA']
filtered_dec2 = filtered_data2['DEC']

filtered_ra = pd.concat([filtered_ra1, filtered_ra2])
filtered_dec = pd.concat([filtered_dec1, filtered_dec2])

coordinates1 = SkyCoord([(float(x))*u.deg for x in (filtered_ra1.tolist())],[float(x)*u.deg for x in filtered_dec1.tolist()])#,[float(x) for x in df_result[2][1:].tolist()])
coordinates2 = SkyCoord([(float(x))*u.deg for x in (filtered_ra2.tolist())],[float(x)*u.deg for x in filtered_dec2.tolist()])#,[float(x) for x in df_result[2][1:].tolist()])

coordinates = np.concatenate([coordinates1, coordinates2])
#coordinates = SkyCoord([(float(x))*u.deg for x in (filtered_ra.tolist())],[float(x)*u.deg for x in filtered_dec.tolist()])#,[float(x) for x in df_result[2][1:].tolist()])

result1 = crossmatch(sky_map, coordinates1)
searched_prob1 = result1.searched_prob

result2 = crossmatch(sky_map, coordinates2)
searched_prob2 = result2.searched_prob 

searched_prob = np.concatenate([searched_prob1, searched_prob2])


points1 = [[(coordinates1.ra[i].value)%180 - (coordinates1.ra[i].value)//180*180,coordinates1.dec[i].value] for i in range(len(coordinates1.ra))]
points2 = [[(coordinates2.ra[i].value)%180 - (coordinates2.ra[i].value)//180*180,coordinates2.dec[i].value] for i in range(len(coordinates2.ra))]

ragrid = 240*3
decgrid = 120*3
ra = np.linspace(-180,180,ragrid) 
dec =  np.linspace(-90,90,decgrid)
ra,dec = np.meshgrid(ra,dec)


#Interpolating onto the new grid
reshaped1 = interpolate.griddata(points1,1-searched_prob1,[[ra.flatten()[i],dec.flatten()[i]] for i in range(len(ra.flatten()))],fill_value = 0.0)
reshaped2 = interpolate.griddata(points2,1-searched_prob2,[[ra.flatten()[i],dec.flatten()[i]] for i in range(len(ra.flatten()))])

raplot = np.linspace(-np.pi,np.pi,ragrid) 
decplot =  np.linspace(-np.pi/2,np.pi/2,decgrid)
raplot,decplot = np.meshgrid(raplot,decplot)

#flattened the meshgrid to interpolate, now restructuring.
arrayform1 = np.ma.masked_invalid(reshaped1.reshape(decgrid,ragrid))
arrayform2 = np.ma.masked_invalid(reshaped2.reshape(decgrid,ragrid))


probability_entirearea = np.concatenate([(1-searched_prob1), (1-searched_prob2)])
print("there are", sum((probability_entirearea>0.99)), "points above 99%")
print("there are", sum((probability_entirearea>0.90)), "points above 90%")
print("there are", sum((probability_entirearea>0.80)), "points above 80%")
print("there are", sum((probability_entirearea>0.70)), "points above 70%")

df_200129 = pd.read_csv('GW200129.txt', delim_whitespace=True, header= 0, index_col= False)
df_200129 = df_200129.apply(pd.to_numeric, errors='coerce')

#%%
fig, ax = plt.subplots(figsize=(15,10)) 
ax= plt.subplot(projection="mollweide")
#ax.pcolormesh(raplot,decplot,arrayform2, vmin = 0.0, vmax = 1.0, cmap='cylon')
cb = ax.pcolormesh(raplot,decplot,arrayform1, vmin = 0.0, vmax = 1.0, cmap='cylon')
ax.grid()
cbar = plt.colorbar(cb,fraction=0.025)

axins2 = ax.inset_axes(
        [30/(180/np.pi), -60/(180/np.pi), 1.2, 1.6], transform=ax.transData)   
axins2.pcolormesh(raplot,decplot,arrayform1, vmin = 0.0, vmax = 1.0, cmap='cylon')
x1tocycle = min(filtered_data1["RA"])/(180/np.pi)
x2tocycle = max(filtered_data1["RA"])/(180/np.pi)
x1, x2, y1, y2 = -50/(180/np.pi), -30/(180/np.pi), -2.5/(180/np.pi), 16.5/(180/np.pi)
#x1, x2, y1, y2 = (x1tocycle)%np.pi - (x1tocycle)//np.pi*np.pi, (x2tocycle)%np.pi - (x2tocycle)//np.pi*np.pi, max(filtered_data1["DEC"])/(180/np.pi), min(filtered_data1["DEC"])/(180/np.pi)
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)
axins2.set_xticks([])
axins2.set_yticks([])
axins2.set_xticklabels([])
axins2.set_yticklabels([])
ax.indicate_inset_zoom(axins2, edgecolor="black")
#axins2.scatter([(x%180-x//180*180)*np.pi/180 for x in df_200129['RA']],[x*np.pi/180 for x in df_200129['DEC']], marker = '.', s=0.001, color = 'lightblue')


for i in range(0,4):
        target = pd.read_csv('GW200129_tile'+str(i)+'_targets.txt', delim_whitespace=True, header= 0, index_col= False)
        target = target.apply(pd.to_numeric, errors='coerce')
        axins2.scatter([(x%180-x//180*180)*np.pi/180 for x in target['RA']],[x*np.pi/180 for x in target['DEC']], marker = '.', s=0.1, color = 'lightsteelblue')
 
for i in range(0,4):
        guide = pd.read_csv('GW200129_tile'+str(i)+'_stars.txt', delim_whitespace=True, header= 0, index_col= False)
        guide = guide.apply(pd.to_numeric, errors='coerce')
        axins2.scatter([(x%180-x//180*180)*np.pi/180 for x in guide['RA']],[x*np.pi/180 for x in guide['DEC']], marker = '.', s=1, color = 'yellow')

""" np.unravel_index(searched_prob.argmin(), searched_prob.shape)
coordinates[30875] """

#maximum prob pixel (ra, dec) in deg (318.36951858, 5.16587679)>
Centers = [318.36951858-360, 5.16587679] #highest prob pixel

axins2.scatter(Centers[0]/(180/np.pi), Centers[1]/(180/np.pi), s = 50, color= 'limegreen', marker = '.')

RADECS = []

center_best = [(318.46212680725296-360, 5.077386916833511),
 (317.90244207413076-360, 6.783337350050521),
 (318.64868838496034-360, 4.30195490173487),
 (317.77806768899245-360, 7.248596559109705)]

for i, j in center_best:
        Centers = [i,j]
        RADEC = [i,j]
        RADECS.append((RADEC[0],RADEC[1]))
        RADIUS = 1
        l = RADEC[0]/(180/np.pi)
        b = (RADEC[1])/(180/np.pi)
        F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
        #F = (raplot-l)**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
        axins2.contour(raplot,decplot,F,[0], linewidths=0.5, colors = ['blue'])



plt.show()
#fig.savefig('circles.png', dpi=300)
#fig.savefig('GW200129_tiles.png', dpi=300)




# %%
