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
import ligo.skymap.plot
import pandas as pd
from matplotlib import pyplot as plt
from astroquery.vizier import VizierClass
from scipy import interpolate
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
import astropy.units as u
from matplotlib.colors import LinearSegmentedColormap

sky_map =  read_sky_map('GW190814_skymap.fits.gz', moc=True)

df_result = pd.read_csv('GW190814.csv', engine='python')
print("started with", len(df_result))
def get_mags(flux, fibre_flux, transmission, flux_ivar):
    mag = 22.5 - 2.5*np.log10(flux/transmission)
    mag_err = np.abs(2.5/(np.log(10)*flux/transmission*np.sqrt(flux_ivar)))
    fibre_mag = 22.5 - 2.5*np.log10(fibre_flux/transmission)
    return mag, mag_err, fibre_mag

mag_g, mag_g_err, fibre_mag_g = get_mags(df_result['FLUX_G'], df_result['FIBERFLUX_G'], df_result['MW_TRANSMISSION_G'], df_result['FLUX_IVAR_G'])
mag_r, mag_r_err, fibre_mag_r = get_mags(df_result['FLUX_R'], df_result['FIBERFLUX_R'], df_result['MW_TRANSMISSION_R'], df_result['FLUX_IVAR_R'])
mag_z, mag_z_err, fibre_mag_z = get_mags(df_result['FLUX_Z'], df_result['FIBERFLUX_Z'], df_result['MW_TRANSMISSION_Z'], df_result['FLUX_IVAR_Z'])

filter_ra_dec_bigblock =  (df_result['RA'] >9.0) & (df_result['RA'] < 17) &  (df_result['DEC'] < -21.0) & (df_result['DEC'] > -28) 
filter_ra_dec_smallblock = (df_result['RA'] >18.0) & (df_result['RA'] < 27.0) &  (df_result['DEC'] < -29.0) & (df_result['DEC'] > -35) 


#MAG_R cut mag_r<20.5
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
allbgsmask2 = basic_cuts & bgs_magcut & bgs_fibremag_cut & filter_gaia & filter_fracmasked & filter_bgsmask & filter_fracflux & filter_fracin & filter_gaia_bright &filter_ra_dec_smallblock

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

coordinates = SkyCoord([(float(x))*u.deg for x in (filtered_ra.tolist())],[float(x)*u.deg for x in filtered_dec.tolist()])#,[float(x) for x in df_result[2][1:].tolist()])
result = crossmatch(sky_map, coordinates)
searched_prob = result.searched_prob

points = [[(coordinates.ra[i].value)%180 - (coordinates.ra[i].value)//180*180,coordinates.dec[i].value] for i in range(len(coordinates.ra))]



ragrid = 240*3
decgrid = 120*3
ra = np.linspace(-180,180,ragrid) 
dec =  np.linspace(-90,90,decgrid)
ra,dec = np.meshgrid(ra,dec)
searched_prob_update = []
""" 
for j in range(len(searched_prob)):
        x = 1-searched_prob[j]
        if x >= 0.5:
                searched_prob_update.append(x)
        else:
                searched_prob_update.append(0.0)
         """
#reshaped_over50 = interpolate.griddata(points,searched_prob_update,[[ra.flatten()[i],dec.flatten()[i]] for i in range(len(ra.flatten()))],fill_value = 0.0)
reshaped = interpolate.griddata(points,1-searched_prob,[[ra.flatten()[i],dec.flatten()[i]] for i in range(len(ra.flatten()))],fill_value = 0.0)
raplot = np.linspace(-np.pi,np.pi,ragrid) 
decplot =  np.linspace(-np.pi/2,np.pi/2,decgrid)
raplot,decplot = np.meshgrid(raplot,decplot)
arrayform = np.ma.masked_invalid(reshaped.reshape(decgrid,ragrid))
#arrayform_over50 = np.ma.masked_invalid(reshaped_over50.reshape(decgrid,ragrid))


probability_entirearea = (1-searched_prob)
print("there are", sum((probability_entirearea>0.9)), "points above 99%")
print("there are", sum((probability_entirearea>0.90)), "points above 90%")
print("there are", sum((probability_entirearea>0.80)), "points above 80%")
print("there are", sum((probability_entirearea>0.40)), "points above 40%")

#%%


fig, ax = plt.subplots(figsize=(15,10)) 
ax= plt.subplot(projection="mollweide")
cb = ax.pcolormesh(raplot,decplot,arrayform, vmin = 0.0, vmax = 1.0, cmap='cylon')
ax.grid()
cbar = plt.colorbar(cb,fraction=0.025)


""" axins = ax.inset_axes(
        [1.0, 0.2, 1.0, 1.5], transform=ax.transData)   
axins.pcolormesh(raplot,decplot,reshaped.reshape(decgrid,ragrid), cmap='cylon')
x1, x2, y1, y2 = 18.25/(180/np.pi), 26.25/(180/np.pi), -28.5/(180/np.pi), -36/(180/np.pi)
axins.set_xlim(x1, x2)
axins.set_ylim(y2, y1)
axins.set_xticks([])
axins.set_yticks([])
axins.set_xticklabels([])
axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")
 """


axins2 = ax.inset_axes(
        [-1.7, 0.2, 1.5, 1.5], transform=ax.transData)   
axins2.pcolormesh(raplot,decplot,reshaped.reshape(decgrid,ragrid), cmap='cylon')
x1, x2, y1, y2 = 8.4/(180/np.pi), 16.9/(180/np.pi), -20.5/(180/np.pi), -28.5/(180/np.pi)
axins2.set_xlim(x1, x2)
axins2.set_ylim(y2, y1)
axins2.set_xticks([])
axins2.set_yticks([])
axins2.set_xticklabels([])
axins2.set_yticklabels([])
ax.indicate_inset_zoom(axins2, edgecolor="black")

for i in range(0,10):
        target = pd.read_csv('GW190814_tile'+str(i)+'_targets.txt', delim_whitespace=True, header= 0, index_col= False)
        target = target.apply(pd.to_numeric, errors='coerce')
        axins2.scatter([(x%180-x//180*180)*np.pi/180 for x in target['RA']],[x*np.pi/180 for x in target['DEC']], marker = '.', s=0.1, color = 'lightsteelblue')

 
allguide = []
for i in range(0,9):
        guide = pd.read_csv('GW190814_tile'+str(i)+'_stars.txt', delim_whitespace=True, header= 0, index_col= False)
        guide = guide.apply(pd.to_numeric, errors='coerce')
        axins2.scatter([(x%180-x//180*180)*np.pi/180 for x in guide['RA']],[x*np.pi/180 for x in guide['DEC']], marker = '.', s=10, color = 'yellow')
        allguide.append(guide)

#########################
RADECS = []
Centershift = [0.0, 0.35]
Centers = [12.75+Centershift[0],-25.35+Centershift[1]]
scale = 0.25
RADEC = [Centers[0],Centers[1]]
RADECS.append((RADEC[0],RADEC[1]))
RADIUS = 1
l = RADEC[0]/(180/np.pi)
b = (RADEC[1])/(180/np.pi)
F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
ax.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])
axins2.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])

RADEC = [Centers[0]+0.5*np.sqrt(3)*scale,Centers[1]+3/2*scale]
RADECS.append((RADEC[0],RADEC[1]))
RADIUS = 1
l = RADEC[0]/(180/np.pi)
b = (RADEC[1])/(180/np.pi)
F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
#ax.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])
axins2.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])

RADEC = [Centers[0]+np.sqrt(3)*scale,Centers[1]]
RADECS.append((RADEC[0],RADEC[1]))
RADIUS = 1
l = RADEC[0]/(180/np.pi)
b = (RADEC[1])/(180/np.pi)
F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
#ax.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])
axins2.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])

RADEC = [Centers[0]+0.5*np.sqrt(3)*scale,Centers[1]-3/2*scale]
RADECS.append((RADEC[0],RADEC[1]))
RADIUS = 1
l = RADEC[0]/(180/np.pi)
b = (RADEC[1])/(180/np.pi)
F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
#ax.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])
axins2.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])

RADEC = [Centers[0]+1.5*np.sqrt(3)*scale,Centers[1]-3/2*scale]
RADECS.append((RADEC[0],RADEC[1]))
RADIUS = 1
l = RADEC[0]/(180/np.pi)
b = (RADEC[1])/(180/np.pi)
F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
#ax.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])
axins2.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])

RADEC = [Centers[0]-0.5*np.sqrt(3)*scale,Centers[1]+3/2*scale]
RADECS.append((RADEC[0],RADEC[1]))
RADIUS = 1
l = RADEC[0]/(180/np.pi)
b = (RADEC[1])/(180/np.pi)
F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
#ax.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])
axins2.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])

RADEC = [Centers[0]-0.5*np.sqrt(3)*scale,Centers[1]-3/2*scale]
RADECS.append((RADEC[0],RADEC[1]))
RADIUS = 1
l = RADEC[0]/(180/np.pi)
b = (RADEC[1])/(180/np.pi)
F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
#ax.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])
axins2.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])

RADEC = [Centers[0]-np.sqrt(3)*scale,Centers[1]]
RADECS.append((RADEC[0],RADEC[1]))
RADIUS = 1
l = RADEC[0]/(180/np.pi)
b = (RADEC[1])/(180/np.pi)
F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
#ax.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])
axins2.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])

RADEC = [Centers[0]-1.5*np.sqrt(3)*scale,Centers[1]+3/2*scale]
RADECS.append((RADEC[0],RADEC[1]))
RADIUS = 1
l = RADEC[0]/(180/np.pi)
b = (RADEC[1])/(180/np.pi)
F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
#ax.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])
axins2.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])

RADEC = [Centers[0],Centers[1]+3*scale]
RADECS.append((RADEC[0],RADEC[1]))
RADIUS = 1
l = RADEC[0]/(180/np.pi)
b = (RADEC[1])/(180/np.pi)
F = (np.cos(b)*(raplot-l))**2 + (decplot-b)**2 - (RADIUS/(180/np.pi))**2
#ax.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])
axins2.contour(raplot,decplot,F,[0], linewidths=1.0, colors = ['blue'])


axins2.plot(12.75/(180/np.pi),-25.35/(180/np.pi), 'o', color = 'limegreen')
plt.show()
#fig.savefig('GW190814_tiles.png', dpi=300)


# %%
