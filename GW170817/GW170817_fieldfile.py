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
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
import astropy.units as u
from matplotlib.colors import LinearSegmentedColormap


sky_map =  read_sky_map('GW170817_skymap.fits', moc=True)

df_result = pd.read_csv('GW170817.csv', engine='python')
print("started with", len(df_result))
def get_mags(flux, fibre_flux, transmission, flux_ivar):
    mag = 22.5 - 2.5*np.log10(flux/transmission)
    mag_err = np.abs(2.5/(np.log(10)*flux/transmission*np.sqrt(flux_ivar)))
    fibre_mag = 22.5 - 2.5*np.log10(fibre_flux/transmission)
    return mag, mag_err, fibre_mag

mag_g, mag_g_err, fibre_mag_g = get_mags(df_result['FLUX_G'], df_result['FIBERFLUX_G'], df_result['MW_TRANSMISSION_G'], df_result['FLUX_IVAR_G'])
mag_r, mag_r_err, fibre_mag_r = get_mags(df_result['FLUX_R'], df_result['FIBERFLUX_R'], df_result['MW_TRANSMISSION_R'], df_result['FLUX_IVAR_R'])
mag_z, mag_z_err, fibre_mag_z = get_mags(df_result['FLUX_Z'], df_result['FIBERFLUX_Z'], df_result['MW_TRANSMISSION_Z'], df_result['FLUX_IVAR_Z'])


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
#bgs_fibremag_cut = ((fibre_mag_r < (5.1 + mag_r)) & (mag_r <= 17.8)) | ((fibre_mag_r < 22.9) & (mag_r > 17.8) & (mag_r < 20))


allbgsmask1 = basic_cuts & bgs_magcut & filter_gaia & filter_fracmasked & filter_bgsmask & filter_fracflux & filter_fracin & filter_gaia_bright #& filter_ra_dec_bigblock

filtered_data1 = df_result.loc[allbgsmask1]

print("first block is", len(filtered_data1))

filtered_ra1 = filtered_data1['RA']
filtered_dec1 = filtered_data1['DEC']


coordinates1 = SkyCoord([(float(x))*u.deg for x in (filtered_ra1.tolist())],[float(x)*u.deg for x in filtered_dec1.tolist()])#,[float(x) for x in df_result[2][1:].tolist()])
Centers = [196.8906399999999906,-24.0086059999999968] #NGC4993 coords

#coordinates = SkyCoord([(float(x))*u.deg for x in (filtered_ra.tolist())],[float(x)*u.deg for x in filtered_dec.tolist()])#,[float(x) for x in df_result[2][1:].tolist()])

result1 = crossmatch(sky_map, coordinates1)
searched_prob1 = result1.searched_prob

print(len(filtered_data1))
cutdown = filtered_data1[result1.searched_prob < 1.0]
print(len(cutdown))

radius = 3.0
cutdown['localised'] = ((cutdown["RA"]-Centers[0])**2 + (cutdown["DEC"]-Centers[1])**2 < radius**2)
cutdown = cutdown[cutdown['localised'] == True]
print(len(cutdown))



points1 = [[(coordinates1.ra[i].value)%180 - (coordinates1.ra[i].value)//180*180,coordinates1.dec[i].value] for i in range(len(coordinates1.ra))]


ragrid = 240*3
decgrid = 120*3       
ra = np.linspace(-180,180,ragrid) 
dec =  np.linspace(-90,90,decgrid)
ra,dec = np.meshgrid(ra,dec)



#Interpolating onto the new grid
reshaped1 = interpolate.griddata(points1,1-searched_prob1,[[ra.flatten()[i],dec.flatten()[i]] for i in range(len(ra.flatten()))],fill_value = 0.0)

raplot = np.linspace(-np.pi,np.pi,ragrid) 
decplot =  np.linspace(-np.pi/2,np.pi/2,decgrid)
raplot,decplot = np.meshgrid(raplot,decplot)

#flattened the meshgrid to interpolate, now restructuring.
arrayform1 = np.ma.masked_invalid(reshaped1.reshape(decgrid,ragrid))


probability_entirearea = np.concatenate([(1-searched_prob1)])
print("there are", sum((probability_entirearea>0.9)), "points above 99%")
print("there are", sum((probability_entirearea>0.90)), "points above 90%")
print("there are", sum((probability_entirearea>0.80)), "points above 80%")
print("there are", sum((probability_entirearea>0.70)), "points above 70%")

#%%
fig, ax = plt.subplots(figsize=(15,10)) 
ax= plt.subplot(projection="mollweide")
cb = ax.pcolormesh(raplot,decplot,arrayform1, vmin = 0.0, vmax = 1.0, cmap='cylon')
#ax.scatter(cat['_RAJ2000'],cat['_DEJ2000'], marker = '.', s=1, color = 'white' )
ax.grid()
cbar = plt.colorbar(cb,fraction=0.025)
#ax.scatter([float((x)%180 - (x)//180*180)*2*np.pi/360 for x in (cutdown['RA'].tolist())],[float(x)*2*np.pi/360 for x in (cutdown['DEC'].tolist())], marker='.', s=0.01)



axins2 = ax.inset_axes(
        [-1.7, 0.2, 1.5, 1.5], transform=ax.transData)   
axins2.pcolormesh(raplot,decplot,arrayform1, vmin = 0.0, vmax = 1.0, cmap='cylon')
x1tocycle = min(filtered_data1["RA"])/(180/np.pi)
x2tocycle = max(filtered_data1["RA"])/(180/np.pi)
x1, x2, y1, y2 = -170/(180/np.pi), -155/(180/np.pi), -20.5/(180/np.pi), -28/(180/np.pi)
#x1, x2, y1, y2 = (x1tocycle)%np.pi - (x1tocycle)//np.pi*np.pi, (x2tocycle)%np.pi - (x2tocycle)//np.pi*np.pi, max(filtered_data1["DEC"])/(180/np.pi), min(filtered_data1["DEC"])/(180/np.pi)
axins2.set_xlim(x1, x2)
axins2.set_ylim(y2, y1)
axins2.set_xticks([])
axins2.set_yticks([])
axins2.set_xticklabels([])
axins2.set_yticklabels([])
ax.indicate_inset_zoom(axins2, edgecolor="black")
#ax.set_xticklabels(tick_labels)
#axins2.scatter([float((x)%180 - (x)//180*180)*2*np.pi/360 for x in (cutdown['RA'].tolist())],[float(x)*2*np.pi/360 for x in (cutdown['DEC'].tolist())], marker='.', s=0.01)


#using galaxy NGC4993 

#########################
Centers = [196.89-360,-24.0] #NGC4993 coords
axins2.scatter(Centers[0]/(180/np.pi), Centers[1]/(180/np.pi), s = 50, color= 'limegreen', marker = '.')



for i in range(0,7):
        guide = pd.read_csv('GW170817_tile'+str(i)+'_stars.txt', delim_whitespace=True, header= 0, index_col= False)
        guide = guide.apply(pd.to_numeric, errors='coerce')
        axins2.scatter([(x%180-x//180*180)*np.pi/180 for x in guide['RA']],[x*np.pi/180 for x in guide['DEC']], marker = '.', s=10, color = 'yellow')
        #ax.scatter([(x%180-x//180*180)*np.pi/180 for x in guide['RA']],[x*np.pi/180 for x in guide['DEC']], marker = '.', s=10, color = 'limegreen')


for i in range(0,7):
        target = pd.read_csv('GW170817_tile'+str(i)+'_targets.txt', delim_whitespace=True, header= 0, index_col= False)
        target = target.apply(pd.to_numeric, errors='coerce')
        axins2.scatter([(x%180-x//180*180)*np.pi/180 for x in target['RA']],[x*np.pi/180 for x in target['DEC']], marker = '.', s=0.1, color = 'lightsteelblue')
        #ax.scatter([(x%180-x//180*180)*np.pi/180 for x in target['RA']],[x*np.pi/180 for x in target['DEC']], marker = '.', s=10, color = 'limegreen')

""" for i in range(0,8):
        sky = pd.read_csv('GW170817_tile'+str(i)+'_sky.txt', delim_whitespace=True, header= 0, index_col= False)
        sky = sky.apply(pd.to_numeric, errors='coerce')
        axins2.scatter([(x%180-x//180*180)*np.pi/180 for x in sky['_RAJ2000']],[x*np.pi/180 for x in sky['_DEJ2000']], marker = '.', s=1, color = 'blue')
        #ax.scatter([(x%180-x//180*180)*np.pi/180 for x in target['RA']],[x*np.pi/180 for x in target['DEC']], marker = '.', s=10, color = 'limegreen')
 """



RADECS = []
center_best = [(196.845152259717-360, -23.167866701867933),
 (197.0237950033236-360, -23.330102203564827),
 (196.66650951611038-360, -23.005631200171035),
 (197.4704018623401-360, -24.06016196120086),
 (196.5771881443071-360, -23.005631200171035),
 (197.73836597775002-360, -24.709103967988444),
 (197.64904460594673-360, -24.790221718836893),
 (196.9344736315203-360, -22.92451344932259)]

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
#plt.savefig("result.png", dpi=300)
#fig.savefig('GW170817_tiles.png', dpi=300)




## %%

# %%
