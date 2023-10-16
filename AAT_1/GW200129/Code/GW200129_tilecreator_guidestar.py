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
from astroquery.vizier import Vizier
from scipy import interpolate
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
import astropy.units as u
from matplotlib.colors import LinearSegmentedColormap
import random


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


#filter_ra_dec_bigblock =  (df_result['RA'] >315.0) & (df_result['RA'] < 325.0) &  (df_result['DEC'] < 20.0) & (df_result['DEC'] > -3.0) 
#filter_ra_dec_smallblock = (df_result['RA'] >290.0) & (df_result['RA'] < 315.0) &  (df_result['DEC'] < 35.0) & (df_result['DEC'] > 10.0) 
filter_ra_dec = (df_result['RA']>360-50) & (df_result['RA'] < 360-30) & (df_result['DEC']> -2.5) & (df_result['DEC']< 16.5)

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


allbgsmask = basic_cuts & bgs_magcut & bgs_fibremag_cut & filter_gaia & filter_fracmasked & filter_bgsmask & filter_fracflux & filter_fracin & filter_gaia_bright & filter_ra_dec
#allbgsmask2 = basic_cuts & bgs_magcut & bgs_fibremag_cut & filter_gaia & filter_fracmasked & filter_bgsmask & filter_fracflux & filter_fracin & filter_gaia_bright & filter_ra_dec_smallblock

filtered_data1 = df_result.loc[allbgsmask]

#filtered_data2 = df_result.loc[allbgsmask2]

filtered_data = filtered_data1
#filtered_data = pd.concat([filtered_data1,filtered_data2])


#filter_ra_dec_entire = (filtered_data['RA']>360-50) & (filtered_data['RA'] < 360-30) & (filtered_data['DEC']> -2.5) & (filtered_data['DEC']< 16.5)
         
         
filtered_ra = filtered_data['RA']
filtered_dec = filtered_data['DEC']


coordinates = SkyCoord([(float(x))*u.deg for x in (filtered_ra.tolist())],[float(x)*u.deg for x in filtered_dec.tolist()])#,[float(x) for x in df_result[2][1:].tolist()])

#coordinates = SkyCoord([(float(x))*u.deg for x in (filtered_ra.tolist())],[float(x)*u.deg for x in filtered_dec.tolist()])#,[float(x) for x in df_result[2][1:].tolist()])

result = crossmatch(sky_map, coordinates)
searched_prob200129 = result.searched_prob[result.searched_prob < 0.6]

cutdown = filtered_data[result.searched_prob < 0.6]
print(len(cutdown))
center = [318.36951858, 5.16587679]

radius_search = 4.0
cutdown['PROB'] = 1-searched_prob200129
cutdown['LOCAL'] = ((np.cos(cutdown["DEC"]*np.pi/180)*(cutdown["RA"]-center[0]))**2 + (cutdown["DEC"]-center[1])**2 < radius_search**2)
cutdown['SEP'] = np.sqrt((np.cos(cutdown["DEC"]*np.pi/180)*(cutdown["RA"]-center[0]))**2 + (cutdown["DEC"]-center[1])**2)
cutdown = cutdown[cutdown['LOCAL'] == True]
localised_Sorted200129 = cutdown.sort_values(by=['PROB'], ascending = 0)
labels= ["GW200129"]

localised_Sorted200129.to_csv(labels[0]+'.txt', index=None, sep=' ', mode='w')
##############
df_200129 = pd.read_csv('GW200129.txt', delim_whitespace=True, header= 0, index_col= False)
df_200129 = df_200129.apply(pd.to_numeric, errors='coerce')

ra_grid = np.linspace(min(df_200129['RA']), max(df_200129['RA']), 50)
dec_grid = np.linspace(min(df_200129['DEC']), max(df_200129['DEC']), 50)
ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
radius_tile = 1
drop = df_200129.drop(df_200129.index)

center_best = []
number_tiles = 4
labels = [0, 1, 2, 3]
for tile_num in zip(labels,range(number_tiles)):
    areaprob_200129 = 0.0
    for center_x, center_y in zip(ra_mesh.flatten(), dec_mesh.flatten()):
        cutdown= pd.concat([df_200129, drop])
        cutdown_200129 = cutdown[cutdown.duplicated(keep=False)==False]
        truth = (np.cos(center_y*np.pi/180)*(cutdown_200129["RA"]-center_x))**2 + (cutdown_200129["DEC"]-center_y)**2 < radius_tile**2
        inside_circ = cutdown_200129[truth]
        if inside_circ['PROB'].sum() > areaprob_200129:
            areaprob_200129 = (inside_circ['PROB'].sum())
            try:
                top500 = inside_circ.sample(500)
            except: 
                top500 = inside_circ.sample(len(inside_circ))
            top500 = top500[top500.duplicated(keep=False)==False]
            center_x_save = center_x
            center_y_save = center_y
    top500tosave = pd.concat([top500, pd.Series()], ignore_index=True)
    top500tosave.to_csv('GW200129_tile'+str(tile_num[1])+'_targets.txt', index=None, sep=' ', mode='w')
    center_best.append((center_x_save, center_y_save))
    drop = pd.concat([drop, top500])
    
tile = ['Tile ' + str(i) for i in range(number_tiles)]

with open("GW200129_centres.txt", "w") as output:
    for x, y in zip(tile,center_best):
        output.write(str(x) + '\t' +str(y) + "\n")


centers_200129 = center_best
Vizier.ROW_LIMIT = -1

label = [0, 1, 2, 3]
for circlecenter in zip(centers_200129, label):
    result = Vizier.query_constraints(catalog='I/322A/out', of='0', db='0', pmRA = '-10 .. 10', pmDE = '-10 .. 10',
                                 Vmag = '11 .. 13', RAJ2000 = (str(circlecenter[0][0]-1)) + ' .. ' + str(circlecenter[0][0]+1), 
                                 DEJ2000 = str(circlecenter[0][1]-1) + ' .. ' + str(circlecenter[0][1]+1))
    for table_name in result.keys():
        query_result = result[table_name]
        #print(query_result)
    query_result['LOCAL'] = ((np.cos(circlecenter[0][1]/(180/np.pi))*(query_result["RAJ2000"]-circlecenter[0][0]))**2 + (query_result["DEJ2000"]-circlecenter[0][1])**2 < radius_tile**2)
    localised_query = query_result[query_result['LOCAL'] == True]
    UCAC4 = []
    RA = []
    DEC = []
    MAG_R = []
    Vmag = []
    pmRA = []
    pmDE = []
    for i in localised_query:
        UCAC4.append(i['UCAC4'])
        RA.append(i['RAJ2000'])
        DEC.append(i['DEJ2000'])
        MAG_R.append(i['rmag'])
        Vmag.append(i['Vmag'])
        pmRA.append(i['pmRA'])  
        pmDE.append(i['pmDE'])
    data = {'UCAC4':UCAC4, 'RA':RA, 'DEC':DEC, 'MAG_R':MAG_R, 'Vmag':Vmag, 'pmRA':pmRA, 'pmDE':pmDE}
    #sort_mag_r = pd.DataFrame(data)
    random_30 = pd.DataFrame(data).sample(30)
    #sort_mag_r.sort_values(by='MAG_R', ascending=True)
    #brightest_30 = sort_mag_r[:30]
    #brightest_30.to_csv('GW200129_tile'+str(circlecenter[1])+'_stars.txt', index=None, sep=' ', mode='w')
    random_30.to_csv('GW200129_tile'+str(circlecenter[1])+'_stars.txt', index=None, sep=' ', mode='w')
    