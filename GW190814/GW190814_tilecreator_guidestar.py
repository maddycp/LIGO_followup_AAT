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

filter_ra_dec_bigblock =  (df_result['RA'] >0.0) & (df_result['RA'] < 30.0) &  (df_result['DEC'] > -30.0) & (df_result['DEC'] < -15.0) 
#filter_ra_dec_smallblock = (df_result['RA'] >18.0) & (df_result['RA'] < 27.0) &  (df_result['DEC'] < -29.0) & (df_result['DEC'] > -35) 


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
#allbgsmask2 = basic_cuts & bgs_magcut & bgs_fibremag_cut & filter_gaia & filter_fracmasked & filter_bgsmask & filter_fracflux & filter_fracin & filter_gaia_bright &filter_ra_dec_smallblock


filtered_data1 = df_result.loc[allbgsmask1]
#filtered_data2 = df_result.loc[allbgsmask2]

#filtered_data = pd.concat([filtered_data1,filtered_data2])
filtered_data = filtered_data1

filtered_ra = filtered_data['RA']
filtered_dec = filtered_data['DEC']

coordinates = SkyCoord([(float(x))*u.deg for x in (filtered_ra.tolist())],[float(x)*u.deg for x in filtered_dec.tolist()])#,[float(x) for x in df_result[2][1:].tolist()])
result = crossmatch(sky_map, coordinates)
searched_prob190814 = result.searched_prob[result.searched_prob < 0.65]

cutdown = filtered_data[result.searched_prob < 0.65]
print(len(cutdown))
center = [12.65524341, -24.84166719]

radius_search = 10.0
cutdown['PROB'] = 1-searched_prob190814
cutdown['LOCAL'] = ((np.cos(center[1]*np.pi/180)*(cutdown["RA"]-center[0]))**2 + (cutdown["DEC"]-center[1])**2 < radius_search**2)
cutdown['SEP'] = np.sqrt((np.cos(cutdown["DEC"]*np.pi/180)*(cutdown["RA"]-center[0]))**2 + (cutdown["DEC"]-center[1])**2)
cutdown = cutdown[cutdown['LOCAL'] == True]
localised_Sorted190814 = cutdown.sort_values(by=['PROB'], ascending = 0)

labels= ["GW190814"]

localised_Sorted190814.to_csv(labels[0]+'.txt', index=None, sep=' ', mode='w')

#0.1 degree

df_190814 = pd.read_csv('GW190814.txt', delim_whitespace=True, header= 0, index_col= False)
df_190814 = df_190814.apply(pd.to_numeric, errors='coerce')

ra_grid = np.linspace(min(df_190814['RA']), max(df_190814['RA']), 50)
dec_grid = np.linspace(min(df_190814['DEC']), max(df_190814['DEC']), 50)
ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
radius_tile = 1
drop = df_190814.drop(df_190814.index)

center_best = []
number_tiles = 10
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for tile_num in zip(labels,range(number_tiles)):
    areaprob_190814 = 0.0
    for center_x, center_y in zip(ra_mesh.flatten(), dec_mesh.flatten()):
        cutdown= pd.concat([df_190814, drop])
        cutdown_190814 = cutdown[cutdown.duplicated(keep=False)==False]
        truth = (np.cos(center_y*np.pi/180)*(cutdown_190814["RA"]-center_x))**2 + (cutdown_190814["DEC"]-center_y)**2 < radius_tile**2
        inside_circ = cutdown_190814[truth]
        if inside_circ['PROB'].sum() > areaprob_190814:
            areaprob_190814 = (inside_circ['PROB'].sum())
            try:
                top500 = inside_circ.sample(500)
            except: 
                top500 = inside_circ.sample(len(inside_circ))
            top500 = top500[top500.duplicated(keep=False)==False]
            center_x_save = center_x
            center_y_save = center_y
    top500tosave = pd.concat([top500, pd.Series()], ignore_index=True)
    top500tosave.to_csv('GW190814_tile'+str(tile_num[1])+'_targets.txt', index=None, sep=' ', mode='w')
    center_best.append((center_x_save, center_y_save))
    drop = pd.concat([drop, top500])
    
tile = ['Tile ' + str(i) for i in range(number_tiles)]

with open("GW190814_centres.txt", "w") as output:
    for x, y in zip(tile,center_best):
        output.write(str(x) + '\t' +str(y) + "\n")

Vizier.ROW_LIMIT = -1
centers_190814 = center_best


label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for circlecenter in zip(centers_190814, label):
    result = Vizier.query_constraints(catalog='I/322A/out', of='0', db='0', pmRA = '-15 .. 15', pmDE = '-15 .. 15',
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
    random_30 = pd.DataFrame(data).sample(30)
    #sort_mag_r.sort_values(by='MAG_R', ascending=True)
    #brightest_30 = sort_mag_r[:30]
    #brightest_30.to_csv('GW190814_tile'+str(circlecenter[1])+'_stars.txt', index=None, sep=' ', mode='w')
    random_30.to_csv('GW190814_tile'+str(circlecenter[1])+'_stars.txt', index=None, sep=' ', mode='w')


