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
#info for 170817
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

allbgsmask = basic_cuts & bgs_magcut & filter_gaia & filter_fracmasked & filter_bgsmask & filter_fracflux & filter_fracin & filter_gaia_bright #& filter_ra_dec_bigblock

filtered_data = df_result.loc[allbgsmask]
filtered_ra = filtered_data['RA']
filtered_dec = filtered_data['DEC']
coordinates = SkyCoord([(float(x))*u.deg for x in (filtered_ra.tolist())],[float(x)*u.deg for x in filtered_dec.tolist()])#,[float(x) for x in df_result[2][1:].tolist()])
result = crossmatch(sky_map, coordinates)
searched_prob170817 = result.searched_prob[result.searched_prob < 1]

print(len(filtered_data))
cutdown = filtered_data[result.searched_prob < 1]
print(len(cutdown))

center = [196.8906399999999906,-24.0086059999999968] #NGC4993 coords

radius_search = 2.0
cutdown['PROB'] = 1-searched_prob170817
cutdown['LOCAL'] = ((np.cos(cutdown["DEC"]*np.pi/180)*(cutdown["RA"]-center[0]))**2 + (cutdown["DEC"]-center[1])**2 < radius_search**2)
cutdown['SEP'] = np.sqrt((np.cos(center[1]*np.pi/180)*(cutdown["RA"]-center[0]))**2 + (cutdown["DEC"]-center[1])**2)
cutdown = cutdown[cutdown['LOCAL'] == True]
localised_Sorted170817 = cutdown.sort_values(by=['PROB'], ascending = 0)
labels= ["GW170817"]

localised_Sorted170817.to_csv(labels[0]+'.txt', index=None, sep=' ', mode='w')


#this one hasnt been shifted, the first circle is centered around NGC4993
df_170817 = pd.read_csv('GW170817.txt', delim_whitespace=True, header= 0, index_col= False)
df_170817 = df_170817.apply(pd.to_numeric, errors='coerce')

center_170817 = (-163.11+360, -24.0) #NGC4993 coords, need to add 360 since the bounds go from 0,360 here instead of -180,180 as in field file

circles = 8
radius_tile = 1

#truth condition just filters all the objects in the txt file (which has been sent over slack) to be within each circle/tile
truth_0 = (np.cos(center_170817[1]*np.pi/180)*(df_170817["RA"]-center_170817[0]))**2 + (df_170817["DEC"]-center_170817[1])**2 < radius_tile**2
inside_circ_0 = df_170817[truth_0]

#calculates total prob inside first circle, sorts based on smalles mag_r, grabs the brkghtest 500 and calculates probability of top 500 brightest
totalprob_0 = inside_circ_0['PROB'].sum()
sort_mag_r_0 = inside_circ_0.sort_values(by='MAG_R', ascending=True)
top500tosave_0 = sort_mag_r_0[:500]
top500_prob_0 = top500tosave_0['PROB'].sum()

#this second section calculates the same things as above but now in a loop that iterates through centers_190814
top500s_170817 = [top500tosave_0]
areaprob_170817 = [totalprob_0]
top500prob_170817 = [top500_prob_0]
drop = top500tosave_0
top500tosave_0.to_csv('GW170817_tile0_targets.txt', index=None, sep=' ', mode='w')


#has a scale of 0.4 and red circle radius of 4, has total prob of 395.66756607038616
centers_170817 = [
 (-162.76358983848624+360, -23.4),
 (-162.41717967697247+360, -24.0),
 (-162.76358983848624+360, -24.6),
 (-162.0707695154587+360, -24.6),
 (-163.4564101615138+360, -23.4),
 (-163.4564101615138+360, -24.6),
 (-163.80282032302756+360, -24.0)]

#drops the brightest 500 from the list so they dont get picked up more than once
label = [1, 2, 3, 4, 5, 6, 7]
for circlecenter in zip(centers_170817, label):
    cutdown = pd.concat([df_170817, drop])
    cutdown_170817 = cutdown[cutdown.duplicated(keep=False)==False]
    truth = (np.cos(circlecenter[0][1]*np.pi/180)*(cutdown_170817["RA"]-circlecenter[0][0]))**2 + (cutdown_170817["DEC"]-circlecenter[0][1])**2 < radius_tile**2
    inside_circ = cutdown_170817[truth]
    areaprob_170817.append(inside_circ['PROB'].sum())
    sort_mag_r = inside_circ.sort_values(by='MAG_R', ascending=True)
    top500 = sort_mag_r[:500]
    top500tosave = pd.concat([top500, pd.Series()], ignore_index=True)
    top500s_170817.append(top500)
    top500prob_170817.append(top500['PROB'].sum())
    drop = pd.concat([drop, top500])
    top500tosave.to_csv('GW170817_tile'+str(circlecenter[1])+'_targets.txt', index=None, sep=' ', mode='w')

    
#creating txt file
centers_170817.insert(0,center_170817)
ras = [x[0] for x in centers_170817]
decs = [x[1] for x in centers_170817]
d = {'RA': ras, 'DEC': decs, 'PROB': top500prob_170817}
data_170817 = pd.DataFrame(data=d)
data_170817.to_csv('GW170817_tilecentre.txt', index=None, sep=' ', mode='w')


print("Total prob", sum(top500prob_170817))
print(data_170817)


radius_tile = 1
centers_170817 = [(-163.11+360, -24.0),
 (-162.76358983848624+360, -23.4),
 (-162.41717967697247+360, -24.0),
 (-162.76358983848624+360, -24.6),
 (-162.0707695154587+360, -24.6),
 (-163.4564101615138+360, -23.4),
 (-163.4564101615138+360, -24.6),
 (-163.80282032302756+360, -24.0)]

Vizier.ROW_LIMIT = -1

label = [0, 1, 2, 3, 4, 5, 6, 7]
for circlecenter in zip(centers_170817, label):
    result = Vizier.query_constraints(catalog='I/322A/out', of='0', db='0', pmRA = '< 10', pmDE = '< 10',
                                 Vmag = '11 .. 13', RAJ2000 = str(circlecenter[0][0]-1) + ' .. ' + str(circlecenter[0][0]+1), 
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
    sort_mag_r = pd.DataFrame(data)
    sort_mag_r.sort_values(by='MAG_R', ascending=True)
    brightest_30 = sort_mag_r[:30]
    print(brightest_30)
    brightest_30.to_csv('GW170817_tile'+str(circlecenter[1])+'_stars.txt', index=None, sep=' ', mode='w')


