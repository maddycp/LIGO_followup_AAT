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
#the list of centers for each GW were determined from another script

#this one has been shifted by Centershift = [0.0, 0.35] which maximises the probability
df_190814 = pd.read_csv('GW190814.txt', delim_whitespace=True, header= 0, index_col= False)
df_190814 = df_190814.apply(pd.to_numeric, errors='coerce')

#center_190814 = [12.75,-25.35] #not using this value because its been shifted so the first circle not centered here
center_190814 = (12.75, -25.0) #shifted value, first center tile

#number of tiles and radius
circles = 10
radius = 1

#centers for GW190814 which are generated from another script (can send if needed)


#this first section calculates all the conditions for the first circle which is given in center_190814

#truth condition just filters all the objects in the txt file (which has been sent over slack) to be within each circle/tile
truth_0 = (np.cos(center_190814[1]*np.pi/180)*(df_190814["RA"]-center_190814[0]))**2 + (df_190814["DEC"]-center_190814[1])**2 < radius**2
inside_circ_0 = df_190814[truth_0]

#calculates total prob inside first circle, sorts based on smalles mag_r, grabs the brkghtest 500 and calculates probability of top 500 brightest
totalprob_0 = inside_circ_0['PROB'].sum()
sort_mag_r_0 = inside_circ_0.sort_values(by='MAG_R', ascending=True)
top500tosave_0 = sort_mag_r_0[:500]
top500_prob_0 = top500tosave_0['PROB'].sum()

#this second section calculates the same things as above but now in a loop that iterates through centers_190814
top500s_190814 = [top500tosave_0]
areaprob_190814 = [totalprob_0]
top500prob_190814 = [top500_prob_0]
drop = top500tosave_0
top500tosave_0.to_csv('GW190814_tile0_targets.txt', index=None, sep=' ', mode='w')

centers_190814 = [(12.96650635094611, -24.625),
 (13.183012701892219, -25.0),
 (12.96650635094611, -25.375),
 (13.399519052838329, -25.375),
 (12.53349364905389, -24.625),
 (12.53349364905389, -25.375),
 (12.316987298107781, -25.0),
 (12.100480947161671, -24.625),
 (12.75, -24.25)]

label = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for circlecenter in zip(centers_190814, label):
    cutdown = pd.concat([df_190814, drop])
    cutdown_190814 = cutdown[cutdown.duplicated(keep=False)==False]
    truth = (np.cos(circlecenter[0][1]*np.pi/180)*(cutdown_190814["RA"]-circlecenter[0][0]))**2 + (cutdown_190814["DEC"]-circlecenter[0][1])**2 < radius**2
    inside_circ = cutdown_190814[truth]
    areaprob_190814.append(inside_circ['PROB'].sum())
    sort_mag_r = inside_circ.sort_values(by='MAG_R', ascending=True)
    top500 = sort_mag_r[:500]
    print(top500)
    top500tosave = pd.concat([top500, pd.Series()], ignore_index=True)
    #top500tosave = pd.concat([top500tosave, top500], ignore_index=True)
    top500s_190814.append(top500)
    top500prob_190814.append(top500['PROB'].sum())
    drop = pd.concat([drop, top500])
#    print(circlecenter[1])
    top500tosave.to_csv('GW190814_tile'+str(circlecenter[1])+'_targets.txt', index=None, sep=' ', mode='w')

#this last bit just puts all the information in a txt file
centers_190814.insert(0,center_190814) #just combines the first circle with the list of the other circles
ras = [x[0] for x in centers_190814]
decs = [x[1] for x in centers_190814]
d = {'RA': ras, 'DEC': decs, 'PROB': top500prob_190814}
data_190814 = pd.DataFrame(data=d)
data_190814.to_csv('GW190814_tilecentre.txt', index=None, sep=' ', mode='w')# %%

print("Total prob", sum(top500prob_190814))
print(data_190814)


radius = 1
centers_190814 = [(12.75, -25.0), (12.96650635094611, -24.625),
 (13.183012701892219, -25.0),
 (12.96650635094611, -25.375),
 (13.399519052838329, -25.375),
 (12.53349364905389, -24.625),
 (12.53349364905389, -25.375),
 (12.316987298107781, -25.0),
 (12.100480947161671, -24.625),
 (12.75, -24.25)]

Vizier.ROW_LIMIT = -1

label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for circlecenter in zip(centers_190814, label):
    result = Vizier.query_constraints(catalog='I/322A/out', of='0', db='0', pmRA = '< 10', pmDE = '< 10',
                                 Vmag = '11 .. 13', RAJ2000 = (str(circlecenter[0][0]-1)) + ' .. ' + str(circlecenter[0][0]+1), 
                                 DEJ2000 = str(circlecenter[0][1]-1) + ' .. ' + str(circlecenter[0][1]+1))
    for table_name in result.keys():
        query_result = result[table_name]
        #print(query_result)
    query_result['LOCAL'] = ((np.cos(centers_190814[0][1]/(180/np.pi))*(query_result["RAJ2000"]-centers_190814[0][0]))**2 + (query_result["DEJ2000"]-centers_190814[0][1])**2 < radius**2)
    localised_query = query_result[query_result['LOCAL'] == True]
    UCAC4 = []
    RA = []
    DEC = []
    MAG_R = []
    for i in localised_query:
        UCAC4.append(i['UCAC4'])
        RA.append(i['RAJ2000'])
        DEC.append(i['DEJ2000'])
        MAG_R.append(i['rmag'])
    data = {'UCAC4':UCAC4, 'RA':RA, 'DEC':DEC, 'MAG_R':MAG_R}
    sort_mag_r = pd.DataFrame(data)
    sort_mag_r.sort_values(by='MAG_R', ascending=True)
    brightest_30 = sort_mag_r[:30]
    brightest_30.to_csv('GW190814_tile'+str(circlecenter[1])+'_stars.txt', index=None, sep=' ', mode='w')
