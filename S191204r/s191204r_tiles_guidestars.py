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
from astroquery.vizier import Vizier
from scipy import interpolate
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
import astropy.units as u
from matplotlib.colors import LinearSegmentedColormap

dr10filtered_s191204r = pd.read_csv('S191204r_filtered.txt', delim_whitespace=True, header= 0, index_col= False)

ra_grid = np.linspace(min(dr10filtered_s191204r['RA']), max(dr10filtered_s191204r['RA']), 50)
dec_grid = np.linspace(min(dr10filtered_s191204r['DEC']), max(dr10filtered_s191204r['DEC']), 50)
ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
radius_tile = 1
drop = dr10filtered_s191204r.drop(dr10filtered_s191204r.index)

center_best = []
number_tiles = 3
#labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
labels = ['A', 'B', 'C'] 
for tile_num in zip(labels,range(number_tiles)):
    areaprob_s191204r = 0.0
    for center_x, center_y in zip(ra_mesh.flatten(), dec_mesh.flatten()):
        cutdown= pd.concat([dr10filtered_s191204r, drop])
        cutdown_s191204r = cutdown[cutdown.duplicated(keep=False)==False]
        truth = (np.cos(center_y*np.pi/180)*(cutdown_s191204r["RA"]-center_x))**2 + (cutdown_s191204r["DEC"]-center_y)**2 < radius_tile**2
        inside_circ = cutdown_s191204r[truth]
        if inside_circ['PROB'].sum() > areaprob_s191204r:
            areaprob_s191204r = (inside_circ['PROB'].sum())
            try:
                top500 = inside_circ.sample(500)
            except: 
                top500 = inside_circ.sample(len(inside_circ))
            top500 = top500[top500.duplicated(keep=False)==False]
            center_x_save = center_x
            center_y_save = center_y
    top500tosave = pd.concat([top500, pd.Series()], ignore_index=True)
    top500tosave.to_csv('GWs191204r_tile_'+str(tile_num[0])+'_targets.txt', index=None, sep=' ', mode='w')
    center_best.append((center_x_save, center_y_save))
    drop = pd.concat([drop, top500])
    
tile = ['Tile ' + str(i) for i in range(number_tiles)]

with open("s191204r_centres.txt", "w") as output:
    for x, y in zip(tile,center_best):
        output.write(str(x) + '\t' +str(y) + "\n")

#%%
Vizier.ROW_LIMIT = -1
centers_s191204r = center_best


label = ['A', 'B', 'C'] 
for circlecenter in zip(centers_s191204r, label):
    result = Vizier.query_constraints(catalog='I/322A/out', of='0', db='0', pmRA = '-25 .. 25', pmDE = '-25 .. 25',
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
    try:
        random_30 =  pd.DataFrame(data).sample(35)
    except: 
        random_30 = pd.DataFrame(data).sample(len(pd.DataFrame(data)))
    #sort_mag_r.sort_values(by='MAG_R', ascending=True)
    #brightest_30 = sort_mag_r[:30]
    #brightest_30.to_csv('GWs191204r_tile'+str(circlecenter[1])+'_stars.txt', index=None, sep=' ', mode='w')
    random_30.to_csv('s191204r_tile'+str(circlecenter[1])+'_stars.txt', index=None, sep=' ', mode='w')



# %%
###################################concat darks and stars#######################################

starsA_191204r = pd.read_csv('S191204r/s191204r_tileA_stars_inspected.txt', delim_whitespace=True, header= 0, index_col= False)
starsB_191204r = pd.read_csv('S191204r/s191204r_tileB_stars_inspected.txt', delim_whitespace=True, header= 0, index_col= False)
starsC_191204r = pd.read_csv('S191204r/s191204r_tileC_stars_inspected.txt', delim_whitespace=True, header= 0, index_col= False)

s191204r_allstars = pd.concat([starsA_191204r, starsB_191204r, starsC_191204r])
s191204r_allstars.to_csv('s191204r_allstars.txt', index=None, sep=' ', mode='w')


fibresA_191204r = pd.read_csv('S191204r/s191204r_tileA_sky.txt', delim_whitespace=True, header= 0, index_col= False)
fibresB_191204r = pd.read_csv('S191204r/s191204r_tileB_sky.txt', delim_whitespace=True, header= 0, index_col= False)
fibresC_191204r = pd.read_csv('S191204r/S191204r_tileC_sky.txt', delim_whitespace=True, header= 0, index_col= False)

s191204r_alldarkfibres = pd.concat([fibresA_191204r, fibresB_191204r, fibresC_191204r])
s191204r_alldarkfibres.to_csv('s191204r_alldarkfibres.txt', index=None, sep=' ', mode='w')












# %%
