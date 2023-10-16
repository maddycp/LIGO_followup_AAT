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

GW190814tile0 = pd.read_csv('GW190814_tile0_runGW01_v18_08.txt', sep=',', header = 0)
GW190814tile0.columns = GW190814tile0.columns.str.strip()
GW190814tile0['RA'] = GW190814tile0['RA'].multiply(180/np.pi)
GW190814tile0['DEC'] = GW190814tile0['DEC'].multiply(180/np.pi)
GW190814tile0 = GW190814tile0[GW190814tile0['QOP']>2]


GW190814tile1 = pd.read_csv('GW190814_tile1_runGW01_v18_07.txt', sep=',', header = 0)
GW190814tile1.columns = GW190814tile1.columns.str.strip()
GW190814tile1['RA'] = GW190814tile1['RA'].multiply(180/np.pi)
GW190814tile1['DEC'] = GW190814tile1['DEC'].multiply(180/np.pi)
GW190814tile1 = GW190814tile1[GW190814tile1['QOP']>2]


GW190814tile2 = pd.read_csv('GW190814_tile2_runGW01_v18_08.txt', sep=',', header = 0)
GW190814tile2.columns = GW190814tile2.columns.str.strip()
GW190814tile2['RA'] = GW190814tile2['RA'].multiply(180/np.pi)
GW190814tile2['DEC'] = GW190814tile2['DEC'].multiply(180/np.pi)
GW190814tile2 = GW190814tile2[GW190814tile2['QOP']>2]


GW190814tile3 = pd.read_csv('GW190814_tile3_runGW01_v18_08.txt', sep=',', header = 0)
GW190814tile3.columns = GW190814tile3.columns.str.strip()
GW190814tile3['RA'] = GW190814tile3['RA'].multiply(180/np.pi)
GW190814tile3['DEC'] = GW190814tile3['DEC'].multiply(180/np.pi)
GW190814tile3 = GW190814tile3[GW190814tile3['QOP']>2]

already_observed = pd.concat([GW190814tile0, GW190814tile1, GW190814tile2, GW190814tile3])

dr10_alldat = pd.read_csv('GW190814.txt', delim_whitespace=True, header= 0, index_col= False)

#find entries the same
already_observed = SkyCoord(ra = already_observed['RA'], dec = already_observed['DEC'], unit = 'deg')
DR10coords = SkyCoord(ra = dr10_alldat['RA'], dec = dr10_alldat['DEC'], unit = 'deg') 
        
nearestneighbor, d2d, _ = already_observed.match_to_catalog_sky(DR10coords) 
dr10matched_observed = dr10_alldat.iloc[nearestneighbor]

d2ds = []
for i in range(len(d2d)):
    d2ds.append(d2d[i].degree)

add_sky = pd.DataFrame(np.array(d2ds))
add_sky = add_sky.set_index(dr10matched_observed.index)
dr10matched_observed_onsky = pd.concat([dr10matched_observed, add_sky], axis = 1)
dr10matched_observed_onsky = dr10matched_observed_onsky.rename(columns={0:"OnSkyDiff"})

dr10matched_observed_onsky.drop(dr10matched_observed_onsky[dr10matched_observed_onsky['OnSkyDiff'] >= 0.0002].index, inplace = True)

plt.hist(dr10matched_observed_onsky['OnSkyDiff'], bins = 10, edgecolor="black", color = 'pink')
plt.xlabel('On-Sky Distance Between Targets')
plt.ylabel('Number of Occurances')

all_dr10_data_observed = pd.concat([dr10_alldat, dr10matched_observed_onsky])

dr10_for_AAT2 = all_dr10_data_observed.drop_duplicates(subset = ['RA', 'DEC', 'FLUX_R'], keep=False)



dr10_for_AAT2.to_csv('GW190814_for_AAT2.txt', index=None, sep=' ', mode='w')


#%%
###########################################

ra_grid = np.linspace(min(dr10_for_AAT2['RA']), max(dr10_for_AAT2['RA']), 50)
dec_grid = np.linspace(min(dr10_for_AAT2['DEC']), max(dr10_for_AAT2['DEC']), 50)
ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
radius_tile = 1
drop = dr10_for_AAT2.drop(dr10_for_AAT2.index)

center_best = []
number_tiles = 6
#labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
labels = ['A', 'B', 'C', 'D', 'E', 'F'] 
for tile_num in zip(labels,range(number_tiles)):
    areaprob_190814 = 0.0
    for center_x, center_y in zip(ra_mesh.flatten(), dec_mesh.flatten()):
        cutdown= pd.concat([dr10_for_AAT2, drop])
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
    top500tosave.to_csv('GW190814_tile_'+str(tile_num[0])+'_targets.txt', index=None, sep=' ', mode='w')
    center_best.append((center_x_save, center_y_save))
    drop = pd.concat([drop, top500])
    
tile = ['Tile ' + str(i) for i in range(number_tiles)]

with open("GW190814_AAT2_centres.txt", "w") as output:
    for x, y in zip(tile,center_best):
        output.write(str(x) + '\t' +str(y) + "\n")

#%%
#############################GUIDE STARS###################################################
Vizier.ROW_LIMIT = -1
centers_190814 = center_best


label = ['A', 'B', 'C', 'D', 'E', 'F'] 
for circlecenter in zip(centers_190814, label):
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
        random_30 =  pd.DataFrame(data).sample(32)
    except: 
        random_30 = pd.DataFrame(data).sample(len(pd.DataFrame(data)))
    #sort_mag_r.sort_values(by='MAG_R', ascending=True)
    #brightest_30 = sort_mag_r[:30]
    #brightest_30.to_csv('GW190814_tile'+str(circlecenter[1])+'_stars.txt', index=None, sep=' ', mode='w')
    random_30.to_csv('GW190814_tile'+str(circlecenter[1])+'_stars.txt', index=None, sep=' ', mode='w')


#%%
####################################################DARK FIBRES################################################

skyA = pd.read_csv('GW190814_tileA_sky.txt', delim_whitespace=True, index_col= False)
skyA = skyA.drop_duplicates()


skyB = pd.read_csv('GW190814_tileB_sky.txt', delim_whitespace=True, index_col= False)
skyB = skyB.drop_duplicates()

skyC = pd.read_csv('GW190814_tileC_sky.txt', delim_whitespace=True, index_col= False)
skyC = skyC.drop_duplicates()


skyD = pd.read_csv('GW190814_tileD_sky.txt', delim_whitespace=True, index_col= False)
skyD = skyD.drop_duplicates()


skyE = pd.read_csv('GW190814_tileE_sky.txt', delim_whitespace=True, index_col= False)
skyE = skyE.drop_duplicates()

skyF = pd.read_csv('GW190814_tileF_sky.txt', delim_whitespace=True, index_col= False)
skyF = skyF.drop_duplicates()


skyA.to_csv('GW190814_tileA_sky.txt', index=None, sep=' ', mode='w')
skyB.to_csv('GW190814_tileB_sky.txt', index=None, sep=' ', mode='w')
skyC.to_csv('GW190814_tileC_sky.txt', index=None, sep=' ', mode='w')
skyD.to_csv('GW190814_tileD_sky.txt', index=None, sep=' ', mode='w')
skyE.to_csv('GW190814_tileE_sky.txt', index=None, sep=' ', mode='w')
skyF.to_csv('GW190814_tileF_sky.txt', index=None, sep=' ', mode='w')


#%%

###################################concat darks and stars#######################################

starsA_191204r = pd.read_csv('GW190814_2/GW190814_tileA_stars_inspected.txt', delim_whitespace=True, header= 0, index_col= False)
starsB_191204r = pd.read_csv('GW190814_2/GW190814_tileB_stars_inspected.txt', delim_whitespace=True, header= 0, index_col= False)
starsC_191204r = pd.read_csv('GW190814_2/GW190814_tileC_stars_inspected.txt', delim_whitespace=True, header= 0, index_col= False)

GW190814allstars = pd.concat([starsA_191204r, starsB_191204r, starsC_191204r])
GW190814allstars.to_csv('GW190814_allstars.txt', index=None, sep=' ', mode='w')


fibresA_191204r = pd.read_csv('GW190814_2/GW190814_tileA_sky.txt', delim_whitespace=True, header= 0, index_col= False)
fibresB_191204r = pd.read_csv('GW190814_2/GW190814_tileB_sky.txt', delim_whitespace=True, header= 0, index_col= False)
fibresC_191204r = pd.read_csv('GW190814_2/GW190814_tileC_sky.txt', delim_whitespace=True, header= 0, index_col= False)

GW190814alldarkfibres = pd.concat([fibresA_191204r, fibresB_191204r, fibresC_191204r])
GW190814alldarkfibres.to_csv('GW190814_alldarkfibres.txt', index=None, sep=' ', mode='w')




















# %%

##############################################RECYCLE TARGETS########################################################################################

#input the txt file of the targets that were used (with RA, DEC)
#need to open .lis files as txt and remove the top couple of lines of junk 
df0 = pd.read_csv('GW170817_tile0_p1.lis', delim_whitespace=True, header= None)

#need to input tile too in order to see what was observed so it can be removed
tile0 =  pd.read_csv('GW170817_tile0_targets.txt', delim_whitespace=True)


tile_targets = tile0 #GW170817tile0#pd.concat([tile0, tile1])
unused_targets = df0 #pd.concat([df0, df1])

unused_targets.rename( columns={0:'Name', 1:'RA_hours', 2:'RA_mins', 3:'RA_secs', 4:'DEC_hours', 5:'DEC_mins', 6:'DEC_secs', 13:'Type'}, inplace=True )

unused_targets.drop(unused_targets[unused_targets['Type'] != 'sky'].index, inplace=True)
print(unused_targets)


unused_targets['RA_mins'] = unused_targets['RA_mins'].apply(lambda x: float(x))
unused_targets['DEC_mins'] = unused_targets['DEC_mins'].apply(lambda x: float(x))

unused_targets['RA_secs'] = unused_targets['RA_secs'].apply(lambda x: float(x))
unused_targets['DEC_secs'] = unused_targets['DEC_secs'].apply(lambda x: float(x))

unused_targets['RA_hours'] = unused_targets['RA_hours']
unused_targets['DEC_hours'] = unused_targets['DEC_hours']

unused_targets['RA_mins'] = unused_targets['RA_mins'].multiply(1.0/60.0)
unused_targets['DEC_mins'] = unused_targets['DEC_mins'].multiply(1.0/60.0)

unused_targets['RA_secs'] = unused_targets['RA_secs'].multiply(1.0/3600.0)
unused_targets['DEC_secs'] = unused_targets['DEC_secs'].multiply(1.0/3600.0)

#change to tpye 64 not object
unused_targets["RA_hours"] = pd.to_numeric(unused_targets["RA_hours"])
unused_targets["RA_mins"] = pd.to_numeric(unused_targets["RA_mins"])
unused_targets["RA_secs"] = pd.to_numeric(unused_targets["RA_secs"])

unused_targets["DEC_hours"] = pd.to_numeric(unused_targets["DEC_hours"])
unused_targets["DEC_mins"] = pd.to_numeric(unused_targets["DEC_mins"])
unused_targets["DEC_secs"] = pd.to_numeric(unused_targets["DEC_secs"])


unused_targets = unused_targets.eval('Sum_RA = (RA_hours + RA_mins + RA_secs)*15') 
#MAKE SURE TO CHECK WHETHER THIS NEEDS TO BE PLUS OR MINUS (if dec is negative, make sue to subtract, and add if dec is positive)
unused_targets = unused_targets.eval('Sum_DEC = DEC_hours - DEC_mins - DEC_secs') 



used_targets_skycoord = SkyCoord(ra = unused_targets['Sum_RA'], dec = unused_targets['Sum_DEC'], unit = 'deg')
all_tile_skycoord = SkyCoord(ra = tile_targets['RA'], dec = tile_targets['DEC'], unit = 'deg') 
     
nearestneighbor, d2d, _ = used_targets_skycoord.match_to_catalog_sky(all_tile_skycoord) 
match_observed_with_tiles = tile_targets.iloc[nearestneighbor]


#get the targets we have observed by comparing those we didnt use with the tile
unused_concat_tile = pd.concat([tile_targets, match_observed_with_tiles])

#remove everything that is duplicated
used_targets = unused_concat_tile.drop_duplicates(keep=False)

#crossmatch match_observed_with_tiles with the entire data set, 
# and remove ONLY the targets that we observed so we can get a new GW190814.txt to input into tile creator
dr10_alldat = pd.read_csv('GW170817.txt', delim_whitespace=True, index_col= False)


dr10_concat_usedtargets = pd.concat([dr10_alldat, used_targets])
dr10_new_usedtargetsdeleted = dr10_concat_usedtargets.drop_duplicates(keep=False)

#put unused ones on the end of dr10_for_AAT2_edited
match_observed_with_tiles.to_csv('GW190814_for_AAT2_edited.txt', index=None, sep=' ', mode='w')


######################################################################################################################################

# %%
