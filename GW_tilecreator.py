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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
centers_190814 = [(12.96650635094611, -24.625),
 (13.183012701892219, -25.0),
 (12.96650635094611, -25.375),
 (13.399519052838329, -25.375),
 (12.53349364905389, -24.625),
 (12.53349364905389, -25.375),
 (12.316987298107781, -25.0),
 (12.100480947161671, -24.625),
 (12.75, -24.25)]

#this first section calculates all the conditions for the first circle which is given in center_190814

#truth condition just filters all the objects in the txt file (which has been sent over slack) to be within each circle/tile
truth = (np.sin(df_190814["DEC"]*np.pi/180)*(df_190814["RA"]-center_190814[0]))**2 + (df_190814["DEC"]-center_190814[1])**2 < radius**2
inside_circ1 = df_190814[truth]

#calculates total prob inside first circle, sorts based on smalles mag_r, grabs the brkghtest 500 and calculates probability of top 500 brightest
totalprob = inside_circ1['PROB'].sum()
sort_mag_r = inside_circ1.sort_values(by='MAG_R', ascending=True)
top500 = sort_mag_r[:500]
top500_prob = top500['PROB'].sum()

#this second section calculates the same things as above but now in a loop that iterates through centers_190814
top500s_190814 = [top500]
areaprob_190814 = [totalprob]
top500prob_190814 = [top500_prob]
drop = top500
for circlecenter in centers_190814:
    cutdown = pd.concat([df_190814, drop])
    cutdown_190817 = cutdown[cutdown.duplicated(keep=False)==False]
    truth = (np.sin(df_190814["DEC"]*np.pi/180)*(cutdown_190817["RA"]-circlecenter[0]))**2 + (cutdown_190817["DEC"]-circlecenter[1])**2 < radius**2
    inside_circ1 = cutdown_190817[truth]
    areaprob_190814.append(inside_circ1['PROB'].sum())
    sort_mag_r = inside_circ1.sort_values(by='MAG_R', ascending=True)
    top500 = sort_mag_r[:500]
    top500s_190814.append(top500)
    top500prob_190814.append(top500['PROB'].sum())
    drop = pd.concat([drop, top500])


#this last bit just puts all the information in a txt file
centers_190814.insert(0,center_190814) #just combines the first circle with the list of the other circles
ras = [x[0] for x in centers_190814]
decs = [x[1] for x in centers_190814]
d = {'RA': ras, 'DEC': decs, 'PROB': top500prob_190814}
data_190814 = pd.DataFrame(data=d)
data_190814.to_csv('tiledata_190814.txt', index=None, sep=' ', mode='w')# %%

print(data_190814)
print("Total prob", sum(top500prob_190814))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#this one hasnt been shifted, the first circle is centered around NGC4993
df_170817 = pd.read_csv('GW170817.txt', delim_whitespace=True, header= 0, index_col= False)
df_170817 = df_170817.apply(pd.to_numeric, errors='coerce')


center_170817 = [196.89,-24.0] #NGC4993 coords
circles = 10
radius = 1

#has a scale of 0.2 and prob total of 249.7360197515219, we want all circles to be within 3 degree diameter
centers_170817 = [(-162.9367949192431+360, -23.7),
 (-162.76358983848624+360, -24.0),
 (-162.9367949192431+360, -24.3),
 (-162.59038475772934+360, -24.3),
 (-163.28320508075691+360, -23.7),
 (-163.28320508075691+360, -24.3),
 (-163.4564101615138+360, -24.0)]

#truth condition just filters all the objects in the txt file (which has been sent over slack) to be within each circle/tile
truth = (np.sin(df_170817["DEC"]*np.pi/180)*(df_170817["RA"]-center_170817[0]))**2 + (df_170817["DEC"]-center_170817[1])**2 < radius**2
inside_circ1 = df_170817[truth]

#calculates total prob inside first circle, sorts based on smalles mag_r, grabs the brkghtest 500 and calculates probability of top 500 brightest
totalprob = inside_circ1['PROB'].sum()
sort_mag_r = inside_circ1.sort_values(by='MAG_R', ascending=True)
top500 = sort_mag_r[:500]
top500_prob = top500['PROB'].sum()

#this second section calculates the same things as above but now in a loop that iterates through centers_190814
top500s_170817 = [top500]
areaprob_170817 = [totalprob]
top500prob_170817 = [top500_prob]
drop = top500 #drops the brightest 500 from the list so they dont get picked up more than once
for circlecenter in centers_170817:
    cutdown = pd.concat([df_170817, drop])
    cutdown_170817 = cutdown[cutdown.duplicated(keep=False)==False]
    truth = (np.sin(df_170817["DEC"]*np.pi/180)*(cutdown_170817["RA"]-circlecenter[0]))**2 + (cutdown_170817["DEC"]-circlecenter[1])**2 < radius**2
    inside_circ1 = cutdown_170817[truth]
    areaprob_170817.append(inside_circ1['PROB'].sum())
    sort_mag_r = inside_circ1.sort_values(by='MAG_R', ascending=True)
    top500 = sort_mag_r[:500]
    top500s_170817.append(top500)
    top500prob_170817.append(top500['PROB'].sum())
    drop = pd.concat([drop, top500])
    
#creating txt file
centers_170817.insert(0,center_170817)
ras = [x[0] for x in centers_170817]
decs = [x[1] for x in centers_170817]
d = {'RA': ras, 'DEC': decs, 'PROB': top500prob_170817}
data_170817 = pd.DataFrame(data=d)
data_170817.to_csv('tiledata_170817.txt', index=None, sep=' ', mode='w')

print("Total prob", sum(top500prob_170817))
print(data_170817)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#this one has been shifted Centershift = [-0.3, 0.0] to maximise probability
df_200129 = pd.read_csv('GW200129.txt', delim_whitespace=True, header= 0, index_col= False)
df_200129 = df_200129.apply(pd.to_numeric, errors='coerce')

center_200129 = [318.36951858, 5.16587679] #highest prob pixel, cant use as first tile since this one has been shifted
#center_200129 = (-41.92999999999999+360, 5.17) #new first tile which has been shifted by Centershift = [-0.3, 0.0]

circles = 4
radius = 1

#i tried a few different tiling of circles, with different scales (which change how close together the circles are/how much overlap)

#scale of 0.5 gives top 500 prob of 1507.488530661799
""" centers = [(-41.496987298107776+360, 5.92),
 (-41.06397459621555+360, 5.17),
 (-41.496987298107776+360, 4.42)] """

#scale of 0.35 gives top 500 prob of 1514.3499571428774
""" centers = [(-41.62689110867544+360, 5.695),
 (-41.32378221735089+360, 5.17),
 (-41.62689110867544+360, 4.645)] """

#scale of 0.6 gives top 500 prob of 1497.3575719363153
""" centers = [(-41.41038475772933+360, 6.07),
 (-40.890769515458665+360, 5.17),
 (-41.41038475772933+360, 4.27)] """

#scale of 0.3 gives top 500 prob of 1517.8744870421817
""" centers = [(-41.67019237886466+360, 5.62),
 (-41.41038475772933+360, 5.17),
 (-41.67019237886466+360, 4.72)]
 """
#scale of 0.15 and get rid of shift 1530.2210903405496
""" centers = [(-41.50009618943233+360, 5.395),
 (-41.370192378864665+360, 5.17),
 (-41.50009618943233+360, 4.945)] """
 
#each tile on top of each other no shift gives prob 500 1544.2571668596838. This was the greatest probability i found
centers_200129 = [(318.36951858, 5.16587679),
 (318.36951858, 5.16587679),
 (318.36951858, 5.16587679)]

#truth condition just filters all the objects in the txt file (which has been sent over slack) to be within each circle/tile
truth = (np.sin(df_200129["DEC"]*np.pi/180)*(df_200129["RA"]-center_200129[0]))**2 + (df_200129["DEC"]-center_200129[1])**2 < radius**2
inside_circ1 = df_200129[truth]

#calculates total prob inside first circle, sorts based on smalles mag_r, grabs the brkghtest 500 and calculates probability of top 500 brightest
totalprob = inside_circ1['PROB'].sum()
sort_mag_r = inside_circ1.sort_values(by='MAG_R', ascending=True)
top500 = sort_mag_r[:500]
top500_prob = top500['PROB'].sum()

#this second section calculates the same things as above but now in a loop that iterates through centers_190814
top500s_200129 = [top500]
areaprob_200129 = [totalprob]
top500prob_200129 = [top500_prob]
drop = top500
for circlecenter in centers_200129:
    cutdown = pd.concat([df_200129, drop])
    cutdown_200129 = cutdown[cutdown.duplicated(keep=False)==False]
    truth = (np.sin(df_200129["DEC"]*np.pi/180)*(cutdown_200129["RA"]-circlecenter[0]))**2 + (cutdown_200129["DEC"]-circlecenter[1])**2 < radius**2
    inside_circ1 = cutdown_200129[truth]
    areaprob_200129.append(inside_circ1['PROB'].sum())
    sort_mag_r = inside_circ1.sort_values(by='MAG_R', ascending=True)
    top500 = sort_mag_r[:500]
    top500s_200129.append(top500)
    top500prob_200129.append(top500['PROB'].sum())
    drop = pd.concat([drop, top500])


#creates txt file
centers_200129.insert(0,center_200129)
ras = [x[0] for x in centers_200129]
decs = [x[1] for x in centers_200129]
d = {'RA': ras, 'DEC': decs, 'PROB': top500prob_200129}
data_200129 = pd.DataFrame(data=d)
data_200129.to_csv('tiledata_200129.txt', index=None, sep=' ', mode='w')

print(data_200129)
print("Total prob", sum(top500prob_200129))

