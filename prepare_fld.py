# Routines to prepare FLD files for AAT observing. Takes a set of targets, standards, guide stars (fiducials) and sky fibre positions
# Splits and prioritises all the targets in each field, and then puts writes the FLD files with appropriate formatting

import re
import numpy as np
import scipy as sp
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import date as dt
from astropy import units as u
from scipy.optimize import minimize
from astropy.coordinates import Angle

def format_sexagesimal(ra, dec):
	RA = Angle(ra, unit=u.deg)
	Dec = Angle(dec, unit=u.deg)
	replace_ra = False
	replace_dec = False
	if round(Dec.dms[2], 1) == 60.0:
		replace_dec = True
	if round(RA.hms[2], 2) == 60.00:
		replace_ra = True
	negate = False
	if Dec.dms[0] < 0:
		negate = True
		dec *= -1
		Dec = Angle(dec, unit=u.deg)
	return (
		f"{RA.hms[0]:02.0f} {RA.hms[1]:02.0f} {59.99 if replace_ra else f'{RA.hms[2]:05.2f}'} "
		f"{'-' if negate else '+'}{Dec.dms[0]:02.0f} "
		f"{Dec.dms[1]:02.0f} {59.9 if replace_dec else f'{Dec.dms[2]:04.1f}'}"
	)

def format_line(ID, coords, ttype, priority, mag, name, pmra=0, pmdec=0):
	return (
		f"{ID} {coords} {ttype} {priority} {mag:05.2f} 0 {pmra:.4f} {pmdec:.4f} {name}"
	)

# Reads in a set of targets. Assigns them all the same priority as to do otherwise (i.e., prioritising based on magnitude) 
# makes interpreting the selection function of our observations more difficult.
def get_field(field, tile):

	# Read in the list of tile centres and return the centre
	centres = {}
	with open(f"./{field}/{field}_centres.txt", 'r') as f:
		lines = f.readlines()
		for ln in lines:
			lnsplit = re.split(r"[(,)]", ln)
			centres[lnsplit[0].split("Tile ")[1][:-1]] = [float(lnsplit[1]), float(lnsplit[2])]
	centres = pd.DataFrame.from_dict(centres, orient='index', columns =['RA', 'DEC'])

	targets = pd.read_csv(f"./{field}/Targets/{field}_tile{tile}_targets.txt", delim_whitespace=True)
	targets["PRIORITY"] = 8

	# Return dataframe containing targets
	return centres.loc[str(tile)], targets


# Reads in the guide and sky fibre files, gives appropriate tag and priorities
def get_guides_sky(field, tile, star_folder, star_name):

	# Read in the guide star file and isolate those in the localisation region
	guides = pd.read_csv( f"./{field}/Stars{star_folder}/{field}_tile{tile}_stars_{star_name}.txt", delim_whitespace=True)

	# Read in the sky fibres file and get sky coordinates in the localisation region. Drop the first row as it contains header rubbish
	sky = pd.read_csv( f"./{field}/SkyFibres/{field}_tile{tile}_sky.txt", delim_whitespace=True).iloc[1:]

	return guides, sky

def write_fld(obs_date, field, tile, centre, targets, guides, sky):

	guide_min, guide_max = guides["Vmag"].min(), guides["Vmag"].max()

	header = (
		f"* Primary Target: {field}\n"
		f"*\n"
		f"* Maddy Cross-Parkin, Cullan Howlett, Tamara Davis\n"
		f"* Last modified: {dt.today()}\n"
		f"*\n"
		f"* gals R <= 20.0 mag, guide stars {guide_min} < V < {guide_max}\n"
		f"LABEL Primary target {field}_tile{tile}\n"
		f"UTDATE {obs_date}\n"
		f"CENTRE {format_sexagesimal(centre['RA'], centre['DEC'])}\n"
		f"EQUINOX J2000.0\n"
		f"PROPER_MOTIONS\n"
		f"* End of header info\n"
		f"* Start of input data\n"
		f"* TargetName(unique for header) RA(hh mm ss.ss) Dec([+/-]dd mm ss.s) TargetType(P,F,S) Priority(9 is highest) Magnitude(mm.mm) 0 pmRA pmDec TargetName"
	)
	strings = [header]

	# Main targets
	for i, row in targets.iterrows():
	    coords = format_sexagesimal(float(row["RA"]), float(row["DEC"]))
	    strings.append(
	        format_line(
	            f"TG{i}",
	            coords,
	            "P",
	            int(row["PRIORITY"]),
	            row["MAG_R"],
	            "Program",
	        )
	    )

	# Guide stars
	for i, row in guides.iterrows():
	    coords = format_sexagesimal(float(row["RA"]), float(row["DEC"]))
	    strings.append(
	        format_line(
	            row["UCAC4"],
	            coords,
	            "F",
	            5,
	            float(row["Vmag"]),
	            "guide",
	            pmra=float(row["pmRA"] / 1000.0) / np.cos(float(row["DEC"])*np.pi/180.0),  # (RA)mas/yr -> "/yr
	            pmdec=float(row["pmDE"] / 1000.0),  # mas/yr -> "/yr
	        )
	    )

	# Sky fibres
	for i, row in sky.iterrows():
	    coords = format_sexagesimal(float(row["_RAJ2000"]), float(row["_DEJ2000"]))
	    strings.append(
	        format_line(
	            f"S{i}",
	            coords,
	            "S",
	            5,
	            23,
	            "sky",
	        )
	    )

	# Write out the FLD file
	with open(f"./{field}/{field}_tile{tile}.fld", "w") as out:
	    for string in strings:
	        out.write(string)
	        out.write("\n")

	return

def prepare_july23():

	obs_date = ["2023 07 07", "2023 07 08"]

	# A list of GW tiles we want to run this for as a set of
	# sub-dictionaries indexed first by observation date, then
	# by field name
	fields = {"2023 07 07": {"GW170817": [0, 1, 2, 3],
	 		  		 		 "GW190814": [0, 1, 2, 3, 4], 
	 		  		 		 "GW200129": [0, 1]},
	 		  "2023 07 08": {"GW170817": [4, 5, 6, 7],
	 		  		 		 "GW190814": [5, 6, 7, 8, 9], 
	 		  		 		 "GW200129": [2, 3]}}

	# Loop over fields and central coordinates, get the data and write the fld file
	for obs_date in fields:
		for field in fields[obs_date]:
				for tile in fields[obs_date][field]:
					print(obs_date, field, tile)
					centre, targets = get_field(field, tile)
					guides, sky = get_guides_sky(field, tile, "Retracted", "retract")
					write_fld(obs_date, field, tile, centre, targets, guides, sky)

def prepare_oct23():

	obs_date = ["2023 10 21"]

	# A list of GW tiles we want to run this for as a set of
	# sub-dictionaries indexed first by observation date, then
	# by field name
	fields = {"2023 10 21": {"GW190814_v2": ["A", "B", "C", "D", "E", "F"], 
	 		  		 		 "S191204r": ["A", "B", "C"]}}

	# Loop over fields and central coordinates, get the data and write the fld file
	for obs_date in fields:
		for field in fields[obs_date]:
				for tile in fields[obs_date][field]:
					print(obs_date, field, tile)
					centre, targets = get_field(field, tile)
					guides, sky = get_guides_sky(field, tile, "", "inspected")
					write_fld(obs_date, field, tile, centre, targets, guides, sky)

if __name__ == "__main__":

	#prepare_july23()
	prepare_oct23()



